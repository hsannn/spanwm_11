import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experiments import role_ablation as cli
from utils.role_ablation import (
    ROLE_SETS, auc, continuation, detection_metrics, final_alignment,
    paired_auc_intervals, sample_seed, summarize_condition,
)


def detection(p=None, n=0, sites=None):
    sites = [] if sites is None else sites
    return dict(p_value=p, score=2. if n else None, num_tested_tokens=n,
                num_green_tokens=n, num_sites=len(sites), sites=sites)


class RoleAblationTests(unittest.TestCase):
    def test_seed_streams_pair_conditions_and_separate_stages(self):
        seeds = {sample_seed(42, i, stage) for i in range(10)
                 for stage in ('draft', 'negative', 'embedding')}
        self.assertEqual(len(seeds), 30)
        self.assertEqual(sample_seed(42, 0, 'embedding'), sample_seed(42, 0, 'embedding'))

    def test_final_alignment_is_distinct_from_local_check(self):
        embedded = [dict(role='NP', anchor=10, anchor_dist=0),
                    dict(role='VP', anchor=30, anchor_dist=1)]
        result = final_alignment(embedded, [dict(anchor=20)])
        self.assertEqual(result['local_aligned_sites'], 2)
        self.assertEqual(result['final_aligned_sites'], 0)
        self.assertEqual(result['final_alignment_flags'], [False, False])
        self.assertFalse(result['fully_verified'])
        self.assertFalse(final_alignment([], [])['fully_verified'])

    def test_ties_and_no_evidence_remain_in_metrics(self):
        empty = detection()
        positive = detection(.001, 4)
        metrics = detection_metrics([positive, empty], [empty, empty], .01)
        self.assertEqual(metrics['AUROC'], .75)
        self.assertEqual(metrics['TPR_at_1pct_FPR'], .5)
        self.assertEqual(metrics['negative_FPR_resolution'], .5)
        self.assertEqual(auc([0, 0], [0, 0]), .5)
        self.assertEqual(auc([1, 2], [0, 0]), 1.)

    def test_paired_identical_conditions_have_zero_difference_interval(self):
        result = paired_auc_intervals([1, 2, -1], [0, 1, -1], [1, 2, -1],
                                      [0, 1, -1], samples=30, seed=4)
        self.assertEqual(result['delta_AUROC_vs_R3'], 0.)
        self.assertEqual(result['delta_AUROC_ci95'], [0., 0.])

    def test_condition_uses_shared_draft_and_scores_only_continuations(self):
        wm = SimpleNamespace()
        site = dict(role='NP', anchor=1, anchor_dist=0)
        wm.last_embedding_info = dict(protocol='test', skipped=False, verified=True,
                                      num_sites=1, num_aligned=1, sites=[site])
        wm.generate_watermarked_text = Mock(return_value='P: edited')
        wm.detect_watermark = Mock(side_effect=[detection(.001, 4, [site]),
                                               detection(), detection()])
        common = dict(row_id=0, index='sample', prompt='P:', draft='P: original',
                      unwatermarked_text='P: independent', natural_text='human continuation')
        seeder = Mock()
        record, scores = next(cli.condition_records(wm, [common], 42, seeder))
        wm.generate_watermarked_text.assert_called_once_with('P:', draft='P: original')
        self.assertEqual([call.args[0] for call in wm.detect_watermark.call_args_list],
                         [' edited', ' independent', 'human continuation'])
        self.assertEqual(record['draft'], common['draft'])
        self.assertEqual(scores['positive']['num_tested_tokens'], 4)
        self.assertTrue(scores['alignment']['fully_verified'])
        seeder.assert_called_once_with(sample_seed(42, 0, 'embedding'))
        with self.assertRaises(ValueError):
            continuation('changed prefix', 'P:')

    def test_mean_N_comes_from_final_positive_including_failures(self):
        rows = []
        for n in (8, 0):
            rows.append(dict(positive=detection(.001 if n else None, n),
                             unwatermarked=detection(.5, 99), natural=detection(.5, 99),
                             alignment=final_alignment([], []), embed_sites=[]))
        summary = summarize_condition(rows, ['NP', 'VP', 'PP'], 4, .01)
        self.assertEqual(summary['mean_tested_tokens'], 4.)
        self.assertEqual(summary['unwatermarked']['mean_tested_tokens'], 99.)
        self.assertEqual(summary['num_texts'], 2)
        self.assertEqual(summary['num_valid_z'], 1)
        self.assertEqual(summary['no_evidence_fraction'], .5)
        self.assertIsNone(summary['final_site_alignment'])

    def test_dry_run_never_imports_models_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'input.jsonl'
            source.write_text('\n'.join(json.dumps(dict(index=i, prompt='P:', natural_text='human'))
                                        for i in range(2)))
            out = base / 'run'
            code = ('import sys; from experiments import role_ablation; role_ablation.main(sys.argv[1:]); '
                    'assert "torch" not in sys.modules; '
                    'assert "transformers" not in sys.modules; '
                    'assert "spacy" not in sys.modules')
            result = subprocess.run([sys.executable, '-c', code, 'run', '--input', str(source),
                                     '--output-dir', str(out), '--limit', '2', '--dry-run'],
                                    cwd=cli.ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(out.exists())
            self.assertEqual(len(json.loads(result.stdout)['conditions']), 6)

    def test_input_rejects_duplicate_ids_and_missing_human_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            row = dict(index=0, prompt='P:', natural_text='human')
            path.write_text(json.dumps(row) + '\n' + json.dumps(row))
            with self.assertRaises(ValueError):
                cli.input_rows(path, 2)
            row.pop('natural_text')
            path.write_text(json.dumps(row) + '\n' + json.dumps(dict(row, index=1)))
            with self.assertRaises(ValueError):
                cli.input_rows(path, 2)

    def fixture_run(self, base):
        """Saved-result fixture, not generation or a real-data experiment."""
        base.mkdir()
        config = dict(max_spans=4, p_threshold=.01, rejection_k=4)
        shared = [dict(row_id=i, index=i, prompt='P:', draft='P: draft',
                       unwatermarked_text='P: negative', natural_text='human') for i in range(2)]
        with (base / 'shared.jsonl').open('x') as f:
            for row in shared:
                cli.write_row(f, row)
        manifest = dict(schema=cli.SCHEMA, status='complete', num_samples=2, model='fixture',
                        conditions={name: dict(config, roles=list(roles)) for name, roles in ROLE_SETS})
        for name, roles in ROLE_SETS:
            meta = dict(config=manifest['conditions'][name],
                        shared_sha256=cli.sha256(base / 'shared.jsonl'))
            cli.write_json(base / f'{name}.meta.json', meta)
            with (base / f'{name}.jsonl').open('x') as f:
                for row in shared:
                    cli.write_row(f, dict(row, watermarked_text='P: marked'))
            records = [dict(row_id=i, index=i, positive=detection(.001 if i == 0 else None, 4 if i == 0 else 0),
                            unwatermarked=detection(), natural=detection(),
                            embed_sites=[], alignment=final_alignment([], [])) for i in range(2)]
            cli.write_json(base / f'{name}.evaluation.json', dict(
                schema=cli.SCHEMA, metadata=meta, records=records, parse_failures=0))
        manifest['files'] = {p.name: cli.sha256(p) for p in base.iterdir()}
        cli.write_json(base / 'manifest.json', manifest)

    def test_offline_report_and_integrity_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'run'
            self.fixture_run(base)
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(['summarize', '--run-dir', str(base), '--bootstrap-samples', '20'])
            report = json.loads((base / 'report/summary.json').read_text())
            self.assertEqual(len(report['conditions']), 6)
            self.assertEqual(report['conditions']['R3']['mean_tested_tokens'], 2.)
            self.assertTrue((base / 'report/table_role_ablation.tex').exists())
            self.assertTrue((base / 'report/summary.csv').exists())
            with self.assertRaises(ValueError):
                cli.summarize(SimpleNamespace(run_dir=base, output_dir=None,
                                             bootstrap_samples=20, seed=123))
            with (base / 'R7.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'Run file changed'):
                cli.load_run(base)

    def test_incomplete_runs_cannot_produce_a_paper_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cli.write_json(base / 'manifest.json', dict(schema=cli.SCHEMA, status='failed'))
            with self.assertRaisesRegex(ValueError, 'complete'):
                cli.load_run(base)


if __name__ == '__main__':
    unittest.main()
