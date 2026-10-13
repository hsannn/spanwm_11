import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.role_ablation import (
    ROLE_SETS, continuation, final_alignment, latex_table, paired_auc_intervals,
    rank_score, sample_seed, summarize_condition,
)

SCHEMA = 'spanwm-role-ablation-1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def input_rows(path, limit):
    if limit < 2:
        raise ValueError('--limit must be at least 2')
    rows = read_jsonl(path)[:limit]
    if len(rows) < limit:
        raise ValueError(f'Requested {limit} rows but input has only {len(rows)}')
    seen = set()
    for i, row in enumerate(rows):
        for key in ('prompt', 'natural_text'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Row {i}: nonempty {key} is required')
        index = row.get('index', i)
        if type(index) not in (int, str) or index in seen:
            raise ValueError(f'Row {i}: index must be a unique integer or string')
        seen.add(index)
        rows[i] = dict(row_id=i, index=index, prompt=row['prompt'], natural_text=row['natural_text'])
    return rows


def condition_records(wm, shared, seed, set_seed):
    """Run one condition; injectable WM/seed function permit offline checks."""
    for row in shared:
        embed_seed = sample_seed(seed, row['row_id'], 'embedding')
        set_seed(embed_seed)
        text = wm.generate_watermarked_text(row['prompt'], draft=row['draft'])
        info = wm.last_embedding_info
        pos = wm.detect_watermark(continuation(text, row['prompt']))
        neg = wm.detect_watermark(continuation(row['unwatermarked_text'], row['prompt']))
        human = wm.detect_watermark(row['natural_text'])
        alignment = final_alignment(info['sites'], pos['sites'])
        record = dict(row, watermarked_text=text, protocol=info['protocol'], seed=embed_seed,
                      embed_skipped=info['skipped'], embed_verified=info['verified'],
                      embed_num_sites=info['num_sites'], embed_num_aligned=info['num_aligned'],
                      embed_sites=info['sites'])
        evaluation = dict(row_id=row['row_id'], index=row['index'], positive=pos,
                          unwatermarked=neg, natural=human, alignment=alignment,
                          embed_sites=info['sites'])
        yield record, evaluation


def run(args):
    rows = input_rows(args.input, args.limit)
    config = json.loads(args.config.read_text())
    if config.get('algorithm_name') != 'SpanWM' or config.get('rejection_k', 0) < 2:
        raise ValueError('Use the current SpanWM configuration with candidate selection enabled')
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError('Invalid generation parameters')
    if args.output_dir.exists():
        raise ValueError('Output directory exists; choose a new path')
    plan = dict(schema=SCHEMA, model=args.model, revision=args.revision,
                num_samples=len(rows), seed=args.seed, device=args.device,
                input=str(args.input.resolve()), input_sha256=sha256(args.input),
                generation=dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                                temperature=args.temperature, top_p=args.top_p),
                conditions={name: dict(config, roles=list(roles)) for name, roles in ROLE_SETS},
                output_dir=str(args.output_dir.resolve()),
                seed_scheme='SHA256(base seed, row ordinal, draft/negative/embedding); shared across conditions',
                scope='continuation only; natural_text must already be a human continuation')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    # Heavy imports are deliberately below the dry-run boundary.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from utils.transformers_config import TransformersConfig
    from watermark.spanwm.spanwm import DEFAULT_CONFIG, PROTOCOL, SpanWM, SpanWMConfig

    if torch.device(args.device).type not in ('cpu', 'cuda'):
        raise ValueError('Use a CPU or CUDA device')
    source_paths = ('experiments/role_ablation.py', 'utils/role_ablation.py',
                    'watermark/spanwm/spanwm.py', 'watermark/spanwm/constituent_ops.py',
                    'watermark/spanwm/span_ops.py', 'watermark/kgw/kgw.py')
    manifest = dict(plan, status='running', protocol=PROTOCOL,
                    torch_version=torch.__version__,
                    source_sha256={p: sha256(ROOT / p) for p in source_paths}, files={})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / 'manifest.json'
    write_json(manifest_path, manifest)
    try:
        load = dict(local_files_only=args.local_files_only)
        if args.revision:
            load['revision'] = args.revision
        tok = AutoTokenizer.from_pretrained(args.model, **load)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype='auto', device_map=args.device, **load).eval()
        tc = TransformersConfig(model, tok, vocab_size=model.config.vocab_size,
                                device=args.device, **plan['generation'])
        manifest['model_commit'] = getattr(model.config, '_commit_hash', None)
        manifest['vocab_size'] = model.config.vocab_size
        manifest['device_type'] = torch.device(args.device).type
        write_json(manifest_path, manifest)
        wm = SpanWM(SpanWMConfig(str(DEFAULT_CONFIG), tc, **plan['conditions']['R3']))

        shared = []
        with (args.output_dir / 'shared.jsonl').open('x') as f:
            for row in rows:
                ds = sample_seed(args.seed, row['row_id'], 'draft')
                ns = sample_seed(args.seed, row['row_id'], 'negative')
                set_seed(ds)
                draft = wm.generate_unwatermarked_text(row['prompt'])
                set_seed(ns)
                negative = wm.generate_unwatermarked_text(row['prompt'])
                continuation(draft, row['prompt'])
                continuation(negative, row['prompt'])
                common = dict(row, draft=draft, unwatermarked_text=negative,
                              draft_seed=ds, negative_seed=ns)
                write_row(f, common)
                shared.append(common)
                print(f"shared {len(shared)}/{len(rows)}", flush=True)

        for ci, (name, _) in enumerate(ROLE_SETS):
            if ci:
                # One parser at a time; never reuse a cache across type inventories.
                del wm
                wm = SpanWM(SpanWMConfig(str(DEFAULT_CONFIG), tc, **plan['conditions'][name]))
            meta = dict(protocol=PROTOCOL, ablation_schema=SCHEMA, condition=name,
                        model=args.model, revision=args.revision,
                        model_commit=manifest['model_commit'], vocab_size=model.config.vocab_size,
                        device_type=manifest['device_type'], seed=args.seed,
                        config=plan['conditions'][name], generation=plan['generation'],
                        input_sha256=plan['input_sha256'], shared_sha256=sha256(args.output_dir / 'shared.jsonl'),
                        source_sha256=manifest['source_sha256'], torch_version=torch.__version__)
            write_json(args.output_dir / f'{name}.meta.json', meta)
            evaluations = []
            with (args.output_dir / f'{name}.jsonl').open('x') as f:
                for record, evaluation in condition_records(wm, shared, args.seed, set_seed):
                    write_row(f, record)
                    evaluations.append(evaluation)
                    print(f"{name} {len(evaluations)}/{len(shared)}", flush=True)
            write_json(args.output_dir / f'{name}.evaluation.json',
                       dict(schema=SCHEMA, metadata=meta, records=evaluations,
                            parse_failures=wm.utils.extractor.n_parse_failures))
        for path in sorted(args.output_dir.iterdir()):
            if path.name != 'manifest.json':
                manifest['files'][path.name] = sha256(path)
        manifest['status'] = 'complete'
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        write_json(manifest_path, manifest)
        raise


def load_run(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Summarization requires a complete role-ablation run')
    required = {'shared.jsonl'} | {
        name + suffix for name, _ in ROLE_SETS
        for suffix in ('.jsonl', '.meta.json', '.evaluation.json')}
    if set(manifest['files']) != required:
        raise ValueError('Run manifest does not contain exactly the six required conditions')
    for name, digest in manifest['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Run file changed: {name}')
    shared = read_jsonl(directory / 'shared.jsonl')
    if len(shared) != manifest['num_samples'] or len(shared) < 2:
        raise ValueError('Shared row count mismatch')
    ids = [(r['row_id'], r['index']) for r in shared]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate shared row identities')
    evaluations = {}
    base_config = manifest['conditions']['R3']
    for name, roles in ROLE_SETS:
        expected = dict(base_config, roles=list(roles))
        if manifest['conditions'][name] != expected:
            raise ValueError('Conditions must differ only in their type inventory')
        meta = json.loads((directory / f'{name}.meta.json').read_text())
        evaluation = json.loads((directory / f'{name}.evaluation.json').read_text())
        records = read_jsonl(directory / f'{name}.jsonl')
        if meta['config'] != expected or evaluation['metadata'] != meta:
            raise ValueError(f'{name}: inconsistent configuration metadata')
        if meta['shared_sha256'] != manifest['files']['shared.jsonl']:
            raise ValueError(f'{name}: shared draft checksum mismatch')
        for content in (records, evaluation['records']):
            if [(r['row_id'], r['index']) for r in content] != ids:
                raise ValueError(f'{name}: row pairing mismatch')
        for row, common in zip(records, shared):
            if any(row.get(k) != v for k, v in common.items()):
                raise ValueError(f'{name}: prompts, drafts, or negatives differ')
        evaluations[name] = evaluation
    return manifest, evaluations


def summarize(args):
    if args.bootstrap_samples < 1:
        raise ValueError('--bootstrap-samples must be positive')
    manifest, evaluations = load_run(args.run_dir)
    output = args.output_dir or args.run_dir / 'report'
    if output.exists():
        raise ValueError('Report directory exists; choose a new path')
    summaries = {}
    base = evaluations['R3']['records']
    for name, roles in ROLE_SETS:
        rows = evaluations[name]['records']
        cfg = manifest['conditions'][name]
        summary = summarize_condition(rows, roles, cfg['max_spans'], cfg['p_threshold'])
        summary['parse_failures'] = evaluations[name]['parse_failures']
        for neg in ('unwatermarked', 'natural'):
            vectors = [[rank_score(r[key]) for r in source]
                       for source, key in ((rows, 'positive'), (rows, neg),
                                           (base, 'positive'), (base, neg))]
            summary[neg].update(paired_auc_intervals(
                *vectors, samples=args.bootstrap_samples, seed=args.seed))
        summaries[name] = summary
    report = dict(schema=SCHEMA, model=manifest['model'], num_samples=manifest['num_samples'],
                  manifest_sha256=sha256(args.run_dir / 'manifest.json'),
                  bootstrap=dict(samples=args.bootstrap_samples, seed=args.seed,
                                 unit='prompt row, jointly resampled across classes and conditions'),
                  notes=['No-site and failed-alignment outputs remain in all detection metrics.',
                         'Mean z excludes undefined scores; num_valid_z is reported.',
                         'Final alignment uses ordered anchor proximity, not token identity.',
                         'CI overlap does not establish equivalence or absence of a cost.',
                         'No text-quality metric is inferred from detection results.'],
                  conditions=summaries)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', report)
    (output / 'table_role_ablation.tex').write_text(latex_table(summaries))
    import csv
    with (output / 'summary.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'roles', 'mean_N', 'final_alignment', 'mean_z',
                         'AUROC_unwatermarked', 'AUROC_human', 'TPR_at_1pct_FPR'])
        for name, s in summaries.items():
            writer.writerow([name, ','.join(s['roles']), s['mean_tested_tokens'],
                             s['final_site_alignment'], s['mean_z'],
                             s['unwatermarked']['AUROC'], s['natural']['AUROC'],
                             s['unwatermarked']['TPR_at_1pct_FPR']])
    print(f'Report written to {output}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    run_ap = commands.add_parser('run', help='Generate shared drafts, insert, and detect six conditions')
    run_ap.add_argument('--input', type=Path, required=True)
    run_ap.add_argument('--output-dir', type=Path, required=True)
    run_ap.add_argument('--config', type=Path, default=ROOT / 'config/SpanWM.json')
    run_ap.add_argument('--model', default='meta-llama/Llama-3.2-3B')
    run_ap.add_argument('--revision')
    run_ap.add_argument('--device', default='cuda')
    run_ap.add_argument('--limit', type=int, default=200)
    run_ap.add_argument('--seed', type=int, default=42)
    run_ap.add_argument('--max-new-tokens', type=int, default=200)
    run_ap.add_argument('--temperature', type=float, default=.8)
    run_ap.add_argument('--top-p', type=float, default=.9)
    run_ap.add_argument('--local-files-only', action='store_true')
    run_ap.add_argument('--dry-run', action='store_true')
    report_ap = commands.add_parser('summarize', help='CPU-only summary from a complete saved run')
    report_ap.add_argument('--run-dir', type=Path, required=True)
    report_ap.add_argument('--output-dir', type=Path)
    report_ap.add_argument('--bootstrap-samples', type=int, default=10000)
    report_ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args(argv)
    try:
        (run if args.command == 'run' else summarize)(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    main()
