import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from detect import metrics
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark
from watermark.spanwm.spanwm import DEFAULT_CONFIG, SpanWM, SpanWMConfig, SpanWMUtils
from watermark.kgw.kgw import KGWUtils


class ProtocolTests(unittest.TestCase):
    def config(self, **kwargs):
        tc = TransformersConfig(None, None, vocab_size=128, device='cpu')
        return SpanWMConfig(str(DEFAULT_CONFIG), tc, **kwargs)

    def test_candidate_selection_required(self):
        self.assertEqual(self.config().rejection_k, 4)
        for overrides in ({'rejection_k': 1}, {'entropy_threshold': 1}, {'detect_pad_chars': 1}):
            with self.assertRaises(ValueError):
                self.config(**overrides)
        with self.assertRaises(ValueError):
            AutoWatermark.load('KGW')

    def test_prompt_preserved_and_excluded_from_editing(self):
        wm = SpanWM.__new__(SpanWM)
        wm._generate_draft = Mock(return_value='PROMPT: draft')
        def embed(text):
            self.assertEqual(text, ' draft')
            self.assertEqual(wm._regeneration_prompt, 'PROMPT:')
            wm.last_embedding_info = {}
            return ' edited'
        wm._embed_continuation = embed
        self.assertEqual(wm.generate_watermarked_text('PROMPT:'), 'PROMPT: edited')
        self.assertEqual(wm.last_embedding_info['draft'], 'PROMPT: draft')
        self.assertFalse(hasattr(wm, '_regeneration_prompt'))

    def test_prompt_state_cleared_on_failure(self):
        wm = SpanWM.__new__(SpanWM)
        wm._generate_draft = Mock(side_effect=ValueError('generation failed'))
        with self.assertRaises(ValueError):
            wm.generate_watermarked_text('prompt')
        self.assertFalse(hasattr(wm, '_regeneration_prompt'))

    def test_supplied_draft_skips_generation_but_keeps_insertion_context(self):
        wm = SpanWM.__new__(SpanWM)
        wm._generate_draft = Mock(side_effect=AssertionError('must reuse the draft'))
        def embed(text):
            self.assertEqual(text, ' shared draft')
            self.assertEqual(wm._regeneration_prompt, 'PROMPT:')
            wm.last_embedding_info = {}
            return ' edited'
        wm._embed_continuation = embed
        result = wm.generate_watermarked_text('PROMPT:', draft='PROMPT: shared draft')
        self.assertEqual(result, 'PROMPT: edited')
        self.assertEqual(wm.last_embedding_info['draft'], 'PROMPT: shared draft')
        wm._generate_draft.assert_not_called()
        self.assertFalse(hasattr(wm, '_regeneration_prompt'))
        with self.assertRaises(ValueError):
            wm.generate_watermarked_text('PROMPT:', draft='wrong prefix')
        self.assertFalse(hasattr(wm, '_regeneration_prompt'))

    def test_supplied_draft_state_cleared_after_insertion_failure(self):
        wm = SpanWM.__new__(SpanWM)
        wm._embed_continuation = Mock(side_effect=ValueError('insertion failed'))
        with self.assertRaises(ValueError):
            wm.generate_watermarked_text('P:', draft='P: draft')
        self.assertFalse(hasattr(wm, '_regeneration_prompt'))

    def test_repeated_events_are_counted(self):
        utils = SpanWMUtils.__new__(SpanWMUtils)
        utils.config = self.config()
        utils.kgw = KGWUtils(utils.config)
        z, p, flags, n, g = utils.score_span([7, 7, 7, 7], [0, 1, 2, 3])
        self.assertEqual(n, 3)  # first position has no seed; repeated events remain
        self.assertIn(g, (0, 3))
        self.assertEqual(flags[0], -1)
        self.assertAlmostEqual(p, .25 ** 3 if g == 3 else 1.)

    def test_detector_unions_overlapping_positions(self):
        wm = SpanWM.__new__(SpanWM)
        site = SimpleNamespace(role='NP', start_char=0, text='sample')
        score = Mock(return_value=(2., .01, [], 2, 2))
        wm.config = SimpleNamespace(p_threshold=.05)
        wm.utils = SimpleNamespace(
            scan_sites=lambda _: [site, site],
            mapper=SimpleNamespace(map_positions=lambda *_: ([1, 2], [0, 1, 2])),
            score_span=score)
        result = wm.detect_watermark('sample')
        score.assert_called_once_with([0, 1, 2], [1, 2])
        self.assertEqual(result['num_tested_tokens'], 2)
        self.assertTrue(result['is_watermarked'])

    def test_no_site_examples_remain_in_metrics(self):
        empty = dict(p_value=None, score=None, num_tested_tokens=0, num_green_tokens=0)
        positive = dict(p_value=.01, score=2., num_tested_tokens=4, num_green_tokens=3)
        result = metrics([positive, empty], [empty, empty])
        self.assertEqual(result['num_positive'], 2)
        self.assertEqual(result['N'], 2)
        self.assertEqual(result['AUROC'], .75)
        self.assertEqual(result['negative_FPR_resolution'], .5)


if __name__ == '__main__':
    unittest.main()
