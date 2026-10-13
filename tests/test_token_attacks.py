import unittest

from utils.token_attacks import attack_text, detection_summary, edit_count, mutate_ids


class TinyTokenizer:
    all_special_ids = [99]

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()] if text else []

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ' '.join(str(x) for x in ids)


class TokenAttackTests(unittest.TestCase):
    def test_edit_count_is_floor_of_original_token_count(self):
        self.assertEqual(edit_count(10, .1), 1)
        self.assertEqual(edit_count(11, .1), 1)
        self.assertEqual(edit_count(10, .5), 5)

    def test_deletion_is_exact_and_reproducible(self):
        a, info = mutate_ids(list(range(10)), 'deletion', .3, 7, [1, 2])
        b, same = mutate_ids(list(range(10)), 'deletion', .3, 7, [1, 2])
        self.assertEqual(a, b)
        self.assertEqual(info, same)
        self.assertEqual(info['num_edits'], 3)
        self.assertEqual(len(a), 7)

    def test_insertion_uses_donor_pool_and_exact_count(self):
        a, info = mutate_ids([10, 11, 12, 13], 'insertion', .5, 5, [20, 21])
        self.assertEqual(info['num_edits'], 2)
        self.assertEqual(len(a), 6)
        self.assertTrue(set(a) <= {10, 11, 12, 13, 20, 21})
        with self.assertRaises(ValueError):
            mutate_ids([1], 'insertion', 1., 5, [])

    def test_zero_edit_keeps_original_text_exactly(self):
        tok = TinyTokenizer()
        text, info = attack_text('1 2 3', tok, 'deletion', .1, 2, [4])
        self.assertEqual(text, '1 2 3')
        self.assertEqual(info['num_edits'], 0)

    def test_detection_summary_keeps_no_evidence_and_classes(self):
        empty = dict(p_value=None, score=None, num_tested_tokens=0,
                     num_green_tokens=0, num_sites=0)
        positive = dict(p_value=.001, score=2., num_tested_tokens=4,
                        num_green_tokens=4, num_sites=1)
        result = detection_summary([positive, empty], [empty, empty], .01)
        self.assertEqual(result['AUROC'], .75)
        self.assertEqual(result['no_evidence_fraction'], .5)
        self.assertEqual(result['negative_FPR_resolution'], .5)


if __name__ == '__main__':
    unittest.main()
