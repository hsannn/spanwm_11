import unittest
from types import SimpleNamespace

import torch

from evaluation.eval_similarity import embed_texts, last_token_pool
from utils.similarity import prepare_pair


class SimilarityTests(unittest.TestCase):
    def test_draft_is_distinct_from_independent_unwatermarked_sample(self):
        row = dict(prompt='P:', draft='P: draft', watermarked_text='P: edited',
                   unwatermarked_text='P: independent')
        self.assertEqual(prepare_pair(row, 'draft', 'watermarked_text', 'continuation'),
                         dict(reference=' draft', candidate=' edited'))
        with self.assertRaises(KeyError):
            prepare_pair({'watermarked_text': 'text'}, 'draft', 'watermarked_text', 'full')

    def test_pooling_ignores_padding_on_either_side(self):
        hidden = torch.arange(24).reshape(2, 4, 3).float()
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
        self.assertTrue(torch.equal(last_token_pool(hidden, mask), hidden[[0, 1], [3, 1]]))
        with self.assertRaises(ValueError):
            last_token_pool(hidden, torch.zeros_like(mask))

    def test_embeddings_normalized_and_truncation_lengths_retained(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                if not kwargs.get('truncation'):
                    return {'input_ids': [[1] * len(t) for t in texts]}
                return Batch(input_ids=torch.ones(len(texts), 2, dtype=torch.long),
                             attention_mask=torch.ones(len(texts), 2, dtype=torch.long))

        class Model:
            def __call__(self, **batch):
                n = batch['input_ids'].shape[0]
                return SimpleNamespace(last_hidden_state=torch.tensor([3., 4.]).expand(n, 2, 2))

        vectors, lengths = embed_texts(['abc', 'de', 'fghi'], Tokenizer(), Model(), 'cpu', 2, 2)
        self.assertEqual(lengths, [3, 2, 4])
        self.assertTrue(torch.allclose(vectors, torch.tensor([[.6, .8]] * 3)))


if __name__ == '__main__':
    unittest.main()
