import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from evaluation.judge_similarity import Similarity, judge_pair, prepare_pair, request_id


class JudgeTests(unittest.TestCase):
    def test_scope_strips_only_exact_prompt(self):
        row = dict(prompt='Prompt:', a='Prompt: Cats sleep.', b='Prompt: Cats rest.')
        self.assertEqual(prepare_pair(row, 'a', 'b', 'continuation'),
                         dict(reference=' Cats sleep.', candidate=' Cats rest.'))
        row['b'] = 'Changed prompt: Cats rest.'
        with self.assertRaises(ValueError):
            prepare_pair(row, 'a', 'b', 'continuation')
        self.assertEqual(prepare_pair(row, 'a', 'b', 'full')['candidate'], row['b'])

    def test_empty_texts_and_invalid_scores(self):
        with self.assertRaises(ValueError):
            prepare_pair(dict(a='', b='text'), 'a', 'b', 'full')
        for value in (-1, 11, 5.5, True, '7'):
            with self.assertRaises(ValidationError):
                Similarity(score=value, reason='test')
        self.assertEqual(Similarity(score=10, reason='Equivalent.').score, 10)

    def test_resume_identity_depends_on_text_and_settings(self):
        pair = dict(reference='A', candidate='B')
        key = request_id(pair, dict(model='one'))
        self.assertNotEqual(key, request_id(pair, dict(model='two')))
        self.assertNotEqual(key, request_id(dict(reference='A', candidate='C'), dict(model='one')))

    def test_api_contract_and_refusal(self):
        message = NS(refusal=None, parsed=Similarity(score=8, reason='Minor differences.'))
        completion = NS(choices=[NS(message=message, finish_reason='stop')],
                        model='model-snapshot', id='response-test', usage=None)
        parse = Mock(return_value=completion)
        client = NS(chat=NS(completions=NS(parse=parse)))
        pair = dict(reference='A', candidate='B')
        self.assertEqual(judge_pair(client, pair, 'rubric')['score'], 8)
        kwargs = parse.call_args.kwargs
        self.assertEqual(kwargs['model'], 'gpt-5.6-luna')
        self.assertEqual(kwargs['response_format'], Similarity)
        self.assertEqual(kwargs['messages'][0]['role'], 'system')
        self.assertNotIn('temperature', kwargs)
        message.refusal = 'refusal'
        with self.assertRaises(RuntimeError):
            judge_pair(client, pair, 'rubric')
        message.refusal = None
        completion.choices[0].finish_reason = 'length'
        with self.assertRaises(RuntimeError):
            judge_pair(client, pair, 'rubric')


if __name__ == '__main__':
    unittest.main()
