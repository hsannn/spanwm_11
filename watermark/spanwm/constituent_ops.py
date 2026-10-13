from collections import OrderedDict

import spacy

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise NotImplementedError("pair inputs not supported by the shim")
        if "T5" in type(self).__name__:
            return list(token_ids_0) + [self.eos_token_id]
        raise NotImplementedError(
            f"no build_inputs_with_special_tokens shim for {type(self).__name__}")
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = \
        _build_inputs_with_special_tokens

import benepar

from watermark.spanwm.span_ops import StructuralSpan


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:

    out: list[str] = []
    idx_map: list[int] = []
    in_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_space:
                continue
            out.append(' ')
            idx_map.append(i)
            in_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            in_space = False
    return ''.join(out), idx_map


class ConstituentExtractor:

    def __init__(self, spacy_model: str, benepar_model: str, labels: list[str],
                 cache_size: int = 32) -> None:
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("benepar", config={"model": benepar_model})
        self.labels = set(labels)
        self._cache: OrderedDict[str, list[StructuralSpan]] = OrderedDict()
        self._cache_size = cache_size
        self.n_parse_failures = 0

    def extract(self, text: str) -> list[StructuralSpan]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        spans = self._extract_uncached(text)
        self._cache[text] = spans
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return spans

    def _extract_uncached(self, text: str) -> list[StructuralSpan]:
        collapsed, idx_map = _collapse_whitespace(text)
        if not collapsed:
            return []
        try:
            doc = self.nlp(collapsed)
        except Exception:
            self.n_parse_failures += 1
            return []

        seen: set[tuple] = set()
        spans: list[StructuralSpan] = []
        for si, sent in enumerate(doc.sents):
            for c in sent._.constituents:
                role = next((lb for lb in c._.labels if lb in self.labels), None)
                if role is None:
                    continue
                if c.end_char <= c.start_char or not collapsed[c.start_char:c.end_char].strip():
                    continue
                s = idx_map[c.start_char]
                e = idx_map[c.end_char - 1] + 1
                key = (role, s, e)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(StructuralSpan(
                    sentence_index=si, role=role,
                    start_char=s, end_char=e, text=text[s:e],
                    parser_token_start=c.start, parser_token_end=c.end,
                ))
        spans.sort(key=lambda sp: (sp.start_char, sp.role, sp.end_char))
        return spans
