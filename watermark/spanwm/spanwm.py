import hashlib
from math import sqrt
from pathlib import Path
import torch
from scipy.stats import binom
from transformers import LogitsProcessorList
from watermark.base import BaseConfig, BaseWatermark
from watermark.kgw.kgw import KGWUtils, KGWLogitsProcessor
from utils.transformers_config import TransformersConfig
from .span_ops import StructuralSpan, TokenSpanMapper
from .constituent_ops import ConstituentExtractor

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config/SpanWM.json'
PROTOCOL = 'spanwm-v1'


class SpanWMConfig(BaseConfig):
    @property
    def algorithm_name(self):
        return 'SpanWM'

    def initialize_parameters(self):
        for name in ['gamma', 'delta', 'hash_key', 'prefix_length', 'p_threshold', 'f_scheme', 'window_scheme', 'master_key', 'roles', 'role_ngram_chars', 'spacy_model', 'benepar_model', 'min_span_tokens', 'max_span_tokens', 'max_regen_attempts', 'max_spans', 'rejection_k', 'rejection_right_tokens']:
            setattr(self, name, self.config_dict[name])
        self.master_key = bytes.fromhex(self.master_key)
        if self.config_dict.get('algorithm_name') != 'SpanWM':
            raise ValueError('Only SpanWM is implemented')
        if not 0 < self.gamma < 1:
            raise ValueError('gamma must be between 0 and 1')
        if self.rejection_k < 2:
            raise ValueError('SpanWM requires candidate selection (rejection_k >= 2)')
        if not self.roles or self.min_span_tokens < 1 or self.max_span_tokens < self.min_span_tokens:
            raise ValueError('Invalid constituent selection settings')
        if min(self.max_spans, self.max_regen_attempts, self.prefix_length, self.rejection_right_tokens) < 1:
            raise ValueError('Counts must be positive')
        if self.config_dict.get('entropy_threshold', 0) != 0 or self.config_dict.get('detect_pad_chars', 0) != 0:
            raise ValueError('Entropy gating and padded detection are not part of SpanWM')


class SpanWMUtils:
    def __init__(self, config):
        self.config = config
        self.kgw = KGWUtils(config)
        self.mapper = TokenSpanMapper(config.generation_tokenizer)
        self.extractor = ConstituentExtractor(config.spacy_model, config.benepar_model, config.roles)

    def role_for_anchor(self, text: str, anchor_char: int) -> str:
        n = self.config.role_ngram_chars
        ngram = text[max(0, anchor_char - n):anchor_char].encode("utf-8")
        digest = hashlib.sha256(
            self.config.master_key + b"role-per-span" + ngram).digest()
        idx = int.from_bytes(digest[:8], "big") % len(self.config.roles)
        return self.config.roles[idx]

    def select_span_from(self, text: str, from_char: int) -> StructuralSpan | None:
        for sp in self.extractor.extract(text):
            if sp.start_char < from_char:
                continue
            if sp.role != self.role_for_anchor(text, sp.start_char):
                continue
            n = self.mapper.count(text, sp)
            if self.config.min_span_tokens <= n <= self.config.max_span_tokens:
                return sp
        return None

    def span_token_count(self, text: str, start_char: int, end_char: int) -> int:
        probe = StructuralSpan(-1, "_RANGE", start_char, end_char,
                               text[start_char:end_char], -1, -1)
        return len(self.mapper.map_positions(text, probe)[0])

    def window_end_char(self, text: str, anchor_char: int) -> int:

        sp = self.select_span_from(text, max(0, anchor_char - 4))
        if sp is not None:
            return sp.end_char
        return anchor_char + 1

    def scan_sites(self, text: str) -> list[StructuralSpan]:
        sites, search_from = [], 0
        for _ in range(self.config.max_spans):
            sp = self.select_span_from(text, search_from)
            if sp is None:
                break
            sites.append(sp)
            search_from = sp.end_char
        return sites

    def score_span(self, input_ids: list[int], positions: list[int]):

        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.config.device)
        green_count = 0
        scored = 0
        flags = [-1] * len(input_ids)
        for idx in positions:
            if idx < self.config.prefix_length:
                continue                                 # no context to seed green list
            greenlist = self.kgw.get_greenlist_ids(ids[:idx])
            if ids[idx] in greenlist:
                green_count += 1
                flags[idx] = 1
            else:
                flags[idx] = 0
            scored += 1
        if scored == 0:
            return None, None, flags, 0, 0
        expected = self.config.gamma
        z = (green_count - expected * scored) / sqrt(scored * expected * (1 - expected))
        # exact one-sided binomial tail: P(X >= G) under X ~ Binom(N, gamma)
        p = float(binom.sf(green_count - 1, scored, expected))
        return z, p, flags, scored, green_count


class SpanWM(BaseWatermark):
    _NO_SEP_BEFORE = set(",.;:!?)]}»’”'\"%…—–-")
    _NO_OPEN = set(",.;:!?)]}»’”%…")

    def __init__(self, algorithm_config=None, transformers_config=None):
        self.config = (algorithm_config if isinstance(algorithm_config, SpanWMConfig)
                       else SpanWMConfig(str(algorithm_config or DEFAULT_CONFIG), transformers_config))
        self.utils = SpanWMUtils(self.config)
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils.kgw)

    def _generate_draft(self, prompt):
        text, _ = self._raw_generate(prompt, watermark=False,
                                    max_new_tokens=self.config.gen_kwargs.get('max_new_tokens', 200))
        return prompt + text

    def generate_unwatermarked_text(self, prompt, *args, **kwargs):
        return self._generate_draft(prompt)

    def _raw_generate(self, context: str, watermark: bool, max_new_tokens: int,
                      min_new_tokens: int | None = None):
        """Base-model completion of `context`. Returns (new_text, new_ids)."""
        if watermark:
            context = self._regeneration_prompt + context
        tok = self.config.generation_tokenizer
        model = self.config.generation_model
        enc = tok(context, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        if enc["input_ids"].shape[1] == 0:
            sid = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            enc = {"input_ids": torch.tensor([[sid]], dtype=torch.long, device=self.config.device),
                   "attention_mask": torch.ones(1, 1, dtype=torch.long, device=self.config.device)}
        plen = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(self.config.gen_kwargs.get("do_sample", True)),
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        for k in ("top_p", "temperature", "top_k"):
            if k in self.config.gen_kwargs:
                gen_kwargs[k] = self.config.gen_kwargs[k]
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if watermark:
            gen_kwargs["logits_processor"] = LogitsProcessorList([self.logits_processor])
        out = model.generate(**enc, **gen_kwargs)
        new_ids = out[0][plen:]
        return tok.decode(new_ids, skip_special_tokens=True), new_ids

    def _splice(self, text: str, anchor_char: int, span_end_char: int, fill: str):
        left = text[:anchor_char]
        right = text[span_end_char:]
        strip_left = left.endswith(" ")
        ctx = left[:-1] if strip_left else left
        if strip_left and fill.lstrip(" ")[:1] in self._NO_OPEN:
            fill = fill.lstrip(" ")
            prefix = ctx
        elif strip_left and fill.startswith(" "):
            prefix = ctx
        else:
            prefix = left
        if fill.endswith(" ") and right and (right[0].isspace() or right[0] in self._NO_SEP_BEFORE):
            fill = fill.rstrip(" ") or fill
        body = prefix + fill
        sep_r = "" if (not right or right[0].isspace() or right[0] in self._NO_SEP_BEFORE
                       or (fill and fill[-1].isspace())) else " "
        return body + sep_r + right, len(body) - anchor_char

    def _right_fit(self, spliced: str, resume_char: int) -> float:
        """Mean logP of the first `rejection_right_tokens` tokens of the right
        context, conditioned on everything before them. Higher = the fill leads
        into the original continuation more naturally."""
        model = self.config.generation_model
        m = self.config.rejection_right_tokens
        ids, offsets = self.utils.mapper.encode_with_offsets(spliced)
        j = next((i for i, (b, e) in enumerate(offsets) if e > resume_char), None)
        if model is None or j is None or j == 0 or j >= len(ids):
            return 0.0
        end = min(len(ids), j + m)
        if end <= j:
            return 0.0
        device = next(model.parameters()).device
        t = torch.tensor([ids[:end]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=t).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(ids[j:end], dtype=torch.long, device=device)
        sel = logp[j - 1:end - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(sel.mean().item())

    def _regen_at(self, text: str, anchor_char: int, span_end_char: int):
        """Generate `rejection_k` watermarked fills of the constituent's length,
        splice each, and keep the one the original right context follows best."""
        left = text[:anchor_char]
        k = max(1, self.utils.span_token_count(text, anchor_char, span_end_char))
        ctx = left[:-1] if left.endswith(" ") else left

        best = None                       # (score, spliced, fill_len)
        for _ in range(self.config.rejection_k):
            fill = None
            for _ in range(3):            # avoid a punctuation opener
                cand, _ = self._raw_generate(ctx, watermark=True,
                                             max_new_tokens=k, min_new_tokens=k)
                fill = cand
                if cand.lstrip(" ")[:1] not in self._NO_OPEN:
                    break
            spliced, fill_len = self._splice(text, anchor_char, span_end_char, fill)
            if self.config.rejection_k == 1:
                return spliced, fill_len
            score = self._right_fit(spliced, anchor_char + fill_len)
            if best is None or score > best[0]:
                best = (score, spliced, fill_len)
        return best[1], best[2]

    def _embed_continuation(self, draft: str) -> str:
        """Draft -> scan sites left->right; at each site regenerate a small
        watermarked window (verify + retry per site) -> final text."""
        text = draft
        sites = []          # per-site dicts for last_embedding_info
        search_from = 0

        for _ in range(self.config.max_spans):
            sp = self.utils.select_span_from(text, search_from)
            if sp is None:
                break
            anchor = sp.start_char
            best = None      # (text_after, fill_len, dist)
            for attempt in range(1, self.config.max_regen_attempts + 1):
                cand, fill_len = self._regen_at(text, anchor, sp.end_char)
                recon = self.utils.select_span_from(cand, search_from)
                dist = abs(recon.start_char - anchor) if recon is not None else 10 ** 9
                if best is None or dist < best[2]:
                    best = (cand, fill_len, dist)
                if dist <= 4:
                    break
            text, fill_len, dist = best
            sites.append({
                "role": sp.role, "anchor": anchor,
                "draft_span": sp.text, "fill": text[anchor:anchor + fill_len],
                "attempts": attempt, "anchor_dist": dist, "verified": dist <= 4,
            })
            search_from = self.utils.window_end_char(text, anchor)

        # final check: does the detector-side scan of the FINAL text land on
        # the same anchors?
        final_anchors = [s.start_char for s in self.utils.scan_sites(text)]
        embedded_anchors = [s["anchor"] for s in sites]
        aligned = sum(1 for a, b in zip(embedded_anchors, final_anchors) if abs(a - b) <= 4)
        verified = len(sites) > 0 and aligned == len(sites) == len(final_anchors)

        self.last_embedding_info = {
            "skipped": len(sites) == 0, "verified": verified,
            "num_sites": len(sites), "num_aligned": aligned,
            "sites": sites, "draft": draft, "final_text": text,
            "attempts": sum(s["attempts"] for s in sites),
        }
        return text

    def generate_watermarked_text(self, prompt, *args, draft=None, **kwargs):
        """Watermark a fresh or supplied full draft (prompt + continuation).

        Supplying a draft lets controlled experiments share the same starting
        text. It bypasses draft generation only; insertion is unchanged.
        """
        if hasattr(self, '_regeneration_prompt'):
            raise RuntimeError('Overlapping calls on one SpanWM instance are unsupported')
        if draft is not None and (not isinstance(draft, str) or not draft.startswith(prompt)):
            raise ValueError('A supplied draft must start with the exact prompt')
        self._regeneration_prompt = prompt
        try:
            if draft is None:
                draft = self._generate_draft(prompt)
            continuation = self._embed_continuation(draft[len(prompt):])
            text = prompt + continuation
            self.last_embedding_info.update(
                protocol=PROTOCOL, draft=draft, final_text=text,
                regeneration_context='original_prompt+left_continuation')
            return text
        finally:
            del self._regeneration_prompt

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Pooled exact binomial test; each site contributes the token extent
        of its reconstructed constituent (variable N per site)."""
        sites = self.utils.scan_sites(text)
        if not sites:
            result = {"is_watermarked": False, "score": None, "p_value": None,
                      "num_tested_tokens": 0, "num_green_tokens": 0,
                      "num_sites": 0, "sites": [], "reconstructed": False}
            return result if return_dict else (False, None)

        positions = []
        input_ids = None
        for sp in sites:
            pos, input_ids = self.utils.mapper.map_positions(text, sp)
            positions.extend(pos)
        positions = sorted(set(positions))

        z, p, _, n, g = self.utils.score_span(input_ids, positions)
        is_watermarked = bool(p is not None and p < self.config.p_threshold)

        result = {
            "is_watermarked": is_watermarked,
            "score": z,
            "p_value": p,
            "num_tested_tokens": n,
            "num_green_tokens": g,
            "num_sites": len(sites),
            "sites": [{"role": sp.role, "anchor": sp.start_char, "text": sp.text}
                      for sp in sites],
            "reconstructed": True,
        }
        return result if return_dict else (is_watermarked, z)
