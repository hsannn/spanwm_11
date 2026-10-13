import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

import torch
from transformers import AutoTokenizer


def load_oracle(name, device, local_files_only):
    from transformers import AutoModelForCausalLM
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText   # multimodal checkpoints
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    return model.to(device).eval()


def ppl(text, prompt, tok, model, device):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    start = next((i for i, (a, b) in enumerate(offsets) if b > len(prompt)), None)
    if start is None:
        return None
    ids = [tok.bos_token_id] + ids
    start += 1
    x = torch.tensor([ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=x).logits[0, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits[start - 1:], x[0, start:], reduction='mean')
    return math.exp(nll.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fields', nargs='+', default=['watermarked_text', 'unwatermarked_text'])
    ap.add_argument('--oracle', default='google/gemma-4-12B')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--local-files-only', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists; choose a new path')
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for r in rows:
        for f in args.fields:
            if not r[f].startswith(r['prompt']):
                raise ValueError(f'record {r.get("index")}: {f} does not start with prompt')
    tok = AutoTokenizer.from_pretrained(args.oracle, local_files_only=args.local_files_only)
    if tok.bos_token_id is None:
        raise ValueError('Oracle tokenizer has no BOS token')
    model = load_oracle(args.oracle, args.device, args.local_files_only)
    per_sample, summary = [], {}
    for f in args.fields:
        vals = [ppl(r[f], r['prompt'], tok, model, args.device) for r in rows]
        per_sample.append({f: vals})
        ok = [v for v in vals if v is not None]
        summary[f] = dict(n=len(ok), empty=len(vals) - len(ok), mean=statistics.mean(ok),
                          median=statistics.median(ok))
    result = dict(oracle=args.oracle, input=str(args.input.resolve()),
                  input_sha256=sha256(raw).hexdigest(), summary=summary,
                  per_sample={f: v for d in per_sample for f, v in d.items()},
                  indices=[r.get('index') for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
