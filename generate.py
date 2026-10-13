"""Generate watermarked and unwatermarked texts from a prompt JSONL file."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from watermark.auto_watermark import AutoWatermark
from watermark.spanwm.spanwm import DEFAULT_CONFIG, PROTOCOL
from utils.transformers_config import TransformersConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='JSONL records with prompt; optional index/natural_text')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='meta-llama/Llama-3.2-3B')
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--max-new-tokens', type=int, default=200)
    ap.add_argument('--local-files-only', action='store_true')
    args = ap.parse_args()
    if args.output.exists() or args.output.with_suffix('.meta.json').exists():
        ap.error('Output exists; choose a new path')
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        if args.limit < 1: ap.error('--limit must be positive')
        rows = rows[:args.limit]
    if not rows or any(not isinstance(r.get('prompt'), str) or not r['prompt'].strip() for r in rows):
        ap.error('Input must contain nonempty string prompts')
    if args.max_new_tokens < 1: ap.error('--max-new-tokens must be positive')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto', device_map=args.device,
                                               local_files_only=args.local_files_only).eval()
    tc = TransformersConfig(model, tok, vocab_size=model.config.vocab_size, device=args.device,
                            max_new_tokens=args.max_new_tokens, do_sample=True, top_p=.9, temperature=.8)
    wm = AutoWatermark.load('SpanWM', str(args.config), tc)
    metadata = dict(protocol=PROTOCOL, model=args.model, vocab_size=model.config.vocab_size,
                    device_type=torch.device(args.device).type, seed=args.seed,
                    config=json.loads(args.config.read_text()), generation=tc.gen_kwargs,
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    torch_version=torch.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with args.output.open('x') as output:
        for i, row in enumerate(rows):
            set_seed(args.seed+i)
            text = wm.generate_watermarked_text(row['prompt'])
            info = wm.last_embedding_info
            set_seed(args.seed+1000000+i)
            negative = wm.generate_unwatermarked_text(row['prompt'])
            record = dict(index=row.get('index', i), prompt=row['prompt'],
                          watermarked_text=text, unwatermarked_text=negative,
                          natural_text=row.get('natural_text', ''), draft=info['draft'],
                          protocol=PROTOCOL, seed=args.seed+i,
                          embed_skipped=info['skipped'], embed_verified=info['verified'],
                          embed_num_sites=info['num_sites'], embed_num_aligned=info['num_aligned'],
                          embed_sites=info['sites'])
            output.write(json.dumps(record, ensure_ascii=False)+'\n'); output.flush()
            print(f"{i+1}/{len(rows)} sites={info['num_sites']} verified={info['verified']}", flush=True)
    print(f'parse_failures={wm.utils.extractor.n_parse_failures}')

if __name__ == '__main__': main()
