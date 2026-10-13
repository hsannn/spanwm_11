import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.similarity import prepare_pair


def last_token_pool(hidden, mask):
    """Pool the last non-padding token for either padding side."""
    import torch
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError('Cannot pool an empty token sequence')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def embed_texts(texts, tokenizer, model, device, batch_size, max_length):
    import torch
    vectors, lengths = [], []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            untruncated = tokenizer(batch_texts, padding=False, truncation=False)['input_ids']
            lengths.extend(len(ids) for ids in untruncated)
            batch = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=max_length, return_tensors='pt').to(device)
            hidden = model(**batch).last_hidden_state
            pooled = last_token_pool(hidden, batch['attention_mask']).float()
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(vectors), lengths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--reference-field', default='draft')
    ap.add_argument('--candidate-field', default='watermarked_text')
    ap.add_argument('--scope', choices=['continuation', 'full'], default='continuation')
    ap.add_argument('--model', default='Qwen/Qwen3-Embedding-4B')
    ap.add_argument('--revision', help='Optional Hugging Face revision/commit')
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    ap.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--local-files-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if min(args.batch_size, args.max_length) < 1 or (args.limit is not None and args.limit < 1):
        ap.error('batch-size, max-length and limit must be positive')
    if not args.dry_run:
        if args.output is None:
            ap.error('--output is required unless --dry-run is used')
        if args.output.exists():
            ap.error('Output exists; choose a new path')
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        ap.error('Input contains no records')
    pairs = []
    for line, row in enumerate(rows, 1):
        try:
            pairs.append(prepare_pair(row, args.reference_field, args.candidate_field, args.scope))
        except (KeyError, ValueError) as exc:
            ap.error(f'Record {line}: {exc}')
    settings = dict(model=args.model, revision=args.revision, scope=args.scope,
                    reference_field=args.reference_field, candidate_field=args.candidate_field,
                    pooling='last-token, L2-normalized', max_length=args.max_length,
                    instruction=None, strip_surrounding_whitespace=True)
    if args.dry_run:
        print(json.dumps(dict(settings=settings, requests=pairs), ensure_ascii=False, indent=2))
        return

    import torch
    from transformers import AutoModel, AutoTokenizer
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    dtype_name = args.dtype if args.dtype != 'auto' else ('float32' if device.startswith('cpu') else 'bfloat16')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              padding_side='left', local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, revision=args.revision,
                                      torch_dtype=getattr(torch, dtype_name),
                                      local_files_only=args.local_files_only).eval().to(device)
    reference, ref_lengths = embed_texts([p['reference'].strip() for p in pairs], tokenizer,
                                         model, device, args.batch_size, args.max_length)
    candidate, cand_lengths = embed_texts([p['candidate'].strip() for p in pairs], tokenizer,
                                          model, device, args.batch_size, args.max_length)
    scores = (reference * candidate).sum(dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError('Embedding model produced nonfinite similarity scores')
    values = scores.tolist()
    per_sample = [dict(index=row.get('index', i), line=i + 1, cosine=value,
                       reference_tokens=nr, candidate_tokens=nc,
                       reference_truncated=nr > args.max_length,
                       candidate_truncated=nc > args.max_length)
                  for i, (row, value, nr, nc) in enumerate(zip(rows, values, ref_lengths, cand_lengths))]
    summary = dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                   population_std=statistics.pstdev(values),
                   truncated_pairs=sum(r['reference_truncated'] or r['candidate_truncated'] for r in per_sample))
    result = dict(settings=settings, input=str(args.input.resolve()), input_sha256=sha256(raw).hexdigest(),
                  device=device, dtype=dtype_name, model_commit=getattr(model.config, '_commit_hash', None),
                  summary=summary, per_sample=per_sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
