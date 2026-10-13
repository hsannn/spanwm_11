import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'c4': ROOT / 'dataset/c4/processed_c4.json',
    'wmt16': ROOT / 'dataset/wmt16_de_en/validation.jsonl',
    'cnn': ROOT / 'dataset/cnn_dailymail/test-00000-of-00001.jsonl',
}
WMT16_MIN_WORDS = 10
CNN_PROMPT_WORDS = 30


def load(name, n):
    rows = []
    with SOURCES[name].open() as f:
        for line in f:
            if len(rows) >= n:
                break
            item = json.loads(line)
            if name == 'c4':
                rows.append((item['prompt'], item['natural_text']))
            elif name == 'wmt16':
                if len(item['en'].split()) >= WMT16_MIN_WORDS:
                    rows.append((item['en'], ''))
            else:
                words = item['article'].split()
                if len(words) > CNN_PROMPT_WORDS:
                    rows.append((' '.join(words[:CNN_PROMPT_WORDS]),
                                 ' '.join(words[CNN_PROMPT_WORDS:])))
    return [dict(index=i, prompt=p, natural_text=t) for i, (p, t) in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--num-samples', type=int, default=200)
    ap.add_argument('--outdir', type=Path, default=ROOT / 'dataset/prompts')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        rows = load(name, args.num_samples)
        out = args.outdir / f'{name}_n{args.num_samples}.jsonl'
        with out.open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'{name}: {len(rows)} -> {out}')


if __name__ == '__main__':
    main()
