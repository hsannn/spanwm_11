import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fields', nargs='+', required=True, help='fields that must have a nonempty continuation')
    args = ap.parse_args()
    kept, dropped = [], []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row['prompt']
        if all(row[f][len(prompt):].strip() for f in args.fields):
            kept.append(row)
        else:
            dropped.append(row['index'])
    with args.output.open('w') as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    args.output.with_suffix('.dropped.json').write_text(
        json.dumps(dict(kept=len(kept), dropped=dropped), indent=2) + '\n')
    print(f'{args.output}: kept {len(kept)}, dropped {dropped}')


if __name__ == '__main__':
    main()
