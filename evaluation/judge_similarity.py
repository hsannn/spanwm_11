import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.similarity import prepare_pair

DEFAULT_PROMPT = Path(__file__).with_name('similarity_judge_prompt.txt')


class Similarity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    score: int = Field(ge=0, le=10, strict=True)
    reason: str = Field(min_length=1)



def request_id(pair, settings):
    return sha256(json.dumps({'pair': pair, 'settings': settings},
                             sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def judge_pair(client, pair, prompt, model='gpt-5.6-luna',
               reasoning_effort='low', max_completion_tokens=2048):
    completion = client.chat.completions.parse(
        model=model,
        messages=[{'role': 'system', 'content': prompt},
                  {'role': 'user', 'content': json.dumps(pair, ensure_ascii=False)}],
        response_format=Similarity,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )
    choice = completion.choices[0]
    if choice.message.refusal:
        raise RuntimeError('Judge refused the evaluation')
    if choice.finish_reason != 'stop' or choice.message.parsed is None:
        raise RuntimeError('Incomplete judge response; increase --max-completion-tokens if needed')
    result = choice.message.parsed.model_dump()
    result.update(response_model=completion.model, response_id=completion.id,
                  usage=completion.usage.model_dump() if completion.usage else None)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, help='JSONL input; defaults to draft vs watermarked_text')
    ap.add_argument('--reference', help='Reference text for a single pair')
    ap.add_argument('--candidate', help='Candidate text for a single pair')
    ap.add_argument('--reference-field', default='draft')
    ap.add_argument('--candidate-field', default='watermarked_text')
    ap.add_argument('--scope', choices=['continuation', 'full'], default='continuation')
    ap.add_argument('--output', type=Path, help='Append-only result JSONL; required for --input')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--api-key', help='OpenAI API key supplied directly (takes precedence over file/environment)')
    ap.add_argument('--api-key-file', type=Path,
                    help='Read the API key from a private file instead of this process environment')
    ap.add_argument('--reasoning-effort', default='low', choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    ap.add_argument('--max-completion-tokens', type=int, default=2048)
    ap.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--resume', action='store_true', help='Reuse successful results only when text and settings match')
    ap.add_argument('--dry-run', action='store_true', help='Print prepared requests without API calls or output-file writes')
    args = ap.parse_args()
    # Resolve relative to the repository root, independent of the caller's directory.
    load_dotenv(ROOT / '.env', override=False)
    if args.input:
        if args.reference is not None or args.candidate is not None:
            ap.error('Use --input or --reference/--candidate, not both')
        if not args.output and not args.dry_run:
            ap.error('--input requires --output')
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        if args.reference is None or args.candidate is None:
            ap.error('Provide both --reference and --candidate, or --input')
        rows = [{args.reference_field: args.reference, args.candidate_field: args.candidate}]
    if args.workers < 1 or (args.limit is not None and args.limit < 1) or args.max_completion_tokens < 1:
        ap.error('workers, limit and max-completion-tokens must be positive')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        ap.error('Input contains no records')
    prompt = args.prompt_file.read_text()
    settings = dict(model=args.model, reasoning_effort=args.reasoning_effort,
                    max_completion_tokens=args.max_completion_tokens,
                    scope=args.scope, reference_field=args.reference_field,
                    candidate_field=args.candidate_field,
                    prompt_sha256=sha256(prompt.encode()).hexdigest())
    jobs = []
    # Validate every pair before spending API tokens; do not silently drop rows.
    for line, row in enumerate(rows, 1):
        try:
            pair = prepare_pair(row, args.reference_field, args.candidate_field, args.scope)
        except (KeyError, ValueError) as exc:
            ap.error(f'Record {line}: {exc}')
        identity = dict(source=str(args.input.resolve()) if args.input else 'single',
                        line=line, index=row.get('index', line - 1))
        jobs.append((identity, pair, request_id(pair, settings)))
    if args.dry_run:
        print(json.dumps(dict(settings=settings, prompt=prompt,
                              requests=[dict(**i, request_id=k, **p) for i, p, k in jobs]),
                         ensure_ascii=False, indent=2))
        return
    cached = {}
    if args.output and args.output.exists():
        if not args.resume:
            ap.error('Output exists; choose a new path or pass --resume')
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'ok':
                cached[row['request_id']] = row
    pending = [job for job in jobs if job[2] not in cached]
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if pending and args.api_key_file and args.api_key is None:
        try:
            api_key = args.api_key_file.expanduser().read_text().strip()
        except OSError:
            ap.error('Cannot read --api-key-file; check its path and permissions')
    if args.api_key is not None:
        api_key = args.api_key.strip()
    if pending and not api_key:
        ap.error('Set OPENAI_API_KEY in the project .env or environment, or provide --api-key/--api-key-file')
    results = [dict(cached[k], **identity) for identity, _, k in jobs if k in cached]
    if pending:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        def evaluate(job):
            identity, pair, key = job
            result = dict(**identity, request_id=key, settings=settings)
            try:
                result.update(judge_pair(client, pair, prompt, args.model,
                                         args.reasoning_effort, args.max_completion_tokens), status='ok')
            except Exception as exc:
                # Errors are NOT zero similarity. Keep them visible for retry.
                result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:500])
            return result
        with OpenAI(api_key=api_key, max_retries=3, timeout=120.0) as client:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                stream = args.output.open('a' if args.resume else 'x') if args.output else None
                try:
                    for result in pool.map(evaluate, pending):
                        results.append(result)
                        if stream:
                            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                            stream.flush()
                        else:
                            print(json.dumps(result, ensure_ascii=False, indent=2))
                finally:
                    if stream:
                        stream.close()
    scores = [r['score'] for r in results if r['status'] == 'ok']
    summary = dict(total=len(jobs), successful=len(scores), failed=len(jobs)-len(scores),
                   reused=len(jobs)-len(pending), settings=settings,
                   mean=statistics.mean(scores) if scores else None,
                   median=statistics.median(scores) if scores else None,
                   population_std=statistics.pstdev(scores) if scores else None)
    if args.output:
        args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
