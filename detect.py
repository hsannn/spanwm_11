"""Detect SpanWM on saved continuations; export scores and empirical metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer
from watermark.auto_watermark import AutoWatermark
from watermark.spanwm.spanwm import PROTOCOL
from utils.transformers_config import TransformersConfig


def rank_score(result):
    return -1.0 if result['p_value'] is None else -float(np.log10(max(result['p_value'], 1e-300)))


def metrics(positive, negative):
    labels = [1]*len(positive)+[0]*len(negative)
    scores = [rank_score(r) for r in positive+negative]
    summary = dict(num_positive=len(positive), num_negative=len(negative))
    if positive:
        n = sum(r['num_tested_tokens'] for r in positive)
        g = sum(r['num_green_tokens'] for r in positive)
        zs = [r['score'] for r in positive if r['score'] is not None]
        summary.update(N=n/len(positive), G=g/len(positive), rho=g/n if n else None,
                       z=float(np.mean(zs)) if zs else None)
    if positive and negative:
        fpr,tpr,_=roc_curve(labels,scores)
        summary['AUROC']=float(roc_auc_score(labels,scores))
        summary['TPR_interpolated']={str(t):float(np.interp(t,fpr,tpr)) for t in (.1,.05,.01,.001)}
        summary['TPR_achievable']={str(t):float(max(tpr[fpr<=t])) for t in (.1,.05,.01,.001)}
        summary['negative_FPR_resolution']=1/len(negative)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--negative',choices=['unwatermarked','natural'],default='unwatermarked')
    ap.add_argument('--local-files-only',action='store_true')
    args=ap.parse_args()
    if args.output.exists(): ap.error('Output exists; choose a new path')
    meta=json.loads(args.input.with_suffix('.meta.json').read_text())
    if meta['protocol']!=PROTOCOL: ap.error('Not a SpanWM generation file')
    if torch.device(args.device).type!=meta['device_type']:
        ap.error('Detection and generation must use the same device type (CPU/CUDA green lists differ)')
    tok=AutoTokenizer.from_pretrained(meta['model'],local_files_only=args.local_files_only)
    tc=TransformersConfig(None,tok,vocab_size=meta['vocab_size'],device=args.device)
    from watermark.spanwm import SpanWM, SpanWMConfig
    from watermark.spanwm.spanwm import DEFAULT_CONFIG
    config=SpanWMConfig(str(DEFAULT_CONFIG),tc,**meta['config'])
    wm=SpanWM(config)
    records=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    pos,neg,details=[],[],[]
    for row in records:
        if row.get('protocol')!=PROTOCOL: raise ValueError('Record protocol mismatch')
        prompt=row['prompt']; text=row['watermarked_text']
        if not text.startswith(prompt): raise ValueError('Prompt prefix changed')
        p=wm.detect_watermark(text[len(prompt):]); pos.append(p)
        negative=row.get(args.negative+'_text',''); n=None
        if negative:
            if args.negative=='unwatermarked':
                if not negative.startswith(prompt): raise ValueError('Negative prompt mismatch')
                negative=negative[len(prompt):]
            if negative:
                n=wm.detect_watermark(negative);neg.append(n)
        details.append(dict(index=row['index'],positive=p,negative=n))
    summary=metrics(pos,neg)
    summary['parse_failures']=wm.utils.extractor.n_parse_failures
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(dict(metadata=meta,negative=args.negative,summary=summary,records=details),f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
