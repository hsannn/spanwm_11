import hashlib
import math


ROLE_SETS = (
    ('R3', ('NP', 'VP', 'PP')),
    ('R4S', ('NP', 'VP', 'PP', 'S')),
    ('R5', ('NP', 'VP', 'PP', 'S', 'SBAR')),
    ('R6', ('NP', 'VP', 'PP', 'S', 'SBAR', 'NML')),
    ('R7', ('NP', 'VP', 'PP', 'S', 'SBAR', 'NML', 'ADJP')),
    ('R4SBAR', ('NP', 'VP', 'PP', 'SBAR')),
)


def sample_seed(seed, row, stage):
    """Separate deterministic streams; condition is deliberately not a seed input."""
    data = f'spanwm-role-ablation:{seed}:{row}:{stage}'.encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')


def continuation(text, prompt):
    if not isinstance(text, str) or not text.startswith(prompt):
        raise ValueError('Generated text must retain the exact prompt prefix')
    return text[len(prompt):]


def rank_score(result):
    p = result['p_value']
    if p is None:
        return -1.0
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid detection tail probability')
    return -math.log10(max(p, 1e-300))


def final_alignment(embedded, detected, tolerance=4):
    """Ordered anchor proximity in final text, matching SpanWM's final check.

    This is an anchor diagnostic, not token identity or boundary equality.
    Missing detector sites count as unaligned embedding sites.
    """
    flags = [i < len(detected) and abs(s['anchor'] - detected[i]['anchor']) <= tolerance
             for i, s in enumerate(embedded)]
    local = [s['anchor_dist'] <= tolerance for s in embedded]
    return dict(embedded_sites=len(embedded), detected_sites=len(detected),
                final_aligned_sites=sum(flags), local_aligned_sites=sum(local),
                final_alignment_flags=flags,
                fully_verified=bool(embedded) and all(flags) and len(embedded) == len(detected))


def auc(positive, negative):
    import numpy as np
    positive = np.asarray(positive, dtype=float)
    negative = np.sort(np.asarray(negative, dtype=float))
    if not len(positive) or not len(negative):
        raise ValueError('AUROC needs both classes')
    left = np.searchsorted(negative, positive, side='left')
    right = np.searchsorted(negative, positive, side='right')
    return float(((left + right) / 2).mean() / len(negative))


def detection_metrics(positive, negative, threshold):
    import numpy as np
    from sklearn.metrics import roc_curve
    ps = [rank_score(r) for r in positive]
    ns = [rank_score(r) for r in negative]
    fpr, tpr, _ = roc_curve([1] * len(ps) + [0] * len(ns), ps + ns)
    return dict(
        num_positive=len(ps), num_negative=len(ns), AUROC=auc(ps, ns),
        TPR_at_1pct_FPR=float(np.max(tpr[fpr <= .01])),
        negative_FPR_resolution=1 / len(ns),
        nominal_FPR=sum(r['p_value'] is not None and r['p_value'] < threshold
                        for r in negative) / len(ns),
        nominal_TPR=sum(r['p_value'] is not None and r['p_value'] < threshold
                        for r in positive) / len(ps),
    )


def paired_auc_intervals(positive, negative, base_positive, base_negative, samples, seed):
    """Bootstrap prompt rows jointly across classes and both conditions.

    Each row contains one positive, an independent LM completion, and a human
    continuation associated with that prompt. Resampling rows preserves these
    dependencies and supplies a paired CI for the AUROC difference from R3.
    """
    import numpy as np
    arrays = [np.asarray(a, float) for a in
              (positive, negative, base_positive, base_negative)]
    n = len(arrays[0])
    if n < 2 or any(len(a) != n for a in arrays) or samples < 1:
        raise ValueError('Paired bootstrap needs equal row counts >= 2 and positive samples')
    rng = np.random.default_rng(seed)
    scores, deltas = np.empty(samples), np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        p, neg, bp, bn = (a[idx] for a in arrays)
        scores[i] = auc(p, neg)
        deltas[i] = scores[i] - auc(bp, bn)
    return dict(AUROC_ci95=np.percentile(scores, [2.5, 97.5]).tolist(),
                delta_AUROC_vs_R3=auc(arrays[0], arrays[1]) - auc(arrays[2], arrays[3]),
                delta_AUROC_ci95=np.percentile(deltas, [2.5, 97.5]).tolist())


def summarize_condition(rows, roles, max_spans, threshold):
    """Controls are computed on final positive continuations, including failures."""
    def mean(values):
        return sum(values) / len(values) if values else None

    pos = [r['positive'] for r in rows]
    align = [r['alignment'] for r in rows]
    embedded = sum(a['embedded_sites'] for a in align)
    by_role = {role: dict(embedded=0, final_aligned=0, detected=0) for role in roles}
    for row in rows:
        for site, ok in zip(row['embed_sites'], row['alignment']['final_alignment_flags']):
            entry = by_role[site['role']]
            entry['embedded'] += 1
            entry['final_aligned'] += int(ok)
        for site in row['positive']['sites']:
            by_role[site['role']]['detected'] += 1
    z = [p['score'] for p in pos if p['score'] is not None]
    result = dict(
        roles=list(roles), role_entropy_bits=math.log2(len(roles)), num_texts=len(rows),
        mean_tested_tokens=mean([p['num_tested_tokens'] for p in pos]),
        mean_green_tokens=mean([p['num_green_tokens'] for p in pos]),
        mean_z=mean(z), num_valid_z=len(z),
        mean_detected_sites=mean([p['num_sites'] for p in pos]),
        full_site_fraction=mean([p['num_sites'] == max_spans for p in pos]),
        no_evidence_fraction=mean([p['num_tested_tokens'] == 0 for p in pos]),
        final_site_alignment=(sum(a['final_aligned_sites'] for a in align) / embedded
                              if embedded else None),
        local_site_alignment=(sum(a['local_aligned_sites'] for a in align) / embedded
                              if embedded else None),
        final_verified_fraction=mean([a['fully_verified'] for a in align]),
        per_role=by_role,
    )
    for negative in ('unwatermarked', 'natural'):
        neg = [r[negative] for r in rows]
        result[negative] = detection_metrics(pos, neg, threshold)
        result[negative]['mean_tested_tokens'] = mean([r['num_tested_tokens'] for r in neg])
    return result


def latex_table(summaries):
    """Descriptive table only: no hardcoded performance or equivalence claims."""
    cols = [summaries[name] for name, _ in ROLE_SETS]
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{\textbf{Role-set size ablation for SpanWM.} '
             r'All conditions share prompts, drafts, and negative texts. '
             r'$N$ is measured on final watermarked continuations; alignment is '
             r'ordered final-anchor proximity within four characters. '
             r'TPR uses an achievable empirical threshold. '
             r'$4^\dagger$ replaces S with SBAR. '
             + f"Each class contains $n={cols[0]['num_texts']}$ examples." + '}',
             r'\label{tab:role-ablation}', r'\begin{tabular}{lcccccc}', r'\toprule',
             r'$|\mathcal{R}|$ & 3 & 4 & 5 & 6 & 7 & $4^\dagger$ \\',
             r'Added label & --- & +S & +SBAR & +NML & +ADJP & $S\to SBAR$ \\']

    def row(label, values, fmt='.3f', scale=1):
        cells = ['---' if v is None else format(v * scale, fmt) for v in values]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')

    row(r'Role entropy (bits)', [c['role_entropy_bits'] for c in cols], '.2f')
    lines.append(r'\midrule')
    row(r'Tested tokens $N$ / text', [c['mean_tested_tokens'] for c in cols], '.1f')
    row(r'Final site alignment (\%)', [c['final_site_alignment'] for c in cols], '.1f', 100)
    row(r'Fully verified texts (\%)', [c['final_verified_fraction'] for c in cols], '.1f', 100)
    row(r'Full site budget (\%)', [c['full_site_fraction'] for c in cols], '.1f', 100)
    row(r'No evidence (\%)', [c['no_evidence_fraction'] for c in cols], '.1f', 100)
    row(r'Mean $z$ (valid texts)', [c['mean_z'] for c in cols], '.2f')
    lines.append(r'\midrule')
    for key, name in [('unwatermarked', 'unwatermarked'), ('natural', 'human')]:
        row('AUROC vs. ' + name, [c[key]['AUROC'] for c in cols], '.4f')
        cells = ['[' + ', '.join(f'{v:.4f}' for v in c[key]['AUROC_ci95']) + ']'
                 for c in cols]
        lines.append(r'95\% CI vs. ' + name + ' & ' + ' & '.join(cells) + r' \\')
    row(r'TPR @ 1\% FPR (unwatermarked)',
        [c['unwatermarked']['TPR_at_1pct_FPR'] for c in cols])
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    return '\n'.join(lines)
