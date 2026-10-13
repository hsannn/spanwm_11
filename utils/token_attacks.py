from decimal import Decimal
import hashlib
import json
import math
import random

from utils.role_ablation import rank_score

DEFAULT_RATES = (.1, .2, .3, .4, .5)
ATTACKS = ('insertion', 'deletion', 'swap')   # swap = substitute a token with a donor token


def attack_seed(seed, index, attack, field):
    # Rate is omitted deliberately: stronger attacks extend the same edit plan.
    data = json.dumps([seed, index, attack, field], ensure_ascii=False).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big')


def edit_count(length, rate):
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Attack rates must be finite fractions between 0 and 1')
    return int(Decimal(str(rate)) * length)


def mutate_ids(ids, attack, rate, seed, pool, start=0):
    """Apply floor(rate * original length) edits, recording the exact edit plan.

    Deletion samples positions without replacement. Insertion samples original
    token gaps and donor token occurrences with replacement. A shared seed makes
    plans nested across rates, but every attack starts from the original text.
    `start` protects ids[:start]: edits (count still relative to the full
    length) only touch positions/gaps at or after it. start=0 is the original
    behaviour, with an identical random stream.
    """
    if attack not in ATTACKS:
        raise ValueError(f'Unknown attack: {attack}')
    ids = list(ids)
    count = edit_count(len(ids), rate)
    if not 0 <= start <= len(ids):
        raise ValueError('start must lie within the token sequence')
    if attack in ('deletion', 'swap') and count > len(ids) - start:
        raise ValueError('Deletion budget exceeds the editable suffix')
    rng = random.Random(seed)
    changes = []
    if attack == 'deletion':
        order = list(range(start, len(ids)))
        rng.shuffle(order)
        removed = set(order[:count])
        changes = [dict(position=i, token_id=ids[i]) for i in sorted(removed)]
        result = [token for i, token in enumerate(ids) if i not in removed]
    elif attack == 'swap':
        if count and not pool:
            raise ValueError('Swap requires a nonempty unwatermarked token pool')
        order = list(range(start, len(ids)))
        rng.shuffle(order)
        result = list(ids)
        for i in sorted(order[:count]):
            for _ in range(16):                      # redraw if the donor equals the original
                donor_index = rng.randrange(len(pool))
                if pool[donor_index] != ids[i]:
                    break
            result[i] = pool[donor_index]
            changes.append(dict(position=i, token_id=ids[i], new_token_id=pool[donor_index],
                                donor_index=donor_index))
    else:
        if count and not pool:
            raise ValueError('Insertion requires a nonempty unwatermarked token pool')
        gaps = {}
        for _ in range(count):
            gap = start + rng.randrange(len(ids) + 1 - start)
            donor_index = rng.randrange(len(pool))
            token = pool[donor_index]
            changes.append(dict(gap=gap, token_id=token, donor_index=donor_index))
            gaps.setdefault(gap, []).append(token)
        result = []
        for gap in range(len(ids) + 1):
            result.extend(gaps.get(gap, []))
            if gap < len(ids):
                result.append(ids[gap])
    return result, dict(original_num_tokens=len(ids), num_edits=count,
                        realized_edit_ratio=count / len(ids) if ids else 0.,
                        edited_num_token_ids=len(result), protected_prefix_tokens=start,
                        changes=changes)


def attack_text(text, tokenizer, attack, rate, seed, pool, protect_frac=0.):
    ids = tokenizer.encode(text, add_special_tokens=False)
    edited, info = mutate_ids(ids, attack, rate, seed, pool, start=int(protect_frac * len(ids)))
    # A zero-budget edit must not introduce tokenizer normalization as an attack.
    attacked = (text if info['num_edits'] == 0 else tokenizer.decode(
        edited, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    retokenized = tokenizer.encode(attacked, add_special_tokens=False)
    info.update(seed=seed, requested_rate=rate, final_num_tokens=len(retokenized),
                retokenization_matches_edited_ids=retokenized == edited)
    return attacked, info


def donor_pool(texts, tokenizer):
    """Occurrence-weighted empirical unigram pool, excluding special-token IDs."""
    special = set(tokenizer.all_special_ids)
    return [token for text in texts for token in tokenizer.encode(text, add_special_tokens=False)
            if token not in special]


def detection_summary(positive, negative, p_threshold):
    """No interpolated ROC points; no-site samples stay at the minimum score."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    if not positive or not negative:
        raise ValueError('Both detection classes must be nonempty')
    ps = [rank_score(p) for p in positive]
    ns = [rank_score(n) for n in negative]
    labels = [1] * len(ps) + [0] * len(ns)
    fpr, tpr, _ = roc_curve(labels, ps + ns)
    z = [p['score'] for p in positive if p['score'] is not None]
    return dict(
        num_positive=len(ps), num_negative=len(ns),
        AUROC=float(roc_auc_score(labels, ps + ns)),
        TPR_achievable={str(target): float(np.max(tpr[fpr <= target]))
                        for target in (.1, .05, .01, .001)},
        negative_FPR_resolution=1 / len(ns),
        nominal_TPR=sum(p['p_value'] is not None and p['p_value'] < p_threshold
                        for p in positive) / len(ps),
        nominal_FPR=sum(p['p_value'] is not None and p['p_value'] < p_threshold
                        for p in negative) / len(ns),
        mean_tested_tokens=sum(p['num_tested_tokens'] for p in positive) / len(ps),
        mean_green_tokens=sum(p['num_green_tokens'] for p in positive) / len(ps),
        mean_detected_sites=sum(p['num_sites'] for p in positive) / len(ps),
        no_evidence_fraction=sum(p['num_tested_tokens'] == 0 for p in positive) / len(ps),
        mean_z=sum(z) / len(z) if z else None, num_valid_z=len(z),
    )


def summarize_condition(rows, clean_negative, threshold):
    positive = [r['positive'] for r in rows]
    attacked_negative = [r['negative'] for r in rows]
    result = dict(vs_clean_negative=detection_summary(positive, clean_negative, threshold),
                  vs_attacked_negative=detection_summary(positive, attacked_negative, threshold))
    for key in ('positive_attack', 'negative_attack'):
        info = [r[key] for r in rows]
        count = sum(r['original_num_tokens'] for r in info)
        result[key] = dict(
            mean_requested_edit_ratio=sum(r['realized_edit_ratio'] for r in info) / len(info),
            pooled_edit_ratio=sum(r['num_edits'] for r in info) / count if count else 0.,
            total_edits=sum(r['num_edits'] for r in info),
            mean_final_num_tokens=sum(r['final_num_tokens'] for r in info) / len(info),
            retokenization_mismatch_fraction=sum(not r['retokenization_matches_edited_ids']
                                                for r in info) / len(info))
    return result

