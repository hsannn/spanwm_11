
def prepare_pair(row, reference_field, candidate_field, scope):
    reference, candidate = row[reference_field], row[candidate_field]
    if not isinstance(reference, str) or not isinstance(candidate, str):
        raise ValueError('Both texts must be strings')
    if scope == 'continuation':
        prompt = row.get('prompt', '')
        if not isinstance(prompt, str):
            raise ValueError('prompt must be a string')
        if not reference.startswith(prompt) or not candidate.startswith(prompt):
            raise ValueError('Prompt mismatch; use --scope full explicitly for legacy prompt-edited data')
        reference, candidate = reference[len(prompt):], candidate[len(prompt):]
    if not reference.strip() or not candidate.strip():
        raise ValueError('Empty reference or candidate; cannot assign a similarity score')
    return {'reference': reference, 'candidate': candidate}

