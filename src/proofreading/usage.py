"""Provider-neutral cache accounting. Output includes reasoning tokens already."""


def summarize_usage(provider, usages):
    total = dict(input_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
                 uncached_input_tokens=0, output_tokens=0, requests=len(usages),
                 requests_with_cache_hits=0)
    for u in usages:
        if provider == 'anthropic':
            read = u.get('cache_read_input_tokens', 0)
            write = u.get('cache_creation_input_tokens', 0)
            uncached = u.get('input_tokens', 0)
            inputs = read + write + uncached
        elif provider == 'gemini':
            inputs = u.get('total_input_tokens', 0)
            read = u.get('total_cached_tokens', 0)
            write = 0
            uncached = inputs - read
        else:
            details = u.get('input_tokens_details', u.get('prompt_tokens_details', {})) or {}
            read = details.get('cached_tokens', 0)
            write = details.get('cache_write_tokens', 0)
            inputs = u.get('input_tokens', u.get('prompt_tokens', 0))
            uncached = inputs - read - write
        for key, value in [('input_tokens', inputs), ('cache_read_tokens', read),
                           ('cache_write_tokens', write), ('uncached_input_tokens', uncached),
                           ('output_tokens', (u.get('total_output_tokens', 0) + u.get('total_thought_tokens', 0)) if provider == 'gemini' else u.get('output_tokens', u.get('completion_tokens', 0)))]:
            total[key] += value
        total['requests_with_cache_hits'] += int(read > 0)
    total['cache_read_fraction'] = total['cache_read_tokens'] / total['input_tokens'] if total['input_tokens'] else None
    return total
