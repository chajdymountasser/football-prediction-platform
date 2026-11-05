# app/utils/cache.py
import time

_cache = {}

def get_cache(key):
    item = _cache.get(key)
    if not item:
        return None

    value, expires_at = item
    if time.time() > expires_at:
        del _cache[key]
        return None

    return value

def set_cache(key, value, ttl_seconds):
    _cache[key] = (value, time.time() + ttl_seconds)
