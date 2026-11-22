import os
import json
import time
import logging
from dotenv import load_dotenv

load_dotenv()

try:
    from upstash_redis import Redis
except ImportError:
    Redis = None

_local_memory_cache = {}

_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    url = (os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL") or "").strip()
    token = (os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN") or "").strip()

    if url and token:
        try:
            if Redis is not None:
                _redis_client = Redis(url=url, token=token)
                logging.info("Connected to Upstash Redis persistent cache via SDK.")
            else:
                logging.warning("upstash-redis package not installed, falling back to REST/memory.")
        except Exception as e:
            logging.warning("Failed to initialize Upstash Redis client: %s", e)
            _redis_client = None

    return _redis_client

def get_cached_transcript(video_id: str):
    if not video_id:
        return None

    cache_key = f"transcript:{video_id}"

    if cache_key in _local_memory_cache:
        logging.info("Transcript Cache HIT (Memory): %s", video_id)
        return _local_memory_cache[cache_key]

    client = get_redis_client()
    if client:
        try:
            raw_val = client.get(cache_key)
            if raw_val:
                logging.info("Transcript Cache HIT (Upstash Redis): %s", video_id)
                data = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                _local_memory_cache[cache_key] = data
                return data
        except Exception as e:
            logging.warning("Upstash Redis get error: %s", e)

    return None

def set_cached_transcript(video_id: str, transcript_data: dict, ttl_seconds: int = 604800):
    if not video_id or not transcript_data:
        return

    cache_key = f"transcript:{video_id}"

    _local_memory_cache[cache_key] = transcript_data

    client = get_redis_client()
    if client:
        try:
            payload = json.dumps(transcript_data)
            client.set(cache_key, payload, ex=ttl_seconds)
            logging.info("Transcript saved to Upstash Redis: %s (TTL: %ds)", video_id, ttl_seconds)
        except Exception as e:
            logging.warning("Upstash Redis set error: %s", e)

def get_cached_analysis(video_id: str):
    if not video_id:
        return None

    cache_key = f"analysis:{video_id}"

    if cache_key in _local_memory_cache:
        logging.info("Analysis Cache HIT (Memory): %s", video_id)
        return _local_memory_cache[cache_key]

    client = get_redis_client()
    if client:
        try:
            raw_val = client.get(cache_key)
            if raw_val:
                logging.info("Analysis Cache HIT (Upstash Redis): %s", video_id)
                data = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                _local_memory_cache[cache_key] = data
                return data
        except Exception as e:
            logging.warning("Upstash Redis get analysis error: %s", e)

    return None

def set_cached_analysis(video_id: str, analysis_data: dict, ttl_seconds: int = 604800):
    if not video_id or not analysis_data:
        return

    cache_key = f"analysis:{video_id}"

    _local_memory_cache[cache_key] = analysis_data

    client = get_redis_client()
    if client:
        try:
            payload = json.dumps(analysis_data)
            client.set(cache_key, payload, ex=ttl_seconds)
            logging.info("Analysis saved to Upstash Redis: %s (TTL: %ds)", video_id, ttl_seconds)
        except Exception as e:
            logging.warning("Upstash Redis set analysis error: %s", e)

def set_sync_payload(sync_code: str, payload_data: dict, ttl_seconds: int = 180):
    if not sync_code or not payload_data:
        return False

    code = str(sync_code).strip()
    cache_key = f"sync:{code}"

    _local_memory_cache[cache_key] = (payload_data, time.time() + ttl_seconds)

    client = get_redis_client()
    if client:
        try:
            client.set(cache_key, json.dumps(payload_data), ex=ttl_seconds)
            logging.info("Sync payload saved to Redis with code: %s (TTL: %ds)", code, ttl_seconds)
        except Exception as e:
            logging.warning("Failed to store sync code in Redis: %s", e)
    return True

def get_sync_payload(sync_code: str):
    if not sync_code:
        return None

    code = str(sync_code).strip()
    cache_key = f"sync:{code}"

    client = get_redis_client()
    if client:
        try:
            raw_val = client.get(cache_key)
            if raw_val:
                logging.info("Sync payload retrieved from Redis: %s", code)
                data = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                return data
        except Exception as e:
            logging.warning("Failed to retrieve sync code from Redis: %s", e)

    if cache_key in _local_memory_cache:
        data, exp = _local_memory_cache[cache_key]
        if time.time() < exp:
            logging.info("Sync payload retrieved from memory: %s", code)
            return data
        else:
            del _local_memory_cache[cache_key]

    return None

