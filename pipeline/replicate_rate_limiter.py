import time
from threading import Lock

# 6 requests per minute = 1 request every 10 seconds
MIN_INTERVAL = 11  # safe buffer

_last_call = 0
_lock = Lock()

def wait_for_slot():
    global _last_call
    with _lock:
        now = time.time()
        elapsed = now - _last_call

        if elapsed < MIN_INTERVAL:
            sleep_time = MIN_INTERVAL - elapsed
            print(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        _last_call = time.time()