from functools import wraps
from typing import Callable, Any
import time


def log_exceptions_with_retry(
    max_retries: int = 5,
    prefix_fn: Callable[[], str] | None = None,
    rethrow: bool = False,
    retry_delay: float = 0.0,
):
    """
    Decorator that retries a function call on exception,
    logging failures and optionally rethrowing.

    Args:
        max_retries: Maximum number of attempts (including first).
        prefix_fn: Optional callable returning a log prefix (e.g. debug_print).
        rethrow: If True, re-raises the exception after final failure.
        retry_delay: Optional delay (seconds) between retries.

    Returns:
        Wrapped function with retry + logging logic.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            failures = 0
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    prefix = prefix_fn() if prefix_fn else ""
                    if failures > 0:
                        print(
                            f"{prefix} {func.__name__} succeeded "
                            f"after {failures} failed attempt(s)"
                        )
                    else:
                        print(f"{prefix} {func.__name__} succeeded")

                    return result

                except Exception as e:
                    failures += 1
                    last_exception = e
                    prefix = prefix_fn() if prefix_fn else ""

                    print(
                        f"{prefix} [FAILED] {func.__name__} "
                        f"attempt {attempt}/{max_retries} – "
                        f"{type(e).__name__}: {e}"
                    )

                    if attempt < max_retries and retry_delay > 0:
                        time.sleep(retry_delay)

            # All attempts failed
            prefix = prefix_fn() if prefix_fn else ""
            print(
                f"{prefix} [ABORTED] {func.__name__} failed "
                f"after {failures}/{max_retries} attempts"
            )

            if rethrow and last_exception is not None:
                raise last_exception

            return None

        return wrapper
    return decorator
"""
USAGE WITH EXISITING FUNCTION:
------------------------------
@log_exceptions_with_retry(
    max_retries=5,
    prefix_fn=debug_print,
    retry_delay=1.0,   # optional
)
def get_share_prices_2(
    tickers,
    start,
    end,
    base_currency,
    vol_window=20,
):
    ...

"""