from functools import wraps
from typing import Callable, Any

def log_exceptions(
    prefix_fn: Callable[[], str] | None = None,
    rethrow: bool = False,
):
    """
    Decorator to catch, log, and optionally rethrow exceptions
    from any function.

    Args:
        prefix_fn: Optional callable returning a string prefix
                   (e.g. debug_print).
        rethrow: If True, re-raises the exception after logging.

    Returns:
        Wrapped function with exception handling.
    """
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)

                if prefix_fn:
                    print(f"{prefix_fn()} {func.__name__}:\n{result}")
                else:
                    print(f"{func.__name__}:\n{result}")

                return result

            except Exception as e:
                prefix = prefix_fn() if prefix_fn else ""
                print(
                    f"{prefix} [FAILED] running {func.__name__} "
                    f"{type(e).__name__}: {e}"
                )

                if rethrow:
                    raise

                return None

        return wrapper
    return decorator
