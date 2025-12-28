import inspect
import pandas as pd

def debug_print(msg=None):
    """
    Print debug information about where this function was called.
    Optionally include a message or object (e.g., a DataFrame row).
    """
    # Get the current stack frame
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    # info.filename: file path
    # info.function: function name
    # info.lineno: line number
    print(f"[DEBUG] File: {info.filename}, Function: {info.function}, Line: {info.lineno}")
    if msg is not None:
        print(msg)
