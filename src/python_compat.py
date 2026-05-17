import sys


def ensure_int_max_str_digits_api():
    """Provide sys int digit limit helpers expected by newer torch builds.

    Some container Python builds report as Python 3.11 but do not expose
    sys.get_int_max_str_digits. Torch imports this API during startup.
    """
    if not hasattr(sys, "get_int_max_str_digits"):
        sys.get_int_max_str_digits = lambda: 0
    if not hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits = lambda maxdigits: None
