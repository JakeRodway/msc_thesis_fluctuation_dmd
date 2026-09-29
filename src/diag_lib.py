"""Loads the internal diagnostic analysis library.

The package name is kept out of the repository. Provide it in src/local_settings.py
(not tracked by git) as LIB_NAME = "<package name>", or via the DIAG_LIB environment variable.
"""

import importlib
import os
import matplotlib as mpl


def _lib_name():
    try:
        from .local_settings import LIB_NAME
        return LIB_NAME
    except ImportError:
        name = os.environ.get("DIAG_LIB")
        if not name:
            raise ImportError("Internal diagnostic library not configured: create "
                              "src/local_settings.py containing LIB_NAME = '<package name>', "
                              "or set the DIAG_LIB environment variable.")
        return name


def submodule(name):
    """Imports and returns <library>.<name>, e.g. submodule('get').

    The import is wrapped in rc_context so any matplotlib style the library
    sets on import is undone, keeping the plotting defaults unchanged.
    """
    with mpl.rc_context():
        return importlib.import_module(f"{_lib_name()}.{name}")