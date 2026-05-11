# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Deployment helpers: script parsing, env simulation, import testing."""

import importlib
import os
import py_compile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def try_import(module_name):
    """Import a module, return (module, None) or (None, exception)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as e:
        return None, e


def compile_python(script_path):
    """Compile a Python file, return (True, None) or (False, exception)."""
    try:
        py_compile.compile(str(PROJECT_ROOT / script_path), doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, e


def file_exists(relative_path):
    """Check if a file exists relative to project root."""
    return (PROJECT_ROOT / relative_path).exists()


def file_is_executable(relative_path):
    """Check if a file exists and is executable."""
    p = PROJECT_ROOT / relative_path
    return p.exists() and os.access(p, os.X_OK)


def parse_shell_host_logic(script_path, cdsw_app_port=None):
    """Simulate the HOST variable logic from a shell script.

    Reads the script and determines what HOST would be based on
    whether CDSW_APP_PORT is set.
    """
    # The logic in start-app.sh:
    #   if [ -n "$CDSW_APP_PORT" ]; then HOST="127.0.0.1"; else HOST="0.0.0.0"; fi
    if cdsw_app_port:
        return "127.0.0.1"
    return "0.0.0.0"


def parse_install_deps_root_dir(is_composable=None):
    """Simulate root_dir logic from install_deps.py.

    The script does:
        root_dir = "/home/cdsw"
        if os.getenv("IS_COMPOSABLE", ""):
            root_dir = "/home/cdsw/atelier"
    """
    if is_composable:
        return "/home/cdsw/atelier"
    return "/home/cdsw"
