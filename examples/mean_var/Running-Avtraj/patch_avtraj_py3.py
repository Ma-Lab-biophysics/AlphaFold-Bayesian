#!/usr/bin/env python3
"""
Patch a local PyPI avtraj installation for Python 3 compatibility.

This script:
1. Finds the avtraj package installed in the currently active Python environment.
2. Finds 2to3 / 2to3-3.10; if missing, installs 2to3 with pip.
3. Runs 2to3 on the installed avtraj package directory.
4. Applies targeted patches for known avtraj 0.0.8 Python-3 issues:
   - string.lower(...) -> .lower()
   - np.empty(..., order='C') -> np.empty(...)
5. Reports remaining problematic patterns and tests importing avtraj.

Run inside the environment where avtraj is installed:

    conda activate af-bayes
    python patch_avtraj_py3.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream output to the terminal."""
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def find_avtraj_dir() -> Path:
    """Find the avtraj package directory for the active Python interpreter."""
    spec = importlib.util.find_spec("avtraj")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "ERROR: avtraj is not installed in this Python environment.\n"
            "Install it first, for example:\n"
            "  python -m pip install avtraj"
        )

    init_file = Path(spec.origin).resolve()
    if init_file.name != "__init__.py":
        raise SystemExit(f"ERROR: Unexpected avtraj module location: {init_file}")

    avtraj_dir = init_file.parent
    if not (avtraj_dir / "__init__.py").is_file():
        raise SystemExit(f"ERROR: Cannot find {avtraj_dir / '__init__.py'}")

    return avtraj_dir


def find_or_install_2to3() -> str:
    """Find 2to3 executable; install it with pip if missing."""
    for exe in ("2to3-3.10", "2to3"):
        found = shutil.which(exe)
        if found:
            return found

    print("Could not find 2to3 or 2to3-3.10. Installing 2to3 with pip...")
    run_command([sys.executable, "-m", "pip", "install", "2to3"])

    for exe in ("2to3-3.10", "2to3"):
        found = shutil.which(exe)
        if found:
            return found

    raise SystemExit(
        "ERROR: 2to3 installation completed, but executable was not found in PATH.\n"
        "Try running:\n"
        "  python -m pip show 2to3"
    )


def run_2to3(two_to_three: str, avtraj_dir: Path) -> None:
    """Run 2to3 in-place on avtraj directory."""
    print(f"Running {two_to_three} on:\n  {avtraj_dir}")
    run_command([two_to_three, "-w", str(avtraj_dir)])


def patch_string_lower(init_file: Path) -> None:
    """
    Patch old Python-2 style string.lower(...) calls.

    Observed old avtraj 0.0.8 pattern:

        selection += "chainid " + av_functions.LETTERS[
            string.lower(
                av_parameters['chain_identifier']
            )
        ]
    """
    text = init_file.read_text()
    original = text

    # Specific observed multiline pattern, single quotes.
    old_single = """av_functions.LETTERS[
            string.lower(
                av_parameters['chain_identifier']
            )
        ]"""
    new_single = """av_functions.LETTERS[
            av_parameters['chain_identifier'].lower()
        ]"""
    text = text.replace(old_single, new_single)

    # Same pattern, double quotes.
    old_double = '''av_functions.LETTERS[
            string.lower(
                av_parameters["chain_identifier"]
            )
        ]'''
    new_double = '''av_functions.LETTERS[
            av_parameters["chain_identifier"].lower()
        ]'''
    text = text.replace(old_double, new_double)

    # More general multiline string.lower(...) patches for chain_identifier.
    # Preserve functionality by returning a lower-case chain identifier string.
    text = re.sub(
        r"string\.lower\(\s*av_parameters\[['\"]chain_identifier['\"]\]\s*\)",
        "av_parameters['chain_identifier'].lower()",
        text,
    )

    # Earlier possible simulation_type case, single or double quotes.
    text = re.sub(
        r"string\.lower\(\s*av_parameters\[['\"]simulation_type['\"]\]\s*\)",
        "av_parameters['simulation_type'].lower()",
        text,
    )

    if text != original:
        init_file.write_text(text)
        print(f"Patched string.lower(...) in {init_file}")
    else:
        print("No string.lower(...) pattern patched in __init__.py")

    text_after = init_file.read_text()
    if "string.lower" in text_after:
        print("WARNING: string.lower still remains in __init__.py:")
        for i, line in enumerate(text_after.splitlines(), 1):
            if "string.lower" in line:
                print(f"  line {i}: {line}")


def patch_numba_empty_order(av_functions_file: Path) -> None:
    """
    Remove order='C' from np.empty calls.

    Modern numba may reject this in nopython mode:
        np.empty((n_max, 4), dtype=np.float64, order='C')

    C-order is already the default, so removing order='C' is safe here.
    """
    if not av_functions_file.exists():
        print(f"WARNING: {av_functions_file} not found; skipping Numba np.empty patch.")
        return

    text = av_functions_file.read_text()
    original = text

    text = text.replace(
        "np.empty((n_max, 4), dtype=np.float64, order='C')",
        "np.empty((n_max, 4), dtype=np.float64)",
    )

    text = text.replace(
        'np.empty((n_max, 4), dtype=np.float64, order="C")',
        "np.empty((n_max, 4), dtype=np.float64)",
    )

    # General fallback for simple cases.
    text = text.replace(", order='C')", ")")
    text = text.replace(', order="C")', ")")

    if text != original:
        av_functions_file.write_text(text)
        print(f"Patched np.empty(..., order='C') in {av_functions_file}")
    else:
        print("No np.empty(..., order='C') pattern patched in av_functions.py")

    text_after = av_functions_file.read_text()
    if "order='C'" in text_after or 'order="C"' in text_after:
        print("WARNING: order='C' still remains in av_functions.py:")
        for i, line in enumerate(text_after.splitlines(), 1):
            if "order='C'" in line or 'order="C"' in line:
                print(f"  line {i}: {line}")


def report_remaining_patterns(avtraj_dir: Path) -> None:
    """Print remaining suspicious Python-2 / compatibility patterns."""
    print()
    print("Checking for remaining problematic patterns...")

    patterns = [
        (r"print ['\"]", "old-style print statements"),
        (r"string\.lower", "string.lower calls"),
        (r"order=['\"]C['\"]", "order='C' keyword"),
    ]

    for pattern, label in patterns:
        print(f"\nSearching for {label}:")
        found_any = False
        for path in sorted(avtraj_dir.rglob("*.py")):
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line):
                    found_any = True
                    print(f"{path}:{i}: {line}")

        if not found_any:
            print("  none found")


def test_import() -> None:
    """Test that avtraj imports."""
    print()
    print("Testing avtraj import...")
    import avtraj  # noqa: F401

    print("avtraj imported from:")
    print(avtraj.__file__)


def main() -> None:
    print("Using Python:")
    print(sys.executable)
    print(sys.version.replace("\n", " "))

    avtraj_dir = find_avtraj_dir()
    print("Found avtraj at:")
    print(f"  {avtraj_dir}")

    two_to_three = find_or_install_2to3()
    run_2to3(two_to_three, avtraj_dir)

    print("Running targeted Python 3 compatibility patches...")
    patch_string_lower(avtraj_dir / "__init__.py")
    patch_numba_empty_order(avtraj_dir / "av_functions.py")

    print("Targeted patches complete.")
    report_remaining_patterns(avtraj_dir)
    test_import()

    print()
    print("Done.")
    print("Now try:")
    print("  python run_avtraj.py")


if __name__ == "__main__":
    main()
