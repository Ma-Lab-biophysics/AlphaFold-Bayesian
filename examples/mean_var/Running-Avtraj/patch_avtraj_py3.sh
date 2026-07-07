#!/usr/bin/env bash
set -euo pipefail

echo "Using Python:"
which python
python --version

# Find installed avtraj directory using the active Python environment
AVTRAJ_DIR=$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("avtraj")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: avtraj is not installed in this Python environment.")

init_file = Path(spec.origin).resolve()
print(init_file.parent)
PY
)

echo "Found avtraj at:"
echo "  ${AVTRAJ_DIR}"

INIT_FILE="${AVTRAJ_DIR}/__init__.py"
AV_FUNCTIONS_FILE="${AVTRAJ_DIR}/av_functions.py"

if [[ ! -f "${INIT_FILE}" ]]; then
    echo "ERROR: Cannot find ${INIT_FILE}"
    exit 1
fi

# Find or install 2to3
if command -v 2to3-3.10 >/dev/null 2>&1; then
    TWO_TO_THREE="2to3-3.10"
elif command -v 2to3 >/dev/null 2>&1; then
    TWO_TO_THREE="2to3"
else
    echo "Could not find 2to3 or 2to3-3.10. Installing 2to3 with pip..."
    python -m pip install 2to3

    if command -v 2to3 >/dev/null 2>&1; then
        TWO_TO_THREE="2to3"
    elif command -v 2to3-3.10 >/dev/null 2>&1; then
        TWO_TO_THREE="2to3-3.10"
    else
        echo "ERROR: 2to3 installation completed, but executable was not found in PATH."
        echo "Try running:"
        echo "  python -m pip show 2to3"
        exit 1
    fi
fi

echo "Running ${TWO_TO_THREE} on:"
echo "  ${AVTRAJ_DIR}"

"${TWO_TO_THREE}" -w "${AVTRAJ_DIR}"

echo "Running targeted Python 3 compatibility patches..."

python - <<PY
from pathlib import Path
import re

avtraj_dir = Path("${AVTRAJ_DIR}")
init_file = avtraj_dir / "__init__.py"
av_functions_file = avtraj_dir / "av_functions.py"

# ------------------------------------------------------------
# Patch 1: string.lower(x) -> x.lower()
# Actual old avtraj 0.0.8 pattern:
#
# selection += "chainid " + av_functions.LETTERS[
#     string.lower(
#         av_parameters['chain_identifier']
#     )
# ]
# ------------------------------------------------------------

text = init_file.read_text()
original = text

# Specific observed multiline pattern, single quotes
text = text.replace(
    """av_functions.LETTERS[
            string.lower(
                av_parameters['chain_identifier']
            )
        ]""",
    """av_functions.LETTERS[
            av_parameters['chain_identifier'].lower()
        ]""",
)

# Same pattern, double quotes
text = text.replace(
    '''av_functions.LETTERS[
            string.lower(
                av_parameters["chain_identifier"]
            )
        ]''',
    '''av_functions.LETTERS[
            av_parameters["chain_identifier"].lower()
        ]''',
)

# More general multiline string.lower(...) patches for chain_identifier
text = re.sub(
    r"string\.lower\(\s*av_parameters\[['\"]chain_identifier['\"]\]\s*\)",
    "av_parameters['chain_identifier'].lower()",
    text,
)

# Earlier possible simulation_type case, single or double quotes
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

# Report remaining string.lower occurrences
text_after = init_file.read_text()
if "string.lower" in text_after:
    print("WARNING: string.lower still remains in __init__.py:")
    for i, line in enumerate(text_after.splitlines(), 1):
        if "string.lower" in line:
            print(f"  line {i}: {line}")

# ------------------------------------------------------------
# Patch 2: remove order='C' from np.empty calls for Numba compatibility
# Observed problem:
# points = np.empty((n_max, 4), dtype=np.float64, order='C')
# ------------------------------------------------------------

if av_functions_file.exists():
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

    # General fallback for simple cases
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
else:
    print(f"WARNING: {av_functions_file} not found; skipping Numba np.empty patch.")

print("Targeted patches complete.")
PY

echo
echo "Checking for remaining problematic patterns..."

grep -R -n "print \"\\|print '" "${AVTRAJ_DIR}" || true
grep -R -n "string.lower" "${AVTRAJ_DIR}" || true
grep -R -n "order='C'\\|order=\"C\"" "${AVTRAJ_DIR}" || true

echo
echo "Testing avtraj import..."
python - <<'PY'
import avtraj
print("avtraj imported from:")
print(avtraj.__file__)
PY

echo
echo "Done."
echo "Now try:"
echo "  python run_avtraj.py"

