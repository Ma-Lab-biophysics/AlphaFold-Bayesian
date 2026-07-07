"""Command-line helper for generating ColabFold AF sampling scripts."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Optional


def _shell_assign(name: str, value: object) -> str:
    return f"{name}={shlex.quote(str(value))}"


def render_script(
    inputfile: str,
    outputdir: str,
    max_seq: int = 512,
    max_extra_seq: Optional[int] = None,
    ncycle: int = 3,
    seed: int = 0,
    nseeds: int = 100,
    nrelax: Optional[int] = None,
    rsteps: int = 100,
) -> str:
    """Return a run_AF.sh script for MSA- and seed-based ColabFold sampling."""
    if max_seq <= 0:
        raise ValueError("max_seq must be positive")
    if max_extra_seq is None:
        max_extra_seq = 2 * max_seq
    if max_extra_seq <= 0:
        raise ValueError("max_extra_seq must be positive")
    if ncycle <= 0:
        raise ValueError("ncycle must be positive")
    if nseeds <= 0:
        raise ValueError("nseeds must be positive")
    if nrelax is None:
        nrelax = 5 * nseeds
    if nrelax < 0:
        raise ValueError("nrelax must be nonnegative")
    if rsteps <= 0:
        raise ValueError("rsteps must be positive")

    assignments = "\n".join(
        [
            _shell_assign("inputfile", inputfile),
            _shell_assign("outputdir", outputdir),
            _shell_assign("max_seq", max_seq),
            _shell_assign("max_extra_seq", max_extra_seq),
            _shell_assign("ncycle", ncycle),
            _shell_assign("seed", seed),
            _shell_assign("nseeds", nseeds),
            _shell_assign("nrelax", nrelax),
            _shell_assign("rsteps", rsteps),
        ]
    )

    return f"""#!/usr/bin/env bash
set -euo pipefail

{assignments}
structures_dir="$PWD/Structures"

colabfold_batch \\
  --msa-mode mmseqs2_uniref_env \\
  --model-type alphafold2_ptm \\
  --max-msa "${{max_seq}}:${{max_extra_seq}}" \\
  --num-models 5 \\
  --num-ensemble 1 \\
  --num-recycle "$ncycle" \\
  --use-dropout \\
  --random-seed "$seed" \\
  --num-seeds "$nseeds" \\
  --rank ptm \\
  --use-gpu-relax \\
  --num-relax "$nrelax" \\
  --relax-max-iterations "$rsteps" \\
  "$inputfile" "$outputdir"

mkdir -p "$structures_dir"
find "$outputdir" -type f -name '*_relaxed_rank_*.pdb' -exec cp {{}} "$structures_dir" ';'
"""


def write_script(args: argparse.Namespace) -> Path:
    script_path = Path("run_AF.sh")
    content = render_script(
        inputfile=args.inputfile,
        outputdir=args.outputdir,
        max_seq=args.max_seq,
        max_extra_seq=args.max_extra_seq,
        ncycle=args.ncycle,
        seed=args.seed,
        nseeds=args.nseeds,
        nrelax=args.nrelax,
        rsteps=args.rsteps,
    )
    script_path.write_text(content, encoding="utf-8")
    mode = script_path.stat().st_mode
    script_path.chmod(mode | 0o111)
    return script_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="AF-sample",
        description="Generate a run_AF.sh script for ColabFold AF sampling.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    make = sub.add_parser("make-script", help="write run_AF.sh in the current directory")
    make.add_argument("--inputfile", required=True, help="input FASTA file for colabfold_batch")
    make.add_argument("--outputdir", required=True, help="ColabFold output directory")
    make.add_argument("--max-seq", type=int, default=512, help="maximum paired/unpaired MSA sequences")
    make.add_argument(
        "--max-extra-seq",
        type=int,
        default=None,
        help="maximum extra MSA sequences; defaults to 2 * --max-seq",
    )
    make.add_argument("--ncycle", type=int, default=3, help="number of AlphaFold recycles")
    make.add_argument("--seed", type=int, default=0, help="starting random seed")
    make.add_argument("--nseeds", type=int, default=100, help="number of random seeds")
    make.add_argument("--nrelax", type=int, default=None, help="number of relaxed structures; defaults to 5 * --nseeds")
    make.add_argument("--rsteps", type=int, default=100, help="maximum relaxation iterations")
    make.set_defaults(func=write_script)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        script_path = args.func(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    print(f"Wrote {script_path}")
    print("Run it manually with: bash run_AF.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
