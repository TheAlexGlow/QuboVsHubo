"""Execute the registered full experiment and verify the complete output."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["full"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path)
    p.add_argument("--retry-errors", action="store_true")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    common = ["--graphs",args.graph_dir.resolve()] if args.graph_dir else []
    engine_graph = ["--graph-dir",args.graph_dir.resolve()] if args.graph_dir else []
    steps = [
        ["check_independent.py","preflight","--results",out,*common],
        ["run.py","full","--out",out,*engine_graph,*(["--retry-errors"] if args.retry_errors else [])],
        ["report.py","--results",out,"--out",out/"derived"],
        ["check_independent.py","postrun","--results",out,*common]]
    for step in steps:
        subprocess.run([sys.executable,*step],cwd=Path(__file__).resolve().parent,check=True)


if __name__=="__main__":
    main()
