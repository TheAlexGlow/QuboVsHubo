"""Download the two original Biq Mac instances and validate the publisher's bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
SPECS = json.loads((HERE / "inputs/models.json").read_text())["models"]


def cache_dir(directory=None):
    if directory is not None:
        return Path(directory)
    if os.environ.get("QUBO_GRAPH_DIR"):
        return Path(os.environ["QUBO_GRAPH_DIR"])
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "qubo-vs-hubo" / "biqmac"


def fetch_graph(case, directory=None):
    """Return verified bytes. A cache entry with unexpected contents is an error."""
    spec = SPECS[case]
    path = cache_dir(directory) / case
    if path.is_file():
        data = path.read_bytes()
    else:
        with urlopen(spec["source"],timeout=30) as stream:
            data = stream.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"{case}: expected {spec['sha256']}, observed {digest}")
    if not path.is_file():
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            os.replace(tmp,path)
        finally:
            tmp.unlink(missing_ok=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir",type=Path)
    args = parser.parse_args()
    for case,spec in SPECS.items():
        if "source" in spec:
            data = fetch_graph(case,args.graph_dir)
            print(f"{case} {len(data)} bytes {spec['sha256']}")
