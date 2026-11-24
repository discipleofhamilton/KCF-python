"""Download VOT2016 sequences (frames and ground truth) for ``evaluate.py``.

The pysot demo video ``bag.avi`` is the VOT2016 ``bag`` sequence at the same 480x360 size, so
its ground truth is reused; SiamMask's ``tennis`` frames have no ground truth and are only used
for speed and visual checks.

    python benchmarks/fetch_vot.py --out benchmarks/data
"""

import argparse
import json
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

BASE = "https://data.votchallenge.net/vot2016/main/"
# Sequences used while developing and tuning; HOLDOUT was only used to check the final choice.
DEVELOPMENT = [
    "bag", "blanket", "book", "butterfly", "car2", "crossing", "glove", "godfather",
    "handball1", "motocross1", "rabbit", "racing", "sheep", "sphere", "tunnel", "wiper",
]  # fmt: skip
HOLDOUT = [
    "ball2", "bmx", "bolt2", "graduate", "iceskater1", "leaves",
    "matrix", "motocross2", "pedestrian1", "shaking", "singer3", "soccer2",
]  # fmt: skip
PYSOT_BAG = "https://github.com/STVIR/pysot/raw/master/demo/bag.avi"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="benchmarks/data")
    parser.add_argument("--sequences", nargs="*", default=DEVELOPMENT + HOLDOUT)
    args = parser.parse_args(argv)
    out = Path(args.out)
    sequences = {s["name"]: s for s in json.loads(fetch(BASE + "description.json"))["sequences"]}
    for name in args.sequences:
        target = out / "vot" / name
        if (target / "groundtruth.txt").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        meta = sequences[name]
        for url in (meta["annotations"]["url"], meta["channels"]["color"]["url"]):
            zipfile.ZipFile(BytesIO(fetch(BASE + url))).extractall(target)
        print(name, flush=True)
    bag = out / "pysot" / "bag.avi"
    if not bag.exists():
        bag.parent.mkdir(parents=True, exist_ok=True)
        bag.write_bytes(fetch(PYSOT_BAG))


if __name__ == "__main__":
    main()
