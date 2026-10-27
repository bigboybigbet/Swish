"""
Vendor the frontend libraries listed in web/vendor.lock.json.

    python scripts/vendor.py           # download what is missing or wrong, verify everything
    python scripts/vendor.py --check   # verify files on disk only (no network); exit 1 on a mismatch
    python scripts/vendor.py --force   # re-download every package

For each package the npm tarball is downloaded from the registry, checked
against its npm `dist.integrity`, and only the listed files are copied out
(plus LICENSE/NOTICE). A file with a `url` instead of a `src` (a LICENSE
the tarball does not ship) is fetched from that pinned URL. Every file is
checked against its sha384 before it is written. Standard library only.
"""
import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT / "web" / "vendor.lock.json"
VENDOR_DIR = ROOT / "web" / "static" / "vendor"


def sri(data: bytes, algo: str = "sha384") -> str:
    """Base64 digest in the Subresource Integrity format (without the prefix)."""
    return base64.b64encode(hashlib.new(algo, data).digest()).decode()


def load_lock(path: Path = LOCK_FILE) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["packages"]


def safe_rel(rel: str) -> PurePosixPath:
    """Reject absolute paths and '..' so a lock entry can't write outside the vendor dir."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"unsafe path in lock file: {rel!r}")
    return p


def target_path(pkg: dict, dest: str) -> Path:
    return VENDOR_DIR.joinpath(*safe_rel(pkg["dir"]).parts, *safe_rel(dest).parts)


def check_package(pkg: dict) -> list:
    """Return a list of problems with this package's files on disk (empty = OK)."""
    problems = []
    for dest, spec in pkg["files"].items():
        path = target_path(pkg, dest)
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
        elif sri(path.read_bytes()) != spec["sha384"]:
            problems.append(f"sha384 mismatch {path.relative_to(ROOT)}")
    return problems


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nba-quant-vendor/1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def verify_integrity(data: bytes, integrity: str, what: str):
    algo, _, expected = integrity.partition("-")
    if sri(data, algo) != expected:
        raise SystemExit(f"integrity mismatch for {what}")


def vendor_package(pkg: dict):
    label = f"{pkg['name']}@{pkg['version']}"
    tarball = fetch(pkg["tarball"])
    verify_integrity(tarball, pkg["integrity"], label)

    out = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        for dest, spec in pkg["files"].items():
            if "src" in spec:
                member = tar.getmember("package/" + str(safe_rel(spec["src"])))
                if not member.isfile():
                    raise SystemExit(f"{label}: {spec['src']} is not a regular file")
                data = tar.extractfile(member).read()
            else:
                data = fetch(spec["url"])
            if sri(data) != spec["sha384"]:
                raise SystemExit(f"{label}: sha384 mismatch for {dest}")
            out[dest] = data

    for dest, data in out.items():
        path = target_path(pkg, dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"vendored {label} ({len(out)} files)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, no network")
    ap.add_argument("--force", action="store_true", help="re-download every package")
    args = ap.parse_args(argv)

    failed = False
    for pkg in load_lock():
        problems = check_package(pkg)
        if args.check:
            for p in problems:
                print(p)
            failed |= bool(problems)
            continue
        if problems or args.force:
            vendor_package(pkg)
            problems = check_package(pkg)
            failed |= bool(problems)
        else:
            print(f"ok {pkg['name']}@{pkg['version']}")

    if failed:
        print("vendor check FAILED", file=sys.stderr)
        return 1
    print("all vendored files match web/vendor.lock.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
