"""
Atomic file writes: write a temp file in the same directory, then os.replace.

Readers never see a half-written file, and a crash or a cancelled job leaves
the previous version in place (plus, at worst, a stray hidden *.tmp file).

    atomic_write_json("elo_ratings.json", elo)
    with atomic_path("master_dataset_v5.csv") as tmp:   # for writers that need a path
        df.to_csv(tmp, index=False)
"""
import json
import os
import tempfile
from contextlib import contextmanager

# mkstemp creates 0600 files; new files should get the usual umask default instead.
_UMASK = os.umask(0)
os.umask(_UMASK)


def _target_mode(path) -> int:
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        return 0o666 & ~_UMASK


@contextmanager
def atomic_path(path):
    """Yield a temp path next to `path`; replace `path` with it if the block succeeds.

    The file keeps its current permissions (new files get the umask default).
    """
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{os.path.basename(path)}.", suffix=".tmp")
    os.close(fd)
    try:
        yield tmp
        os.chmod(tmp, _target_mode(path))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path, text: str, encoding: str = "utf-8"):
    """Write text to `path` atomically (flushed and fsynced before the rename)."""
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_json(path, obj, **dump_kwargs):
    """json.dump `obj` to `path` atomically. Keyword args go to json.dumps."""
    atomic_write_text(path, json.dumps(obj, **dump_kwargs))


def atomic_joblib_dump(obj, path):
    """joblib.dump to a temp file, then replace `path` (never leaves half a model)."""
    from joblib import dump
    with atomic_path(path) as tmp:
        dump(obj, tmp)
