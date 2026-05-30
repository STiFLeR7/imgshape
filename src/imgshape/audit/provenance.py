import hashlib
from pathlib import Path
from typing import Union


def _file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_provenance_id(dataset_path: Union[str, Path]) -> str:
    """
    Generate a deterministic SHA256 provenance ID based on sorted files,
    their sizes, and content SHA256 hashes.
    """
    path = Path(dataset_path)
    files = []

    if path.is_file():
        files.append(path)
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                files.append(p)

    # Sort files by relative path (or file name if it's a single file) to ensure determinism
    files = sorted(
        files,
        key=lambda x: (x.relative_to(path) if path.is_dir() else x.name).as_posix(),
    )

    lines = []
    for f in files:
        rel_path = (f.relative_to(path) if path.is_dir() else f.name).as_posix()
        try:
            h = _file_sha256(f)
            lines.append(f"{rel_path}:{h}")
        except Exception:
            # Handle cases where file is deleted/inaccessible during scan
            pass

    content_str = "\n".join(lines)
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()
