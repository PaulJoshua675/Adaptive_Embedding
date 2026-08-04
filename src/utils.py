from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_supported_files(root_dir: str | Path) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    supported_extensions = {".txt", ".md", ".pdf"}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in supported_extensions]
    return sorted(files)


def load_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: str | Path, content: str) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    safe_content = content.encode("utf-8", errors="ignore").decode("utf-8")
    path.write_text(safe_content, encoding="utf-8")
    return path
