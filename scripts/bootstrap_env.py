"""Keep bundled scripts isolated from user-site Python packages."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def isolate_from_user_site() -> None:
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    filtered: list[str] = []

    for entry in sys.path:
        if not entry:
            filtered.append(entry)
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            filtered.append(entry)
            continue
        if root_str in str(resolved):
            filtered.append(entry)
            continue
        if "site-packages" in str(resolved) and "AppData\\Roaming\\Python" in str(resolved):
            continue
        filtered.append(entry)

    sys.path[:] = filtered
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    site.ENABLE_USER_SITE = False


isolate_from_user_site()
