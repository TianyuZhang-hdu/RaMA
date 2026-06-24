#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


BLOCKED_DIR_PARTS = {
    "llm_scores",
    "debug_output_llm",
    "debug_output_sam",
    "json_results",
    "sam_results",
    "private_cache",
    "repro_outputs",
    "repro_training",
}
BLOCKED_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"api[_-]?key\\s*[:=]", re.IGNORECASE),
    re.compile(r"dashscope\\.aliyuncs\\.com", re.IGNORECASE),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def audit(root: Path) -> list[str]:
    issues: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in BLOCKED_DIR_PARTS for part in rel.parts):
            issues.append(f"blocked private-cache path: {rel}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            issues.append(f"large file over {MAX_FILE_BYTES} bytes: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(text):
                issues.append(f"sensitive pattern {pattern.pattern!r}: {rel}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit public release folder.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    issues = audit(args.root.resolve())
    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
