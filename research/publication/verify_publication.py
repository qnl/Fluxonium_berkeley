"""Check the published handoff without importing scientific simulation code.

Run from any checkout with Python 3.9+: python research/publication/verify_publication.py
Only the standard library is required. No files are written and no numerical
calculation, figure generation, network request, or experiment is launched.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    release = (HERE / "RELEASE_FILES.txt").read_text().splitlines()
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().rstrip("\0").split("\0"))
    failures = []
    for name in release:
        if name not in tracked:
            failures.append("not tracked: " + name)
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append("missing: " + item["path"])
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append("hash/size mismatch: " + item["path"])

    dependencies = list(csv.DictReader((HERE / "DEPENDENCIES.csv").open()))
    documented = {(row["source"], row["target"]) for row in dependencies
                  if row["status"] != "published"}
    link_count = external_dependencies = 0
    for name in release:
        if not name.endswith(".md"):
            continue
        text = (ROOT / name).read_text()
        # Fenced examples are historical commands, not Markdown links.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$|`[^`\n]*`", "", text, flags=re.S)
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)\n]+)\)", text):
            target = match.group(1).strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (ROOT / name).parent / unquote(target)
            relative = str(resolved.resolve().relative_to(ROOT))
            link_count += 1
            if relative not in tracked and not any(
                    item.startswith(relative.rstrip("/") + "/") for item in tracked):
                if (name, relative) in documented:
                    external_dependencies += 1
                else:
                    failures.append("undocumented link: " + name + " -> " + relative)
    result = {
        "status": "PASS" if not failures else "FAIL",
        "release_files": len(release),
        "hashed_payload_files": len(manifest["files"]),
        "markdown_local_links_checked": link_count,
        "documented_unpublished_link_targets": external_dependencies,
        "scientific_eigensolves": 0,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
