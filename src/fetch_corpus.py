"""Download the kubernetes/community corpus and save its provenance."""

import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["corpus"]


def clone_repo(url, target):
    if target.exists():
        print(f"Repository already exists: {target}")
        return
    # Shallow clone: we need only the current files, not the history
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


def get_commit_info(repo_dir):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "-1", "--format=%H %cs"],
        capture_output=True, text=True, check=True,
    )
    commit, date = result.stdout.split()
    return commit, date


def select_documents(repo_dir, exclude_dirs, exclude_patterns):
    all_docs = sorted(repo_dir.rglob("*.md"))
    selected = []
    for path in all_docs:
        rel_path = path.relative_to(repo_dir)
        if rel_path.parts[0] in exclude_dirs:
            continue
        # Patterns are regular expressions, e.g. translations README-xx.md
        if any(re.search(pattern, rel_path.as_posix()) for pattern in exclude_patterns):
            continue
        selected.append(path)
    return all_docs, selected


def main():
    config = load_config()
    raw_dir = ROOT / config["raw_dir"]

    clone_repo(config["repo_url"], raw_dir)
    commit, commit_date = get_commit_info(raw_dir)
    all_docs, selected = select_documents(raw_dir, config["exclude_dirs"], config["exclude_patterns"])

    manifest = {
        "source": config["repo_url"],
        "commit": commit,
        "commit_date": commit_date,
        "license": "Apache-2.0",
        "excluded_dirs": config["exclude_dirs"],
        "excluded_patterns": config["exclude_patterns"],
        "docs_total": len(all_docs),
        "docs_selected": len(selected),
    }

    manifest_path = ROOT / config["manifest_path"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()