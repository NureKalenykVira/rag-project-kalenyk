"""Clean selected markdown documents and save them to JSONL."""

import html
import json
import random
import re
import unicodedata
from pathlib import Path

import yaml

from fetch_corpus import select_documents

ROOT = Path(__file__).resolve().parent.parent

HTML_TAGS = re.compile(
    r"</?(a|img|br|p|div|span|details|summary|table|thead|tbody|tr|td|th|"
    r"b|i|strong|em|ul|ol|li|h[1-6]|sub|sup|hr|center|font|code|pre)\b[^>]*>",
    re.IGNORECASE,
)
HTML_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
MD_IMAGES = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINKS = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MD_LINK_DEFS = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
MD_REF_LINKS = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# Shortcut links like [Slack]; not code like args[0]
MD_SHORT_LINKS = re.compile(r"(?<![\w\]])\[([A-Za-z][^\]\[]*)\](?![(\[:])")


def load_config():
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    text = text.replace("\r\n", "\n")
    return unicodedata.normalize("NFC", text)


def clean_text(text):
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENTS.sub("", text)
    text = HTML_TAGS.sub("", text)
    text = html.unescape(text)
    text = MD_IMAGES.sub("", text)
    text = MD_LINKS.sub(r"\1", text)
    text = MD_LINK_DEFS.sub("", text)
    text = MD_REF_LINKS.sub(r"\1", text)
    text = MD_SHORT_LINKS.sub(r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_title(text, doc_id):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return doc_id


def save_sample(docs, path, size, seed):
    random.seed(seed)
    sample = random.sample(docs, size)
    with open(path, "w", encoding="utf-8") as f:
        for doc in sample:
            f.write(f"===== {doc['doc_id']} ({doc['n_chars']} chars) =====\n")
            f.write(doc["text"][:1500] + "\n\n")


def main():
    config = load_config()
    raw_dir = ROOT / config["corpus"]["raw_dir"]
    params = config["cleaning"]

    _, selected = select_documents(raw_dir, config["corpus"]["exclude_dirs"],
                                   config["corpus"]["exclude_patterns"])
    docs, too_short = [], []
    chars_before = chars_after = 0
    for path in selected:
        doc_id = path.relative_to(raw_dir).as_posix()
        raw_text = read_text(path)
        text = clean_text(raw_text)
        chars_before += len(raw_text)

        if len(text) < params["min_chars"]:
            too_short.append(doc_id)
            continue

        chars_after += len(text)
        docs.append({
            "doc_id": doc_id,
            "title": get_title(text, doc_id),
            "text": text,
            "n_chars": len(text),
        })

    output_path = ROOT / params["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    save_sample(docs, ROOT / params["sample_path"], params["sample_size"], params["seed"])

    print(f"Selected documents:  {len(selected)}")
    print(f"Removed (< {params['min_chars']} chars): {len(too_short)} {too_short}")
    print(f"Kept documents:      {len(docs)}")
    print(f"Characters before:   {chars_before}")
    print(f"Characters after:    {chars_after}")


if __name__ == "__main__":
    main()