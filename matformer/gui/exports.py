# matformer/gui/exports.py
"""One export path for tables, text and logs: CSV, JSON, HTML or plain text.

Every page used to grow its own `asksaveasfilename` and writer. This module keeps the format
choices, the escaping and the HTML skeleton in one place, so an exported log, analysis table or
generation transcript opens the same way everywhere."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from tkinter import filedialog

TABLE_FORMATS = [("CSV", "*.csv"), ("JSON", "*.json"), ("HTML", "*.html"), ("Text", "*.txt")]
TEXT_FORMATS = [("Text", "*.txt"), ("JSON", "*.json"), ("HTML", "*.html")]


def ask(title, initialfile="", formats=TABLE_FORMATS):
    return filedialog.asksaveasfilename(
        title=title, initialfile=initialfile, filetypes=[("All files", "*.*"), *formats])


def _fmt(path):
    return Path(path).suffix.lower().lstrip(".") or "txt"


def _html_page(title, body):
    return ("<!doctype html>\n<html><head><meta charset='utf-8'><title>"
            + html.escape(title) + "</title><style>body{font-family:system-ui,sans-serif;"
            "margin:2rem;color:#1d1c1a}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #d9d7d2;padding:.35rem .5rem;text-align:left;"
            "font-size:.9rem}th{background:#f4f4f2}code{font-family:ui-monospace,monospace}"
            "</style></head><body>\n<h1>" + html.escape(title) + "</h1>\n" + body
            + "\n</body></html>\n")


def write_rows(path, rows, columns=None, title="Matformer export"):
    """Write rows (dicts) as CSV, JSON or an HTML table; text falls back to tab-separated."""
    rows = list(rows or [])
    columns = list(columns or (rows[0].keys() if rows else []))
    fmt = _fmt(path)
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False, default=str)
    elif fmt == "html":
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
            + "</tr>" for row in rows)
        Path(path).write_text(_html_page(title, f"<table><thead><tr>{head}</tr></thead>"
                                                f"<tbody>{body}</tbody></table>"), "utf-8")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\t".join(columns) + "\n")
            for row in rows:
                handle.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


def write_text(path, text, title="Matformer export"):
    fmt = _fmt(path)
    if fmt == "json":
        Path(path).write_text(json.dumps({"title": title, "text": text}, indent=2,
                                         ensure_ascii=False), "utf-8")
    elif fmt == "html":
        Path(path).write_text(_html_page(title, f"<pre>{html.escape(text)}</pre>"), "utf-8")
    else:
        Path(path).write_text(text, "utf-8")
