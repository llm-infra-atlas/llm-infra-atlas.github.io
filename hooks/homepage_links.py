"""Rewrite repo-root-relative chapter links on the homepage.

The root ``README.md`` is the single content source for both the GitHub repo
page and the site homepage (``docs/index.md`` pulls it in). Chapter links in
README are written as repo-relative paths like ``docs/parallel/README.md`` so
they work on GitHub. When MkDocs builds the homepage this hook strips the
``docs/`` prefix, turning them into links relative to ``index.md`` that MkDocs
resolves to directory URLs and validates under ``--strict``.

``on_page_markdown`` runs before pymdownx.snippets expands ``--8<--``, so this
hook reads README.md itself and replaces the snippet directive with the
rewritten content.
"""

from __future__ import annotations

import re
from pathlib import Path

from mkdocs.exceptions import PluginError

_SNIPPET_RE = re.compile(r'--8<--\s+"README\.md"')
# Inline markdown links whose target starts with docs/ and ends in .md,
# preserving any trailing anchor/title. Image links (![...]) are not touched.
_LINK_RE = re.compile(r"(?<!!)\]\(docs/([^)\s#]+\.md)(#[^)\s]*)?([^)]*)\)")


def _rewrite_readme(content: str, files) -> str:
    def repl(match: re.Match) -> str:
        target, anchor, rest = match.group(1), match.group(2) or "", match.group(3) or ""
        if files.get_file_from_path(target) is None:
            raise PluginError(
                f"README.md links to docs/{target}, which is not a built documentation page"
            )
        return f"]({target}{anchor}{rest})"

    return _LINK_RE.sub(repl, content)


def on_page_markdown(markdown: str, page, config, files) -> str:
    if page.file.src_uri != "index.md":
        return markdown
    if not _SNIPPET_RE.search(markdown):
        raise PluginError('docs/index.md must include the root README via --8<-- "README.md"')
    root = Path(config.config_file_path).resolve().parent
    content = (root / "README.md").read_text(encoding="utf-8")
    return _SNIPPET_RE.sub(lambda _: _rewrite_readme(content, files), markdown)
