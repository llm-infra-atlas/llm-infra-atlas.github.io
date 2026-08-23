"""Render and validate pinned upstream source links.

Markdown syntax:
    [[megatron-lm:megatron/core/transformer/moe/moe_utils.py#L579-L590]]
    [[fla:fla/ops/kda/]]
    [[atlas:docs/parallel/01_dp/dp_lab.ipynb|运行 DP lab]]

Repository URLs and revisions live in ``mkdocs.yml`` under
``extra.source_repositories``.  Keeping this hook deliberately small makes the
source-link contract explicit without generating or copying any documentation.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from mkdocs.exceptions import PluginError


SHORTCODE_RE = re.compile(
    r"\[\[(?P<project>[a-z0-9][a-z0-9.-]*):"
    r"(?P<path>[^\]|#]*?)"
    r"(?:#(?P<lines>L\d+(?:-L?\d+)?(?:,L?\d+(?:-L?\d+)?)*)?)?"
    r"(?:\|(?P<label>[^\]]+))?\]\]"
)
UNRESOLVED_RE = re.compile(r"\[\[[a-z0-9][a-z0-9.-]*:")
LINE_RE = re.compile(r"L?(?P<first>\d+)(?:-L?(?P<last>\d+))?$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)(\1)")
UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")

_repositories: dict[str, dict[str, str]] = {}


def _without_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub("", line)


def _validate_math(markdown: str, page_path: str) -> None:
    """Enforce one source-level math convention before Markdown rendering."""
    in_fence = False
    display_start: int | None = None

    for lineno, line in enumerate(markdown.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        prose = _without_inline_code(line)
        stripped = prose.strip()
        location = f"{page_path}:{lineno}"

        if any(marker in prose for marker in (r"\(", r"\)", r"\[", r"\]")):
            raise PluginError(
                f"{location}: legacy math delimiter; use $...$ inline or "
                "standalone $$ delimiters"
            )

        if "$$" in prose:
            if stripped != "$$":
                raise PluginError(
                    f"{location}: display math delimiter $$ must occupy its own line"
                )
            display_start = None if display_start is not None else lineno
            continue

        if display_start is not None:
            continue

        dollars = list(UNESCAPED_DOLLAR_RE.finditer(prose))
        if len(dollars) % 2:
            raise PluginError(f"{location}: unclosed inline math delimiter $")
        for opening, closing in zip(dollars[0::2], dollars[1::2]):
            content = prose[opening.end() : closing.start()]
            if not content or content[0].isspace() or content[-1].isspace():
                raise PluginError(
                    f"{location}: inline math must be non-empty and have no "
                    "whitespace next to $"
                )

        if re.match(r"^\s*#{1,6}\s", prose) and dollars:
            raise PluginError(
                f"{location}: math is not allowed in headings; use plain text or Unicode"
            )
        if re.match(r"^\s*!\[", prose) and dollars:
            raise PluginError(
                f"{location}: math is not allowed in image alt text; use plain text"
            )
        if re.search(r"(?<!!)\[[^\]]*(?<!\\)\$[^\]]*\]\(", prose):
            raise PluginError(
                f"{location}: math is not allowed in link text; use plain text or Unicode"
            )

    if display_start is not None:
        raise PluginError(
            f"{page_path}:{display_start}: unclosed display math delimiter $$"
        )


def on_config(config):
    global _repositories
    configured = config.get("extra", {}).get("source_repositories", {})
    if not configured:
        raise PluginError("extra.source_repositories is required")

    errors: list[str] = []
    for name, repository in configured.items():
        url = str(repository.get("url", "")).rstrip("/")
        commit = str(repository.get("commit", ""))
        if not url.startswith("https://github.com/"):
            errors.append(f"{name}: url must be a GitHub HTTPS URL")
        if not commit or not re.fullmatch(r"[0-9a-f]{40}|[A-Za-z0-9._/-]+", commit):
            errors.append(f"{name}: invalid commit or branch {commit!r}")
        repository["url"] = url
        repository["commit"] = commit
    if errors:
        raise PluginError("invalid source repository config:\n- " + "\n- ".join(errors))

    _repositories = configured
    return config


def on_pre_build(config) -> None:
    """Validate the shared README homepage, which snippets include later."""
    root = Path(config.config_file_path).resolve().parent
    readme = root / "README.md"
    index = root / "docs" / "index.md"
    expected_index = '--8<-- "README.md"'

    if index.read_text(encoding="utf-8").strip() != expected_index:
        raise PluginError(
            "docs/index.md must only include README.md via pymdownx.snippets"
        )
    readme_text = readme.read_text(encoding="utf-8")
    _validate_math(readme_text, "README.md")
    if UNRESOLVED_RE.search(readme_text):
        raise PluginError(
            "README.md cannot use source shortcodes because it is included after hooks; "
            "use a normal pinned URL on the shared homepage"
        )


def _target(repository: dict[str, str], path: str, fragment: str = "") -> str:
    kind = "tree" if not path or path.endswith("/") else "blob"
    encoded_path = quote(path.rstrip("/"), safe="/@:+-._~")
    suffix = f"/{encoded_path}" if encoded_path else ""
    return (
        f"{repository['url']}/{kind}/{repository['commit']}{suffix}{fragment}"
    )


def _default_label(project: str, path: str, first: int | None, last: int | None) -> str:
    if path:
        name = PurePosixPath(path.rstrip("/")).name
    else:
        name = project
    if path.endswith("/"):
        name += "/"
    if first is not None:
        name += f":L{first}"
        if last is not None:
            name += f"–L{last}"
    return f"`{name}`"


def _render(match: re.Match[str], page_path: str) -> str:
    project = match.group("project")
    path = match.group("path")
    lines = match.group("lines")
    custom_label = match.group("label")

    if project not in _repositories:
        raise PluginError(f"{page_path}: unknown source repository {project!r}")
    if not path and project == "atlas":
        raise PluginError(f"{page_path}: atlas links must include a repository path")
    if path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise PluginError(f"{page_path}: unsafe source path {path!r}")

    repository = _repositories[project]
    if not lines:
        label = custom_label or _default_label(project, path, None, None)
        return f"[{label}]({_target(repository, path)})"

    anchors: list[tuple[int, int | None]] = []
    for item in lines.split(","):
        parsed = LINE_RE.fullmatch(item)
        if parsed is None:
            raise PluginError(f"{page_path}: invalid source line range {item!r}")
        first = int(parsed.group("first"))
        last = int(parsed.group("last")) if parsed.group("last") else None
        if first < 1 or (last is not None and last < first):
            raise PluginError(f"{page_path}: invalid source line range {item!r}")
        anchors.append((first, last))

    rendered: list[str] = []
    for index, (first, last) in enumerate(anchors):
        fragment = f"#L{first}" + (f"-L{last}" if last is not None else "")
        if custom_label and index == 0:
            label = custom_label
        elif index == 0:
            label = _default_label(project, path, first, last)
        else:
            label = f"`L{first}" + (f"–L{last}" if last is not None else "") + "`"
        rendered.append(f"[{label}]({_target(repository, path, fragment)})")
    return "、".join(rendered)


def on_page_markdown(markdown, page, config, files):
    """Expand shortcodes outside fenced blocks and reject local mirror links."""
    output: list[str] = []
    in_fence = False
    page_path = page.file.src_uri
    _validate_math(markdown, page_path)

    for line in markdown.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue
        if "references/" in line:
            raise PluginError(
                f"{page_path}: local references/ path leaked into published prose; "
                "use [[project:path#Lx-Ly]]"
            )
        output.append(SHORTCODE_RE.sub(lambda match: _render(match, page_path), line))

    rendered = "".join(output)
    if UNRESOLVED_RE.search(rendered):
        raise PluginError(f"{page_path}: malformed or unresolved source shortcode")
    return rendered
