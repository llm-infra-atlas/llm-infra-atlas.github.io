"""Expose repository-level ``assets/`` through MkDocs without mixing it into docs/."""

from __future__ import annotations

from pathlib import Path

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File


def _root(config) -> Path:
    return Path(config.config_file_path).resolve().parent


def on_files(files, config):
    root = _root(config)
    asset_dir = root / "assets"
    if not asset_dir.is_dir():
        raise PluginError("repository-level assets/ directory is missing")

    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        source_uri = path.relative_to(root).as_posix()
        if files.get_file_from_path(source_uri) is not None:
            raise PluginError(f"duplicate MkDocs asset path: {source_uri}")
        files.append(
            File(
                source_uri,
                str(root),
                config.site_dir,
                config.use_directory_urls,
            )
        )
    return files


def on_serve(server, config, builder):
    server.watch(str(_root(config) / "assets"), builder)
    return server
