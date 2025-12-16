#!/usr/bin/env python3
"""
Lightweight helper to keep dependencies in sync.

Usage examples:
  python scripts/manage_deps.py add "numpy==1.26.4" --pip-install
  python scripts/manage_deps.py remove numpy
  python scripts/manage_deps.py freeze
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys
from typing import List, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

DEPENDENCY_BLOCK = re.compile(r"dependencies\s*=\s*\[(.*?)\]", re.S)
NAME_SPLIT = re.compile(r"[<>=! \\[]")


def _normalize_name(dep: str) -> str:
    base = NAME_SPLIT.split(dep, maxsplit=1)[0]
    return base.lower().replace("_", "-")


def _load_dependencies() -> Tuple[List[str], str]:
    text = PYPROJECT.read_text()
    match = DEPENDENCY_BLOCK.search(text)
    if not match:
        sys.exit("Could not find [project].dependencies block in pyproject.toml")

    raw_block = match.group(1)
    # Use literal_eval to parse the dependency list without extra libraries.
    deps = ast.literal_eval(f"[{raw_block}]")
    return deps, text


def _write_dependencies(deps: List[str], original_text: str) -> None:
    inner = ",\n".join(f'    "{d}"' for d in deps)
    new_block = f"dependencies = [\n{inner}\n]"
    updated = DEPENDENCY_BLOCK.sub(new_block, original_text, count=1)
    PYPROJECT.write_text(updated)


def _write_requirements(deps: List[str]) -> None:
    REQUIREMENTS.write_text("\n".join(deps) + "\n")


def add_dependency(dep: str, pip_install: bool) -> None:
    deps, text = _load_dependencies()
    dep_name = _normalize_name(dep)

    replaced = False
    for idx, existing in enumerate(deps):
        if _normalize_name(existing) == dep_name:
            if existing != dep:
                deps[idx] = dep
            replaced = True
            break

    if not replaced:
        deps.append(dep)

    deps = sorted(deps, key=_normalize_name)
    _write_dependencies(deps, text)
    _write_requirements(deps)

    if pip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)


def remove_dependency(dep: str, pip_uninstall: bool) -> None:
    deps, text = _load_dependencies()
    dep_name = _normalize_name(dep)

    filtered = [d for d in deps if _normalize_name(d) != dep_name]
    if len(filtered) == len(deps):
        sys.exit(f"Dependency '{dep}' not found in pyproject.toml")

    deps = sorted(filtered, key=_normalize_name)
    _write_dependencies(deps, text)
    _write_requirements(deps)

    if pip_uninstall:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", dep_name],
            check=True,
        )


def freeze_requirements() -> None:
    deps, text = _load_dependencies()
    deps = sorted(deps, key=_normalize_name)
    _write_dependencies(deps, text)
    _write_requirements(deps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage project dependencies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add or replace a dependency")
    add_parser.add_argument("spec", help='Package spec, e.g. "numpy==1.26.4"')
    add_parser.add_argument(
        "--pip-install",
        action="store_true",
        help="Install the package into the current environment after updating files",
    )

    rm_parser = subparsers.add_parser("remove", help="Remove a dependency by name")
    rm_parser.add_argument("name", help="Package name (no version needed)")
    rm_parser.add_argument(
        "--pip-uninstall",
        action="store_true",
        help="Uninstall the package from the current environment after updating files",
    )

    subparsers.add_parser(
        "freeze",
        help="Regenerate requirements.txt from pyproject.toml dependencies",
    )

    args = parser.parse_args()

    if args.command == "add":
        add_dependency(args.spec, args.pip_install)
    elif args.command == "remove":
        remove_dependency(args.name, args.pip_uninstall)
    elif args.command == "freeze":
        freeze_requirements()
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
