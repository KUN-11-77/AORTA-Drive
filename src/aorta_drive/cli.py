"""Minimal command-line entry point for repository diagnostics."""

from __future__ import annotations

import argparse

from aorta_drive import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aorta-drive")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main() -> None:
    """Run the command-line interface."""

    build_parser().parse_args()


if __name__ == "__main__":
    main()
