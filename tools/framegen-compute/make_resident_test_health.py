#!/usr/bin/env python3
"""Inject one report-identity fault into the real health shader for owner tests."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    anchor = "stateWords[at + GSC_REPORT_EPOCH_LO_AT] = p.epochLo;"
    if source.count(anchor) != 1:
        raise ValueError("expected exactly one epoch-low report store in health shader")
    replacement = ("stateWords[at + GSC_REPORT_EPOCH_LO_AT] = "
                   "p.epochLo ^ (p.attempt == 4u ? 1u : 0u);")
    args.output.write_text(source.replace(anchor, replacement, 1))


if __name__ == "__main__":
    main()
