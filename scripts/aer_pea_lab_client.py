#!/usr/bin/env python3
"""Solver-facing client for one isolated ScienceWorld pea episode.

This file deliberately contains no world configuration, oracle access, or grader logic.  It
only forwards public commands over the per-episode Unix socket created by the operator runner.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path


def _request(socket_path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        connection.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        reader = connection.makefile("r", encoding="utf-8")
        response = reader.readline()
    if not response:
        raise RuntimeError("the lab service closed without a response")
    return json.loads(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interact with the greenhouse simulator")
    parser.add_argument(
        "command",
        choices=("task", "state", "actions", "objects", "valid", "act", "batch", "record"),
    )
    parser.add_argument("argument", nargs="?", default="")
    parser.add_argument("--socket", default="scienceworld.sock")
    args = parser.parse_args()

    payload: dict[str, object] = {"command": args.command}
    if args.command == "act":
        if not args.argument.strip():
            parser.error("act requires one quoted ScienceWorld action")
        payload["action"] = args.argument.strip()
    elif args.command == "valid":
        payload["filter"] = args.argument.strip()
    elif args.command == "batch":
        actions = [line.strip() for line in sys.stdin if line.strip()]
        if not actions:
            parser.error("batch requires one ScienceWorld action per stdin line")
        payload["actions"] = actions
    elif args.command == "record":
        if not args.argument.strip():
            parser.error("record requires one quoted JSON notebook entry")
        try:
            record = json.loads(args.argument)
        except json.JSONDecodeError as error:
            parser.error(f"record argument is not valid JSON: {error}")
        if not isinstance(record, dict):
            parser.error("record JSON must be an object")
        payload["record"] = record

    result = _request(Path(args.socket).resolve(), payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
