"""Append-only command log for the reproduction bundle.

Usage:  python runlog.py [--log FILE] -- <command> [args...]

Runs the command with stdout/stderr passed through, then appends one JSON line
{utc, cmd, exit_code, duration_s} to commands.jsonl next to this file (or to
--log FILE).  Exit code mirrors the wrapped command.
"""
import datetime
import json
import pathlib
import subprocess
import sys
import time


def main() -> int:
    argv = sys.argv[1:]
    log = pathlib.Path(__file__).resolve().parent / "commands.jsonl"
    if argv[:1] == ["--log"]:
        log = pathlib.Path(argv[1])
        argv = argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("runlog.py: no command given", file=sys.stderr)
        return 2
    t0 = time.time()
    proc = subprocess.run(argv)
    rec = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "cmd": " ".join(argv),
        "exit_code": proc.returncode,
        "duration_s": round(time.time() - t0, 2),
    }
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
