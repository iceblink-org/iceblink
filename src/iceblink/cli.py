"""Service and operator commands. Closing a viewer never controls the service."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

from .journal import Journal
from .service import Client, private_directory, read_config, serve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path, help="Private node-local state directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Explicitly create the journal for a new installation")
    service = sub.add_parser("serve", help="Run the independent monitoring service")
    service.add_argument("--config", required=True, type=Path)
    sub.add_parser("stop-service")
    status = sub.add_parser("status")
    status.add_argument("--run")
    for name in ("events", "export"):
        events = sub.add_parser(name)
        events.add_argument("--run")
        events.add_argument("--after", type=int, default=0)
        if name == "events":
            events.add_argument("--follow", action="store_true")
        else:
            events.add_argument("--output", required=True, type=Path)
    backup = sub.add_parser("backup")
    backup.add_argument("--output", required=True, type=Path)
    clear = sub.add_parser("clear-stop")
    clear.add_argument("run")
    args = parser.parse_args(argv)
    state = private_directory(args.state_dir)
    journal = Journal(state / "journal.sqlite", create=args.command == "init")
    try:
        if args.command == "init":
            journal.event("journal_initialized", component="operator")
        elif args.command == "serve":
            asyncio.run(serve(state, journal, *read_config(args.config)))
        elif args.command == "stop-service":
            Client(state).call("stop_service")
        elif args.command == "status":
            try:
                active = Client(state).call("status")
            except (OSError, ValueError):
                active = None
            result = {"service": active}
            if args.run:
                result["run"] = {k: v for k, v in journal.run(args.run).items() if k != "capability"}
                result["health"] = journal.health(args.run)
            print(json.dumps(result, indent=2))
        elif args.command == "clear-stop":
            journal.clear(args.run)
        elif args.command == "backup":
            journal.backup(args.output)
        else:
            cursor = args.after
            stream = sys.stdout if args.command == "events" else args.output.open("x", encoding="utf-8")
            try:
                while True:
                    rows = journal.events(cursor, args.run)
                    for row in rows:
                        print(json.dumps(row, separators=(",", ":")), file=stream, flush=True)
                        cursor = row["sequence"]
                    if not rows:
                        if not getattr(args, "follow", False):
                            break
                        time.sleep(0.2)
            finally:
                if stream is not sys.stdout:
                    stream.close()
    except KeyboardInterrupt:
        pass
    finally:
        journal.close()


if __name__ == "__main__":
    main()
