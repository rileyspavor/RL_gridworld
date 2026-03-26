from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

from rl_dashboard.data import DashboardRepository
from rl_dashboard.models import DashboardPaths
from rl_dashboard.queue_store import ExperimentQueue
from rl_dashboard.summaries import available_options, filter_runs, sort_runs, summarize_combinations, summarize_overview


class DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        repository: DashboardRepository,
        queue_store: ExperimentQueue,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.repository = repository
        self.queue_store = queue_store


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.startswith("/api/"):
            self._handle_api_get(path, query)
            return

        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
            return

        if path.startswith("/static/"):
            rel_path = path.removeprefix("/static/")
            self._serve_static(rel_path)
            return

        if path.startswith("/artifacts/"):
            rel_path = unquote(path.removeprefix("/artifacts/"))
            self._serve_artifact(rel_path)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/"):
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._handle_api_post(path)

    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        repository = self.server.repository
        queue_store = self.server.queue_store

        if path == "/api/health":
            self._send_json({"ok": True})
            return

        if path == "/api/overview":
            runs = repository.scan_runs()
            configs = repository.list_config_templates()
            queue_entries = queue_store.list_entries()
            combinations = summarize_combinations(runs)
            payload = {
                "repo_root": str(repository.paths.root),
                "queue_file": _path_for_client(repository.paths.root, repository.paths.queue_file),
                "overview": summarize_overview(runs, combinations, queue_entries, configs),
                "options": available_options(runs, configs),
            }
            self._send_json(payload)
            return

        if path == "/api/runs":
            runs = repository.scan_runs()
            filters = {
                key: _first(query.get(key))
                for key in ("family", "status", "algorithm", "observation", "reward", "map_suite", "q")
            }
            filtered = filter_runs(runs, filters)
            sort_by = _first(query.get("sort"))
            order = _first(query.get("order"))
            sorted_runs = sort_runs(filtered, sort_by=sort_by, order=order)
            self._send_json(
                {
                    "total": len(sorted_runs),
                    "sort": {
                        "by": sort_by or "coverage",
                        "order": "asc" if str(order).strip().lower() == "asc" else "desc",
                    },
                    "items": sorted_runs,
                }
            )
            return

        if path.startswith("/api/runs/"):
            run_id = unquote(path.removeprefix("/api/runs/"))
            record = repository.get_run(run_id)
            if record is None:
                self._send_json({"error": "Run not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(record)
            return

        if path == "/api/combinations":
            runs = repository.scan_runs()
            filters = {
                key: _first(query.get(key))
                for key in ("family", "status", "algorithm", "observation", "reward", "map_suite", "q")
            }
            filtered = filter_runs(runs, filters)
            self._send_json({"items": summarize_combinations(filtered)})
            return

        if path == "/api/configs":
            templates = repository.list_config_templates()
            self._send_json({"items": templates})
            return

        if path == "/api/results":
            self._send_json(repository.load_results_bundle())
            return

        if path == "/api/queue":
            payload = {
                "queue_file": _path_for_client(repository.paths.root, repository.paths.queue_file),
                "items": queue_store.list_entries(),
            }
            self._send_json(payload)
            return

        self._send_json({"error": "Unknown API endpoint"}, status=HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        repository = self.server.repository
        queue_store = self.server.queue_store
        payload = self._read_json_body()
        if payload is None:
            return

        templates = repository.list_config_templates()

        try:
            if path == "/api/queue/preview":
                preview = queue_store.build_preview(payload, templates)
                self._send_json(preview)
                return

            if path == "/api/queue":
                entry = queue_store.enqueue(payload, templates)
                self._send_json(entry, status=HTTPStatus.CREATED)
                return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Unknown API endpoint"}, status=HTTPStatus.NOT_FOUND)

    def _serve_static(self, relative_path: str) -> None:
        static_dir = self.server.repository.paths.static_dir.resolve()
        target = (static_dir / relative_path).resolve()
        if not _is_relative_to(target, static_dir) or not target.is_file():
            self._send_json({"error": "Static file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_file(target)

    def _serve_artifact(self, relative_path: str) -> None:
        root = self.server.repository.paths.root.resolve()
        target = (root / relative_path).resolve()
        if not _is_relative_to(target, root) or not target.is_file():
            self._send_json({"error": "Artifact not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            payload = path.read_bytes()
        except OSError:
            self._send_json({"error": "Unable to read file"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict[str, Any] | None:
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=HTTPStatus.BAD_REQUEST)
            return None

        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Expected JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return None

        if not isinstance(decoded, dict):
            self._send_json({"error": "JSON object expected"}, status=HTTPStatus.BAD_REQUEST)
            return None
        return cast(dict[str, Any], decoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _path_for_client(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _first(values: list[str] | None) -> str:
    if not values:
        return ""
    return values[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Coverage RL dashboard.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8765, type=int, help="Port for the dashboard server.")
    parser.add_argument(
        "--queue-file",
        default="report/EXPERIMENT_QUEUE.jsonl",
        help="Queue file path, relative to repo root unless absolute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    paths = DashboardPaths.from_root(root, queue_file=args.queue_file)
    repository = DashboardRepository(paths)
    queue_store = ExperimentQueue(paths.queue_file)

    server = DashboardHTTPServer((args.host, int(args.port)), DashboardRequestHandler, repository, queue_store)
    queue_display = _path_for_client(paths.root, paths.queue_file)
    print(f"[dashboard] Serving {paths.root}")
    print(f"[dashboard] Queue file: {queue_display}")
    print(f"[dashboard] Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
