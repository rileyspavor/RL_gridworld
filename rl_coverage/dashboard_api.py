from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from rl_coverage.dashboard_data import ProjectDashboardService, QueueJobRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard data service for Coverage Gridworld runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start a local HTTP JSON API for the dashboard.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    serve.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765")
    serve.add_argument("--project-root", default=None, help="Override the project root.")

    snapshot = subparsers.add_parser("snapshot", help="Write a static JSON snapshot for the dashboard.")
    snapshot.add_argument("--output", default="results/dashboard_snapshot.json", help="Path for the snapshot JSON.")
    snapshot.add_argument("--project-root", default=None, help="Override the project root.")

    queue_add = subparsers.add_parser("queue-add", help="Create a pending queue job JSON file.")
    queue_add.add_argument("--project-root", default=None, help="Override the project root.")
    queue_add.add_argument("--config-id", default=None, help="Indexed config id to queue.")
    queue_add.add_argument("--config-path", default=None, help="Config TOML path to queue.")
    queue_add.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    queue_add.add_argument("--output-dir", default=None, help="Optional output dir override.")
    queue_add.add_argument("--notes", default="", help="Optional human note for the queued job.")
    queue_add.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        default=[],
        help="Additional arg to append to the eventual training command. Repeatable.",
    )

    queue_list = subparsers.add_parser("queue-list", help="Print current queue state as JSON.")
    queue_list.add_argument("--project-root", default=None, help="Override the project root.")

    return parser.parse_args(argv)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    service: ProjectDashboardService

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        if route == "/api/health":
            self._send_json({"ok": True, "service": "rl_coverage.dashboard_api"})
            return
        if route == "/api/index":
            self._send_json(self.service.snapshot(refresh=True))
            return
        if route == "/api/runs":
            self._send_json(self.service.list_runs(refresh=True))
            return
        if route.startswith("/api/runs/"):
            run_id = route.removeprefix("/api/runs/")
            payload = self.service.get_run(run_id, refresh=True)
            if payload is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown run id: {run_id}")
                return
            self._send_json(payload)
            return
        if route == "/api/configs":
            self._send_json(self.service.list_configs(refresh=True))
            return
        if route.startswith("/api/configs/"):
            config_id = route.removeprefix("/api/configs/")
            payload = self.service.get_config(config_id, refresh=True)
            if payload is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown config id: {config_id}")
                return
            self._send_json(payload)
            return
        if route == "/api/results":
            self._send_json(self.service.list_results(refresh=True))
            return
        if route == "/api/leaderboard":
            self._send_json(self.service.snapshot(refresh=True)["leaderboard"])
            return
        if route == "/api/queue":
            self._send_json(self.service.queue_state(refresh=True))
            return
        if route == "/api/command":
            params = parse_qs(parsed.query)
            config_path = params.get("config_path", [None])[0]
            if not config_path:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "config_path query param is required")
                return
            seed_raw = params.get("seed", [None])[0]
            output_dir = params.get("output_dir", [None])[0]
            seed = int(seed_raw) if seed_raw else None
            payload = self.service.build_train_command(config_path=config_path, seed=seed, output_dir=output_dir)
            self._send_json(payload)
            return
        if route.startswith("/artifacts/"):
            relative_path = route.removeprefix("/artifacts/")
            self._serve_artifact(relative_path)
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {route}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        if route != "/api/queue":
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {route}")
            return
        payload = self._read_json_body()
        if payload is None:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Expected a JSON request body.")
            return
        try:
            request = QueueJobRequest(
                config_id=payload.get("config_id"),
                config_path=payload.get("config_path"),
                seed=payload.get("seed"),
                output_dir=payload.get("output_dir"),
                notes=payload.get("notes", ""),
                extra_args=payload.get("extra_args") or [],
            )
            job = self.service.enqueue_job(request)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json(job, status=HTTPStatus.CREATED)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> dict[str, Any] | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return None
        try:
            raw = self.rfile.read(int(content_length))
            return json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": status.value}, status=status)

    def _serve_artifact(self, relative_path: str) -> None:
        candidate = (self.service.project_root / relative_path).resolve()
        try:
            candidate.relative_to(self.service.project_root)
        except ValueError:
            self._send_error_json(HTTPStatus.FORBIDDEN, "Artifact path escapes project root.")
            return

        if not candidate.exists() or not candidate.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Artifact not found: {relative_path}")
            return

        content_type, _ = mimetypes.guess_type(candidate.name)
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_handler(service: ProjectDashboardService):
    class Handler(DashboardRequestHandler):
        pass

    Handler.service = service
    return Handler


def command_serve(args: argparse.Namespace) -> int:
    service = ProjectDashboardService(project_root=args.project_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"[dashboard] serving {service.project_root} on http://{args.host}:{args.port}")
    print("[dashboard] endpoints: /api/index, /api/runs, /api/configs, /api/results, /api/queue, /artifacts/<path>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping")
    finally:
        server.server_close()
    return 0


def command_snapshot(args: argparse.Namespace) -> int:
    service = ProjectDashboardService(project_root=args.project_root)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = service.project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(service.snapshot(refresh=True), indent=2, sort_keys=True) + "\n")
    print(f"[dashboard] wrote snapshot to {output_path}")
    return 0


def command_queue_add(args: argparse.Namespace) -> int:
    service = ProjectDashboardService(project_root=args.project_root)
    job = service.enqueue_job(
        QueueJobRequest(
            config_id=args.config_id,
            config_path=args.config_path,
            seed=args.seed,
            output_dir=args.output_dir,
            notes=args.notes,
            extra_args=args.extra_args,
        )
    )
    print(json.dumps(job, indent=2, sort_keys=True))
    return 0


def command_queue_list(args: argparse.Namespace) -> int:
    service = ProjectDashboardService(project_root=args.project_root)
    print(json.dumps(service.queue_state(refresh=True), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "serve":
        return command_serve(args)
    if args.command == "snapshot":
        return command_snapshot(args)
    if args.command == "queue-add":
        return command_queue_add(args)
    if args.command == "queue-list":
        return command_queue_list(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
