"""
REST API server for external monitoring of agbench benchmark runs.

Provides endpoints to query live results and logs.
"""

import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, parse_qs

from .live_results import load_live_results


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the benchmark API."""

    # Class-level configuration (set by APIServer)
    results_dir: str = ""
    result_tracker: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def _send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def _send_text_response(self, text: str, status: int = 200) -> None:
        """Send a plain text response."""
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def _send_error(self, message: str, status: int = 404) -> None:
        """Send an error response."""
        self._send_json_response({"error": message}, status)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Route requests
        if path == "/api/status":
            self._handle_status()
        elif path.startswith("/api/logs/"):
            self._handle_logs(path)
        elif path == "/api/health":
            self._handle_health()
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    def _handle_status(self) -> None:
        """Handle /api/status endpoint - returns current result.json."""
        # Prefer live tracker if available
        if self.result_tracker is not None:
            try:
                data = self.result_tracker.get_result_dict()
                self._send_json_response(data)
                return
            except Exception:
                pass

        # Fall back to reading from file
        results = load_live_results(self.results_dir)
        if results is not None:
            self._send_json_response(results)
        else:
            self._send_error("No results available yet", 404)

    def _handle_logs(self, path: str) -> None:
        """Handle /api/logs/{task_id}/{rep_id} endpoint - returns console_log.txt."""
        # Parse path: /api/logs/HumanEval_0/0
        parts = path.replace("/api/logs/", "").strip("/").split("/")

        if len(parts) < 2:
            self._send_error("Invalid path. Expected /api/logs/{task_id}/{rep_id}", 400)
            return

        task_id = parts[0]
        rep_id = parts[1]

        # Construct log file path
        log_path = os.path.join(self.results_dir, task_id, rep_id, "console_log.txt")

        if not os.path.exists(log_path):
            self._send_error(f"Log file not found: {task_id}/{rep_id}", 404)
            return

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self._send_text_response(content)
        except Exception as e:
            self._send_error(f"Failed to read log file: {e}", 500)

    def _handle_health(self) -> None:
        """Handle /api/health endpoint - returns server health status."""
        self._send_json_response({
            "status": "ok",
            "results_dir": self.results_dir,
        })


class APIServer:
    """
    REST API server for benchmark monitoring.

    Runs in a background thread and provides endpoints for querying results.
    """

    def __init__(
        self,
        results_dir: str,
        port: int = 8080,
        result_tracker: Any = None,
    ) -> None:
        """
        Initialize the API server.

        Args:
            results_dir: Directory containing benchmark results
            port: Port to listen on
            result_tracker: Optional LiveResultTracker for real-time results
        """
        self.results_dir = results_dir
        self.port = port
        self.result_tracker = result_tracker
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the API server in a background thread."""
        if self._running:
            return

        # Configure the handler class
        APIHandler.results_dir = self.results_dir
        APIHandler.result_tracker = self.result_tracker

        try:
            self._server = HTTPServer(("", self.port), APIHandler)
            self._running = True

            def serve() -> None:
                while self._running and self._server is not None:
                    self._server.handle_request()

            self._thread = threading.Thread(target=serve, daemon=True)
            self._thread.start()

            print(f"API server started on port {self.port}")
            print(f"  - Status: http://localhost:{self.port}/api/status")
            print(f"  - Logs:   http://localhost:{self.port}/api/logs/{{task_id}}/{{rep_id}}")
            print(f"  - Health: http://localhost:{self.port}/api/health")

        except OSError as e:
            print(f"Failed to start API server on port {self.port}: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop the API server."""
        if not self._running:
            return

        self._running = False

        if self._server is not None:
            self._server.shutdown()
            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        print("API server stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    @property
    def url(self) -> str:
        """Get the base URL of the server."""
        return f"http://localhost:{self.port}"
