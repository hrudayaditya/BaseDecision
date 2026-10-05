"""Reference HTTP server for the Jev/SystemOne API: ``POST /v1/systemone``.

    python examples/systemone_server.py --model /path/to/model
    curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d @request.json

This is a *reference*, not a hardened production server (see SECURITY.md): put it behind
your own gateway for TLS, authentication, rate limiting and quotas. What it does:

* binds to 127.0.0.1 by default and refuses a non-loopback address unless an API key is set
  (``--api-key-env NAME`` reads the key from that environment variable; clients send
  ``Authorization: Bearer <key>``);
* caps the request body (default 1 MiB), requires ``Content-Type: application/json`` and a
  ``Content-Length`` header, and applies a socket timeout against slow clients;
* never logs request bodies, answers or exception messages (only method, path, status);
* maps errors to HTTP statuses (table in ``docs/SYSTEMONE.md``) with the body
  ``{"error": {"code", "message", "field"?}}``.

Inference is serialized by the model's lock, so concurrent requests queue.
"""

import argparse
import hmac
import ipaddress
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from basedecision import SystemOne, SystemOneError
from basedecision.systemone import parse_request_body

ROUTE = "/v1/systemone"
DEFAULT_MAX_BODY_BYTES = 1 << 20
DEFAULT_TIMEOUT_SECONDS = 30.0

STATUS_BY_CODE = {
    "invalid_request": HTTPStatus.BAD_REQUEST,
    "unknown_field": HTTPStatus.BAD_REQUEST,
    "model_not_found": HTTPStatus.NOT_FOUND,
    "limit_exceeded": HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    "context_length_exceeded": HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    "unsupported_modality": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_question": HTTPStatus.UNPROCESSABLE_ENTITY,
    "invalid_criteria": HTTPStatus.UNPROCESSABLE_ENTITY,
    "unsupported_backend": HTTPStatus.NOT_IMPLEMENTED,
}


def check_bind_safety(host: str, api_key: str | None) -> None:
    """Refuse to listen on a non-loopback address without an API key."""
    if api_key:
        return
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError(
            f"refusing to listen on {host!r} without an API key; "
            "set one with --api-key-env or bind to 127.0.0.1"
        )


def make_server(
    service: SystemOne,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    api_key: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ThreadingHTTPServer:
    """Create (but do not start) the HTTP server for ``service``."""
    check_bind_safety(host, api_key)

    socket_timeout = timeout

    class Handler(BaseHTTPRequestHandler):
        server_version = "BaseDecision-SystemOne"
        sys_version = ""  # do not advertise the Python version
        timeout = socket_timeout  # applied to every socket read (slow-client protection)

        # -- helpers
        def send_error(
            self, code: int, message: str | None = None, explain: str | None = None
        ) -> None:
            """Answer protocol-level errors with JSON and fixed text (never echo the request)."""
            try:
                status = HTTPStatus(code)
            except ValueError:
                status = HTTPStatus.BAD_REQUEST
            self.close_connection = True
            self._error(status, "http_error", status.phrase)

        def _send(self, status: HTTPStatus, body: dict[str, Any], **headers: str) -> None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: HTTPStatus, code: str, message: str, **headers: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}}, **headers)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied.encode(), f"Bearer {api_key}".encode())

        # -- routes
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send(HTTPStatus.OK, {"status": "ok"})
            elif path == ROUTE:
                self._error(
                    HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "use POST", Allow="POST"
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "no such route")

        def _method_not_allowed(self) -> None:
            self._error(
                HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "use POST", Allow="POST"
            )

        do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _method_not_allowed  # noqa: N815

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != ROUTE:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "no such route")
                return
            if not self._authorized():
                self._error(
                    HTTPStatus.UNAUTHORIZED,
                    "unauthorized",
                    "missing or invalid API key",
                    WWW_Authenticate="Bearer",
                )
                return
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                self._error(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_media_type",
                    "Content-Type must be application/json",
                )
                return
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._error(
                    HTTPStatus.LENGTH_REQUIRED, "length_required", "Content-Length is required"
                )
                return
            if not (raw_length.isascii() and raw_length.isdigit() and len(raw_length) <= 12):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "invalid Content-Length")
                return
            length = int(raw_length)
            if length > max_body_bytes:
                self.close_connection = True
                self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "limit_exceeded",
                    f"request body exceeds {max_body_bytes} bytes",
                )
                return
            try:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ConnectionError("short read")
                response = service(parse_request_body(payload))
            except SystemOneError as error:
                status = STATUS_BY_CODE.get(error.code, HTTPStatus.BAD_REQUEST)
                self._send(status, error.to_dict())
            except (TimeoutError, ConnectionError):
                self.close_connection = True
                self._error(
                    HTTPStatus.REQUEST_TIMEOUT, "request_timeout", "incomplete request body"
                )
            except Exception as exc:
                sys.stderr.write(f"internal error: {type(exc).__name__}\n")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "internal error")
            else:
                self._send(HTTPStatus.OK, dict(response))

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128  # the default backlog of 5 resets bursts of connections

    return Server((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the exported checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["fp32", "bf16"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="environment variable holding the bearer token clients must send",
    )
    parser.add_argument(
        "--served-model",
        action="append",
        dest="served",
        help="accepted value(s) of the request's 'model' field (default: any)",
    )
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    parser.add_argument("--max-questions", type=int, default=64)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        parser.error(f"environment variable {args.api_key_env} is empty or unset")
    try:
        check_bind_safety(args.host, api_key)
    except ValueError as error:
        parser.error(str(error))

    from basedecision import load

    model = load(args.model, device=args.device or "auto", precision=args.precision)
    service = SystemOne(model, accepted_models=args.served, max_questions=args.max_questions)
    server = make_server(
        service, args.host, args.port, api_key=api_key, max_body_bytes=args.max_body_bytes
    )
    print(f"serving POST {ROUTE} on http://{args.host}:{server.server_port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
