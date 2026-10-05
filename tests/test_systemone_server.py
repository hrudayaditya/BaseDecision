"""Tests for the reference HTTP server (``examples/systemone_server.py``).

A real server runs on an ephemeral loopback port with a scripted backend, so these tests
cover the actual HTTP behaviour: routing, limits, authentication, error mapping, and the
guarantee that internal errors and request text never reach the client.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from basedecision import SystemOne

try:  # unittest discover (-s tests) vs. python -m unittest tests.<module>
    from test_systemone import REFERENCE_REQUEST, ScriptedBackend
except ImportError:
    from tests.test_systemone import REFERENCE_REQUEST, ScriptedBackend

SERVER_PATH = Path(__file__).resolve().parents[1] / "examples" / "systemone_server.py"
JSON = {"Content-Type": "application/json"}


def load_server_module() -> Any:
    spec = importlib.util.spec_from_file_location("systemone_server", SERVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SERVER_PATH.is_file(), "examples/ not present (installed copy of the tests)")
class _ServerCase(unittest.TestCase):
    """Starts a server per test class and offers HTTP helpers; defines no tests itself."""

    api_key: str | None = None
    max_body_bytes = 64 * 1024

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_server_module()
        cls.backend = ScriptedBackend()
        cls.service = SystemOne(cls.backend)
        cls.server = cls.module.make_server(
            cls.service,
            "127.0.0.1",
            0,
            api_key=cls.api_key,
            max_body_bytes=cls.max_body_bytes,
            timeout=2.0,
        )
        cls.server.RequestHandlerClass.log_message = lambda *args, **kwargs: None  # quiet tests
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(
                method, path, body=body, headers=headers if headers is not None else {}
            )
            response = connection.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), json.loads(raw)
        finally:
            connection.close()

    def post(self, payload: Any, headers: dict[str, str] | None = None) -> tuple[int, Any]:
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        status, _, parsed = self.request(
            "POST", "/v1/systemone", body, JSON if headers is None else headers
        )
        return status, parsed

    def raw_exchange(self, data: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while chunk := sock.recv(65536):
                chunks.append(chunk)
        return b"".join(chunks)


@unittest.skipUnless(SERVER_PATH.is_file(), "examples/ not present (installed copy of the tests)")
class ServerTests(_ServerCase):
    # ------------------------------------------------------------------ happy path
    def test_valid_request_returns_the_adapter_response(self) -> None:
        status, body = self.post(REFERENCE_REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(body, self.service(REFERENCE_REQUEST))

    def test_response_headers(self) -> None:
        status, headers, _ = self.request(
            "POST", "/v1/systemone", json.dumps(REFERENCE_REQUEST), JSON
        )
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Python", headers.get("Server", ""))

    def test_content_type_with_charset_is_accepted(self) -> None:
        status, _ = self.post(
            REFERENCE_REQUEST, {"Content-Type": "application/json; charset=utf-8"}
        )
        self.assertEqual(status, 200)

    def test_utf8_round_trip(self) -> None:
        body = {"model": "ünï", "state": "état ✓", "questions": {"q": {"type": "noul"}}}
        status, response = self.post(json.dumps(body, ensure_ascii=False).encode())
        self.assertEqual((status, response["model"]), (200, "ünï"))

    def test_healthz(self) -> None:
        status, _, body = self.request("GET", "/healthz")
        self.assertEqual((status, body), (200, {"status": "ok"}))

    def test_parallel_requests(self) -> None:
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: self.post(REFERENCE_REQUEST), range(24)))
        self.assertTrue(all(status == 200 for status, _ in results))

    # ------------------------------------------------------------------ error mapping
    def test_validation_errors_map_to_http_statuses(self) -> None:
        cases: list[tuple[str, Any, int, str]] = [
            ("not json", "{oops", 400, "invalid_request"),
            ("duplicate keys", '{"model":"m","model":"n"}', 400, "invalid_request"),
            ("NaN", '{"model":"m","state":NaN,"questions":{}}', 400, "invalid_request"),
            ("array body", "[]", 400, "invalid_request"),
            (
                "missing state",
                {"model": "m", "questions": {"q": {"type": "noul"}}},
                400,
                "invalid_request",
            ),
            ("unknown field", {**REFERENCE_REQUEST, "x": 1}, 400, "unknown_field"),
            ("images", {**REFERENCE_REQUEST, "images": ["a"]}, 422, "unsupported_modality"),
            (
                "bad type",
                {"model": "m", "state": "s", "questions": {"q": {"type": "x"}}},
                422,
                "invalid_question",
            ),
            (
                "one option",
                {
                    "model": "m",
                    "state": "s",
                    "questions": {"q": {"type": "choice", "criteria": {"a": "x"}}},
                },
                422,
                "invalid_criteria",
            ),
        ]
        for name, payload, status, code in cases:
            with self.subTest(name):
                got_status, body = self.post(payload)
                self.assertEqual((got_status, body["error"]["code"]), (status, code))

    def test_error_body_shape(self) -> None:
        status, body = self.post({**REFERENCE_REQUEST, "images": [1]})
        self.assertEqual(status, 422)
        self.assertEqual(set(body), {"error"})
        self.assertEqual(body["error"]["field"], "images")
        self.assertIn("message", body["error"])

    def test_context_length_maps_to_413_with_token_counts(self) -> None:
        self.backend.max_context_chars = 20
        try:
            status, body = self.post(
                {"model": "m", "state": "x" * 50, "questions": {"q": {"type": "noul"}}}
            )
        finally:
            self.backend.max_context_chars = 10**9
        self.assertEqual((status, body["error"]["code"]), (413, "context_length_exceeded"))
        self.assertEqual(body["error"]["required_tokens"], 50)

    def test_internal_errors_do_not_leak(self) -> None:
        secret = "SECRET-STATE-9917"  # noqa: S105

        class Exploding(ScriptedBackend):
            def predict_batch(self, requests: Any) -> list[Any]:
                raise RuntimeError(f"boom {secret}")

        server = self.module.make_server(SystemOne(Exploding()), "127.0.0.1", 0, timeout=2.0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = {"model": "m", "state": secret, "questions": {"q": {"type": "noul"}}}
            connection.request("POST", "/v1/systemone", json.dumps(body), JSON)
            response = connection.getresponse()
            raw = response.read().decode()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(raw)["error"]["code"], "internal_error")
        self.assertNotIn(secret, raw)
        self.assertNotIn("boom", raw)

    # ------------------------------------------------------------------ HTTP-level checks
    def test_unknown_routes_and_methods(self) -> None:
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/nope", "{}", JSON)[0], 404)
        status, headers, body = self.request("GET", "/v1/systemone")
        self.assertEqual(
            (status, headers["Allow"], body["error"]["code"]), (405, "POST", "method_not_allowed")
        )
        for method in ("PUT", "DELETE", "PATCH"):
            with self.subTest(method):
                self.assertEqual(self.request(method, "/v1/systemone", "{}", JSON)[0], 405)

    def test_query_string_is_ignored_for_routing(self) -> None:
        status, _, _ = self.request(
            "POST", "/v1/systemone?x=1", json.dumps(REFERENCE_REQUEST), JSON
        )
        self.assertEqual(status, 200)

    def test_content_type_is_required(self) -> None:
        for headers in ({}, {"Content-Type": "text/plain"}, {"Content-Type": "application/xml"}):
            with self.subTest(headers=headers):
                status, body = self.post(json.dumps(REFERENCE_REQUEST), headers)
                self.assertEqual((status, body["error"]["code"]), (415, "unsupported_media_type"))

    def test_content_length_is_required_and_validated(self) -> None:
        head = b"POST /v1/systemone HTTP/1.0\r\nContent-Type: application/json\r\n"
        missing = self.raw_exchange(head + b"\r\n{}")
        self.assertIn(b" 411 ", missing.split(b"\r\n", 1)[0])
        for bad in (b"abc", b"-5", b"\xc2\xb2", b"9" * 5000, b"1e3"):
            with self.subTest(bad=bad[:12]):
                reply = self.raw_exchange(head + b"Content-Length: " + bad + b"\r\n\r\n{}")
                self.assertIn(b" 400 ", reply.split(b"\r\n", 1)[0])

    def test_oversized_body_is_rejected_before_reading_it(self) -> None:
        body = b'{"model":"m","state":"' + b"x" * (self.max_body_bytes + 1) + b'"}'
        reply = self.raw_exchange(
            b"POST /v1/systemone HTTP/1.0\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
        )  # the body is never even sent
        self.assertIn(b" 413 ", reply.split(b"\r\n", 1)[0])

    def test_truncated_body_times_out_cleanly(self) -> None:
        reply = self.raw_exchange(
            b"POST /v1/systemone HTTP/1.0\r\nContent-Type: application/json\r\n"
            b"Content-Length: 500\r\n\r\n{"
        )
        self.assertIn(b" 408 ", reply.split(b"\r\n", 1)[0])

    def test_protocol_errors_return_json_without_echoing_the_request(self) -> None:
        reply = self.raw_exchange(b"THIS IS NOT HTTP secret-token-4410\r\n\r\n")
        # A header-less (HTTP/0.9-style) probe gets only the body, so parse from the first "{".
        error = json.loads(reply[reply.index(b"{") :])["error"]
        self.assertEqual(error["code"], "http_error")
        self.assertNotIn(b"secret-token-4410", reply)
        self.assertNotIn(b"<html", reply.lower())

    def test_the_server_survives_garbage(self) -> None:
        self.raw_exchange(b"\x00\xff\xfe" * 100)
        self.assertEqual(self.post(REFERENCE_REQUEST)[0], 200)


@unittest.skipUnless(SERVER_PATH.is_file(), "examples/ not present (installed copy of the tests)")
class AuthenticatedServerTests(_ServerCase):
    api_key = "test-key-123"

    def bearer(self, key: str) -> dict[str, str]:
        return {**JSON, "Authorization": f"Bearer {key}"}

    def test_valid_credentials_succeed(self) -> None:
        status, body = self.post(REFERENCE_REQUEST, self.bearer("test-key-123"))
        self.assertEqual((status, body["model"]), (200, "clef"))

    def test_missing_or_wrong_credentials_are_rejected(self) -> None:
        for headers in (
            JSON,
            self.bearer("wrong"),
            self.bearer("test-key-12"),  # prefix of the real key
            self.bearer("test-key-1234"),  # real key plus a character
            self.bearer(""),
            {**JSON, "Authorization": "Basic dGVzdA=="},
            {**JSON, "Authorization": "test-key-123"},  # no scheme
            {**JSON, "Authorization": "bearer test-key-123"},  # scheme is case-sensitive here
        ):
            with self.subTest(authorization=headers.get("Authorization")):
                status, response_headers, body = self.request(
                    "POST", "/v1/systemone", json.dumps(REFERENCE_REQUEST), headers
                )
                self.assertEqual((status, body["error"]["code"]), (401, "unauthorized"))
                self.assertEqual(response_headers["WWW-Authenticate"], "Bearer")

    def test_authentication_is_checked_before_the_body_is_read(self) -> None:
        reply = self.raw_exchange(
            b"POST /v1/systemone HTTP/1.0\r\nContent-Type: application/json\r\n"
            b"Content-Length: 50000\r\n\r\n"
        )
        self.assertIn(b" 401 ", reply.split(b"\r\n", 1)[0])

    def test_error_responses_do_not_reveal_the_key(self) -> None:
        _, headers, body = self.request(
            "POST", "/v1/systemone", "{}", {**JSON, "Authorization": "Bearer nope"}
        )
        self.assertNotIn("test-key-123", json.dumps(body) + json.dumps(headers))

    def test_health_and_unknown_routes_need_no_credentials(self) -> None:
        self.assertEqual(self.request("GET", "/healthz")[0], 200)
        self.assertEqual(self.request("GET", "/nope")[0], 404)


class BindSafetyTests(unittest.TestCase):
    @unittest.skipUnless(SERVER_PATH.is_file(), "examples/ not present")
    def test_non_loopback_requires_an_api_key(self) -> None:
        check = load_server_module().check_bind_safety
        for host in ("127.0.0.1", "::1", "localhost", "127.1.2.3"):
            check(host, None)
        for host in ("0.0.0.0", "::", "192.168.1.5", "example.com", ""):
            with self.subTest(host=host), self.assertRaises(ValueError):
                check(host, None)
            check(host, "some-key")


if __name__ == "__main__":
    unittest.main()
