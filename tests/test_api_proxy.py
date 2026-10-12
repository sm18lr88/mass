# Offline tests for the logging API proxy: SSE assembly, log shape, and handle() against a fake upstream.
import http.server
import importlib.util
import itertools
import json
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

PATH = Path(__file__).resolve().parents[1] / "runtime" / "api_proxy.py"
SPEC = importlib.util.spec_from_file_location("api_proxy_under_test", PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def sse(*objs: object) -> bytes:
    lines = [f"data: {o if isinstance(o, str) else json.dumps(o)}\n\n" for o in objs]
    return "".join(lines).encode("utf-8")


class AssembleSseTest(unittest.TestCase):
    def test_content_tools_usage_done(self) -> None:
        raw = sse(
            {"id": "c1", "model": "m", "created": 7, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "th"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "ink", "content": "Hel"}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "lo",
                            "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "rea", "arguments": '{"a":'}}],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"name": "d", "arguments": "1}"}}]},
                        "finish_reason": "tool_calls",
                        "stop_reason": 5,
                    }
                ]
            },
            {"choices": [], "usage": {"total_tokens": 9}},
            "not json",
            "[DONE]",
        )
        out = proxy.assemble_sse(raw)
        self.assertEqual(out["object"], "chat.completion.reassembled")
        self.assertEqual((out["id"], out["model"], out["created"]), ("c1", "m", 7))
        self.assertEqual(out["usage"], {"total_tokens": 9})
        self.assertEqual((out["n_chunks"], out["n_bad_chunks"]), (5, 1))
        (choice,) = out["choices"]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["stop_reason"], 5)
        msg = choice["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Hello")
        self.assertEqual(msg["reasoning_content"], "think")
        self.assertEqual(msg["tool_calls"], [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"a":1}'}}])

    def test_empty_stream(self) -> None:
        out = proxy.assemble_sse(b"")
        self.assertEqual(out["choices"], [])
        self.assertEqual(out["n_chunks"], 0)


class WriteLogTest(unittest.TestCase):
    def test_json_line_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.jsonl"
            with open(p, "a", encoding="utf-8") as fh:
                proxy.write_log({"log_fh": fh}, {"seq": 0, "text": "caf\u00e9"})
                proxy.write_log({"log_fh": fh}, {"seq": 1})
            lines = p.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(x)["seq"] for x in lines], [0, 1])
        self.assertIn("caf\u00e9", lines[0])


class HandleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "api_requests.jsonl"

    async def _client(self, upstream: str) -> TestClient:
        app = web.Application()
        app["upstream"] = upstream
        app["log_fh"] = open(self.log, "a", encoding="utf-8")
        app["keep_raw_sse"] = False
        app["seq"] = itertools.count()
        app.on_startup.append(proxy.on_startup)
        app.on_cleanup.append(proxy.on_cleanup)
        app.router.add_route("*", "/{tail:.*}", proxy.handle)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    def _records(self) -> list[dict]:
        return [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()]

    async def test_json_and_stream_relay(self) -> None:
        stream_body = sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]}, "[DONE]")

        async def chat(request: web.Request) -> web.StreamResponse:
            req = await request.json()
            if req.get("stream"):
                return web.Response(body=stream_body, content_type="text/event-stream")
            return web.json_response({"echo": req})

        up_app = web.Application()
        up_app.router.add_post("/v1/chat", chat)
        upstream = TestServer(up_app, host="127.0.0.1")
        await upstream.start_server()
        self.addAsyncCleanup(upstream.close)
        client = await self._client(f"http://127.0.0.1:{upstream.port}")

        r = await client.post("/v1/chat?x=1", json={"m": 1})
        self.assertEqual(r.status, 200)
        self.assertEqual(await r.json(), {"echo": {"m": 1}})
        r = await client.post("/v1/chat", json={"stream": True})
        self.assertEqual(await r.read(), stream_body)

        plain, streamed = self._records()
        self.assertEqual((plain["seq"], plain["path"], plain["stream"]), (0, "/v1/chat?x=1", False))
        self.assertEqual(plain["request"], {"m": 1})
        self.assertEqual(plain["response"], {"echo": {"m": 1}})
        self.assertTrue(streamed["stream"])
        self.assertEqual(streamed["response_bytes"], len(stream_body))
        self.assertEqual(streamed["response"]["choices"][0]["message"]["content"], "hi")

    async def test_upstream_connection_error_gives_502_and_record(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        client = await self._client(f"http://127.0.0.1:{port}")
        r = await client.post("/v1/chat", json={"m": 1})
        self.assertEqual(r.status, 502)
        (rec,) = self._records()
        self.assertIn("proxy_error", rec)
        self.assertEqual(rec["request"], {"m": 1})
        self.assertNotIn("status", rec)


class _FakeUpstream(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def _stop_process_tree(proc: subprocess.Popen) -> None:
    # A Windows venv python.exe is a launcher with the interpreter as a child; end the whole tree.
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.terminate()
    proc.wait(timeout=30)


class ProxyProcessTest(unittest.TestCase):
    def test_proxy_process_starts_and_forwards_on_this_platform(self) -> None:
        upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.shutdown)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "api_requests.jsonl"
            proc = subprocess.Popen(
                [sys.executable, str(PATH), "--listen", str(port), "--upstream", f"http://127.0.0.1:{upstream.server_port}", "--log", str(log)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
            try:
                lines: queue.Queue[str] = queue.Queue()
                stdout = proc.stdout
                assert stdout is not None

                def pump() -> None:
                    for line in stdout:
                        lines.put(line)

                threading.Thread(target=pump, daemon=True).start()
                self.assertTrue(lines.get(timeout=30).startswith("READY"))
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as resp:
                    self.assertEqual(json.load(resp)["data"][0]["id"], "fake-model")
                self.assertIsNone(proc.poll(), "proxy process exited after serving")
            finally:
                _stop_process_tree(proc)
            record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((record["path"], record["status"]), ("/v1/models", 200))


if __name__ == "__main__":
    unittest.main()
