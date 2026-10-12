# Offline tests for JSON extraction, usage parsing, and bounded background polling.
"""Tests for harness_improvement.llm_backend / llm_json (no network)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from harness_improvement.llm_backend import (
    BackgroundResponseTimeout,
    JsonLLMBackend,
    _await_background_response,
    _chat_usage_metrics,
    _extract_json_object,
)
from harness_improvement.llm_json import EXCERPT_CHARS, LLMJsonError


class ExtractJsonTests(unittest.TestCase):
    def test_plain(self) -> None:
        self.assertEqual(_extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self) -> None:
        self.assertEqual(_extract_json_object('```json\n{"a": 2}\n```'), {"a": 2})

    def test_prefixed_and_suffixed(self) -> None:
        self.assertEqual(_extract_json_object('Sure! {"a": 3} Done.'), {"a": 3})

    def test_non_json_has_truncated_excerpt(self) -> None:
        with self.assertRaises(LLMJsonError) as ctx:
            _extract_json_object("x" * (EXCERPT_CHARS + 500))
        self.assertEqual(len(ctx.exception.excerpt), EXCERPT_CHARS)

    def test_invalid_json_chains_cause(self) -> None:
        with self.assertRaises(LLMJsonError) as ctx:
            _extract_json_object("pre {not json} post")
        self.assertIsNotNone(ctx.exception.__cause__)
        self.assertIn("not json", ctx.exception.excerpt)

    def test_empty_and_non_object(self) -> None:
        with self.assertRaises(LLMJsonError):
            _extract_json_object("   ")
        with self.assertRaises(LLMJsonError):
            _extract_json_object("[1, 2]")

    def test_is_runtime_error(self) -> None:
        self.assertTrue(issubclass(LLMJsonError, RuntimeError))


class _Usage:
    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return self._data


class UsageMetricsTests(unittest.TestCase):
    def test_parses_usage(self) -> None:
        resp = SimpleNamespace(
            id="r1",
            usage=_Usage(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 2},
                }
            ),
        )
        m = _chat_usage_metrics(resp, model="m")
        self.assertEqual(
            m["usage"],
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 2},
            },
        )
        self.assertEqual(m["response_id"], "r1")
        self.assertEqual(m["served_model"], "m")
        self.assertEqual(m["estimated_cost_usd"], 0.0)

    def test_missing_usage(self) -> None:
        m = _chat_usage_metrics(SimpleNamespace(id=None, usage=None), model="m")
        self.assertEqual(m["usage"]["total_tokens"], 0)
        self.assertIsNone(m["response_id"])


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Responses:
    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls = 0

    def retrieve(self, response_id: str):
        self.calls += 1
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(item, Exception):
            raise item
        return item


def _resp(status: str) -> SimpleNamespace:
    return SimpleNamespace(id="bg1", status=status, error=None, incomplete_details=None)


class PollingTests(unittest.TestCase):
    def test_times_out_with_fake_clock(self) -> None:
        clock = _Clock()
        client = SimpleNamespace(responses=_Responses([_resp("in_progress")]))
        with self.assertRaises(BackgroundResponseTimeout) as ctx:
            _await_background_response(
                client,
                _resp("queued"),
                timeout_seconds=100,
                poll_interval_seconds=30,
                clock=clock,
                sleep=clock.sleep,
            )
        self.assertIn("bg1", str(ctx.exception))
        self.assertEqual(clock.sleeps, [30, 30, 30, 30])
        self.assertIsInstance(ctx.exception, TimeoutError)

    def test_completes(self) -> None:
        clock = _Clock()
        client = SimpleNamespace(
            responses=_Responses([_resp("in_progress"), _resp("completed")])
        )
        out = _await_background_response(
            client, _resp("queued"), timeout_seconds=1000, clock=clock, sleep=clock.sleep
        )
        self.assertEqual(out.status, "completed")
        self.assertEqual(clock.sleeps, [30.0, 30.0])

    def test_poll_errors_bounded(self) -> None:
        clock = _Clock()
        responses = _Responses([ConnectionError("boom")])
        client = SimpleNamespace(responses=responses)
        with self.assertRaises(ConnectionError):
            _await_background_response(
                client, _resp("queued"), timeout_seconds=10**6, clock=clock, sleep=clock.sleep
            )
        self.assertEqual(responses.calls, 6)

    def test_failed_status_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _await_background_response(
                SimpleNamespace(), _resp("failed"), timeout_seconds=1
            )


class ClientTimeoutTests(unittest.TestCase):
    def test_explicit_timeouts(self) -> None:
        backend = JsonLLMBackend(
            backend="local-vllm",
            model="m",
            api_key="k",
            base_url="http://127.0.0.1:1/v1",
            timeout_seconds=1234.0,
        )
        timeout = backend.client.timeout
        self.assertEqual(timeout.read, 1234.0)
        self.assertEqual(timeout.connect, 30.0)
        self.assertEqual(backend.client.max_retries, 0)


if __name__ == "__main__":
    unittest.main()
