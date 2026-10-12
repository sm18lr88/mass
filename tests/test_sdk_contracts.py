# Offline contract tests: real openai/anthropic SDK clients over httpx2.MockTransport, exercising request building and parsing.
import json
import unittest

import httpx2
from anthropic import Anthropic
from openai import OpenAI

from evaluation_claudecodex.pairwise_providers import judge_anthropic, judge_openai
from harness_improvement.llm_backend import JsonLLMBackend, _await_background_response

VERDICT = '{"winner": "B", "rationale": "Synthetic contract verdict."}'


def openai_client(handler, base_url="http://llm.test/v1"):
    return OpenAI(api_key="test-key", base_url=base_url, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def responses_payload(status="completed", text=VERDICT):
    output = (
        []
        if status != "completed"
        else [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "judge-model",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 18,
        },
    }


def chat_payload(content, **message_extra):
    return {
        "id": "chat_1",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content, **message_extra}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    }


class LocalVLLMContractTests(unittest.TestCase):
    def backend(self, handler):
        backend = JsonLLMBackend(
            backend="local-vllm",
            model="local-model",
            api_key="EMPTY",
            base_url="http://llm.test/v1",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_output_tokens=64,
            enable_thinking=True,
            seed=7,
        )
        backend.client = openai_client(handler)
        return backend

    def test_chat_request_carries_vllm_extras_and_parses_json(self):
        seen = {}

        def handler(request):
            if request.url.path == "/v1/models":
                return httpx2.Response(200, json={"object": "list", "data": [{"id": "local-model", "object": "model", "created": 0, "owned_by": "vllm"}]})
            seen["body"] = json.loads(request.content)
            return httpx2.Response(200, json=chat_payload(VERDICT))

        backend = self.backend(handler)
        backend.verify_model()
        result = backend.call_json(system_prompt="sys", user_prompt="usr")
        body = seen["body"]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual((body["seed"], body["max_tokens"]), (7, 64))
        self.assertEqual(result.payload["winner"], "B")

    def test_unserved_model_is_rejected(self):
        backend = self.backend(lambda request: httpx2.Response(200, json={"object": "list", "data": []}))
        with self.assertRaisesRegex(RuntimeError, "does not serve"):
            backend.verify_model()

    def test_empty_content_reports_reasoning_under_either_field_name(self):
        for field in ("reasoning", "reasoning_content"):
            backend = self.backend(lambda request, f=field: httpx2.Response(200, json=chat_payload("", **{f: "thinking..."})))
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, r"reasoning_chars=11"):
                backend.call_json(system_prompt="sys", user_prompt="usr")


class ResponsesContractTests(unittest.TestCase):
    def test_responses_backend_sends_json_format_and_reasoning(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx2.Response(200, json=responses_payload())

        backend = JsonLLMBackend(backend="openai-responses", model="judge-model", api_key="test-key", reasoning_effort="high", max_output_tokens=50)
        backend.client = openai_client(handler, base_url="https://api.test/v1")
        result = backend.call_json(system_prompt="sys", user_prompt="usr")
        body = seen["body"]
        self.assertEqual((body["instructions"], body["input"]), ("sys", "usr"))
        self.assertEqual(body["text"], {"format": {"type": "json_object"}})
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(result.payload["winner"], "B")
        self.assertEqual(result.metrics["usage"]["input_tokens"], 11)

    def test_background_job_is_polled_until_completed(self):
        calls = []

        def handler(request):
            calls.append(request.method)
            status = "queued" if request.method == "POST" else "completed"
            return httpx2.Response(200, json=responses_payload(status=status))

        client = openai_client(handler, base_url="https://api.test/v1")
        queued = client.responses.create(model="judge-model", input="usr", background=True)
        final = _await_background_response(client, queued, timeout_seconds=60, sleep=lambda _: None)
        self.assertEqual((final.status, final.output_text), ("completed", VERDICT))
        self.assertEqual(calls, ["POST", "GET"])

    def test_pairwise_openai_judge_parses_verdict(self):
        client = openai_client(lambda request: httpx2.Response(200, json=responses_payload()), base_url="https://api.test/v1")
        verdict, metrics = judge_openai(client, model="judge-model", instructions="sys", user="usr", temperature=None, reasoning_effort="xhigh")
        self.assertEqual(verdict.winner, "B")
        self.assertEqual(metrics.to_json_dict()["usage"]["output_tokens"], 7)


def sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class AnthropicContractTests(unittest.TestCase):
    def test_streamed_judge_sends_adaptive_thinking_and_parses_final_message(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            body = sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "judge-model",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 4},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": VERDICT}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 12}},
                {"type": "message_stop"},
            )
            return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

        client = Anthropic(
            api_key="test-key", base_url="https://anthropic.test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )
        verdict, metrics = judge_anthropic(
            client, model="judge-model", instructions="sys", user="usr", temperature=None, reasoning_effort="max", judge_max_tokens=1024
        )
        body = seen["body"]
        self.assertTrue(body["stream"])
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "max"})
        self.assertEqual((body["system"], body["max_tokens"]), ("sys", 1024))
        self.assertEqual(verdict.winner, "B")
        usage = metrics.to_json_dict()["usage"]
        self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (10, 12))


if __name__ == "__main__":
    unittest.main()
