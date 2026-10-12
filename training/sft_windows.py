# Renders conversations into tokenized blocks with loss masks and packs them into SFT windows (torch-free).
"""Block rendering (any HF-style tokenizer object) and window packing for SFT data."""

from typing import Any

from training.sft_requests import vllm_postprocess

IGNORE = -100

Block = dict[str, Any]


class Renderer:
    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer
        self.gen_prefix_ids = self.tok("<|im_start|>assistant\n<think>\n", add_special_tokens=False)["input_ids"]
        self.im_end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.think_end_id = self.tok.convert_tokens_to_ids("</think>")
        self.n_empty_reasoning = 0

    def render(self, messages: list[dict[str, Any]], tools: Any, enable_thinking: bool = True) -> str:
        return self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False, enable_thinking=enable_thinking)

    @staticmethod
    def block_plan(msgs: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
        """Expected block structure: (start_idx, end_idx_exclusive, kind). Consecutive
        tool messages share one block (the template merges them into one user turn)."""
        bounds = []
        i = 0
        while i < len(msgs):
            role = msgs[i]["role"]
            if role == "tool":
                j = i
                while j < len(msgs) and msgs[j]["role"] == "tool":
                    j += 1
                bounds.append((i, j, "tool"))
                i = j
            else:
                bounds.append((i, i + 1, role))
                i += 1
        return bounds

    def segments(self, messages: list[dict[str, Any]], tools: Any) -> list[list[Block]]:
        """Split the conversation into query segments (one per user query). The Qwen
        template keeps <think> only for assistant turns after the LAST user query, so
        turns of segment s were generated with the context rendered from msgs[:end_s]
        (earlier segments' reasoning stripped). Returns a list of block lists, one per
        segment; loss masks are set only on that segment's assistant blocks."""
        msgs = vllm_postprocess(messages)
        q_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if not q_idx or msgs[0]["role"] != "system" or q_idx[0] != 1:
            raise ValueError("conversation must start with system, user")
        out = []
        for s, q in enumerate(q_idx):
            end = q_idx[s + 1] if s + 1 < len(q_idx) else len(msgs)
            sub = msgs[:end]
            plan = self.block_plan(sub)
            text = self.render(sub, tools)
            pieces = text.split("<|im_start|>")
            if pieces[0] != "":
                raise ValueError("render does not start with <|im_start|>")
            pieces = ["<|im_start|>" + p for p in pieces[1:]]
            if len(pieces) != len(plan):
                raise ValueError(f"segment {s}: {len(pieces)} <|im_start|> pieces vs {len(plan)} planned blocks (content containing the literal marker?)")
            blocks = []
            for piece, (a, b, kind) in zip(pieces, plan, strict=True):
                head = {
                    "system": "<|im_start|>system\n",
                    "user": "<|im_start|>user\n",
                    "assistant": "<|im_start|>assistant\n",
                    "tool": "<|im_start|>user\n<tool_response>",
                }[kind]
                if not piece.startswith(head):
                    raise ValueError(f"segment {s}: block {kind}@{a} does not start with {head!r}: {piece[:40]!r}")
                ids = self.tok(piece, add_special_tokens=False)["input_ids"]
                mask = [0] * len(ids)
                in_segment = a >= q
                if kind == "assistant" and in_segment:
                    n = len(self.gen_prefix_ids)
                    if ids[:n] == self.gen_prefix_ids:
                        start = n
                    elif ids[: n - 1] == self.gen_prefix_ids[:-1]:
                        # empty reasoning: "<think>\n\n</think>" merges the newlines into one
                        # token; mask it and start the loss at </think>
                        try:
                            start = ids.index(self.think_end_id, n - 1)
                        except ValueError as err:
                            raise ValueError(f"segment {s}: assistant block @{a} has no </think>") from err
                        self.n_empty_reasoning += 1
                    else:
                        raise ValueError(f"segment {s}: assistant block @{a} lacks the generation prefix")
                    try:
                        last = len(ids) - 1 - ids[::-1].index(self.im_end_id)
                    except ValueError as err:
                        raise ValueError("assistant block has no <|im_end|>") from err
                    for t in range(start, last + 1):
                        mask[t] = 1
                blocks.append({"kind": kind, "ids": ids, "mask": mask, "msg_range": [a, b], "segment": s, "is_query": (kind == "user" and a == q)})
            whole = self.tok(text, add_special_tokens=False)["input_ids"]
            cat = [t for b in blocks for t in b["ids"]]
            if whole != cat:
                raise ValueError(f"segment {s}: block-wise tokenization differs from whole ({len(whole)} vs {len(cat)})")
            out.append(blocks)
        return out


def make_windows(blocks: list[Block], window: int, overlap: int, max_header_frac: float = 0.9) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Partition this segment's loss-bearing assistant blocks into windows:
    header (system + first user) + [<= overlap tokens of preceding blocks, masked;
    always incl. the segment's query block] + contiguous blocks from a loss block."""
    assert blocks[0]["kind"] == "system" and blocks[1]["kind"] == "user"
    header = blocks[:2]
    header_len = sum(len(b["ids"]) for b in header)
    if header_len > window * max_header_frac:
        raise ValueError(f"header ({header_len} tokens) exceeds {max_header_frac:.0%} of window {window}")
    body = blocks[2:]
    q_block = next((j for j, b in enumerate(body) if b.get("is_query")), None)  # None for segment 0
    has_loss = [any(b["mask"]) for b in body]
    windows, skipped = [], []
    i = 0
    while i < len(body):
        if not has_loss[i]:
            i += 1
            continue
        ctx, used = [], 0
        j = i - 1
        while j >= 0 and used + len(body[j]["ids"]) <= overlap:
            ctx.insert(0, j)
            used += len(body[j]["ids"])
            j -= 1
        if q_block is not None and q_block < i and q_block not in ctx:
            # the segment's own query must be visible: drop the oldest ctx blocks to make room
            ctx = [c for c in ctx if c > q_block]
            ctx.insert(0, q_block)
            used = sum(len(body[c]["ids"]) for c in ctx)
        budget = window - header_len - used
        if len(body[i]["ids"]) > budget:
            if len(body[i]["ids"]) <= window - header_len:
                ctx = [q_block] if (q_block is not None and q_block < i and len(body[q_block]["ids"]) + len(body[i]["ids"]) <= window - header_len) else []
                used = sum(len(body[c]["ids"]) for c in ctx)
                budget = window - header_len - used
            else:
                skipped.append({"block": i + 2, "tokens": len(body[i]["ids"])})
                i += 1
                continue
        k, total, last_loss = i, 0, i
        while k < len(body) and total + len(body[k]["ids"]) <= budget:
            total += len(body[k]["ids"])
            if has_loss[k]:
                last_loss = k
            k += 1
        end = last_loss + 1
        ids, labels, loss_blocks = [], [], []
        for b in header:
            ids += b["ids"]
            labels += [IGNORE] * len(b["ids"])
        for c in ctx:
            ids += body[c]["ids"]
            labels += [IGNORE] * len(body[c]["ids"])
        for b_idx in range(i, end):
            b = body[b_idx]
            ids += b["ids"]
            labels += [t if m else IGNORE for t, m in zip(b["ids"], b["mask"], strict=True)]
            if has_loss[b_idx]:
                loss_blocks.append(b_idx + 2)
        windows.append({
            "input_ids": ids,
            "labels": labels,
            "n_tokens": len(ids),
            "n_loss_tokens": sum(1 for l in labels if l != IGNORE),
            "blocks": [i + 2, end + 2],
            "ctx_blocks": [c + 2 for c in ctx],
            "loss_blocks": loss_blocks,
            "header_tokens": header_len,
        })
        i = end
    return windows, skipped
