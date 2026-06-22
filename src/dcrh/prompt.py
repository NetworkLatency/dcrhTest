from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(slots=True)
class PromptEncoding:
    input_ids: torch.LongTensor
    rendered_text: str
    question_start: int
    question_end: int
    sink_positions: tuple[int, ...]
    base_prompt_tokens: int
    shared_text_tokens: int
    control_tokens: int

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[-1])


class PromptBuilder:
    def __init__(
        self,
        tokenizer,
        system_prompt: str,
        enable_thinking: bool,
        sink_prefix_tokens: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.sink_prefix_tokens = max(0, int(sink_prefix_tokens))

    def _render_base(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _tokenize_text(self, text: str, offsets: bool = False):
        kwargs = dict(add_special_tokens=False, return_tensors=None)
        if offsets:
            kwargs["return_offsets_mapping"] = True
        return self.tokenizer(text, **kwargs)

    @staticmethod
    def _find_overlapping_tokens(
        offsets: Iterable[tuple[int, int]], char_start: int, char_end: int
    ) -> tuple[int, int]:
        indices: list[int] = []
        for i, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if start < char_end and end > char_start:
                indices.append(i)
        if not indices:
            raise ValueError("Could not map the original question to token offsets")
        return min(indices), max(indices) + 1

    def _question_span(
        self, full_text: str, question: str, input_ids: list[int], base_char_end: int
    ) -> tuple[int, int]:
        char_start = full_text.rfind(question, 0, base_char_end)
        if char_start < 0:
            raise ValueError(
                "The rendered chat template does not contain the original question verbatim. "
                "Use a template that preserves user content or customize PromptBuilder."
            )
        char_end = char_start + len(question)

        if getattr(self.tokenizer, "is_fast", False):
            enc = self._tokenize_text(full_text, offsets=True)
            offsets = enc["offset_mapping"]
            return self._find_overlapping_tokens(offsets, char_start, char_end)

        # Slow-tokenizer fallback: locate the question token sequence in the full prompt.
        q_ids = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        if not q_ids:
            raise ValueError("Question tokenization is empty")
        matches: list[int] = []
        qn = len(q_ids)
        for i in range(0, len(input_ids) - qn + 1):
            if input_ids[i : i + qn] == q_ids:
                matches.append(i)
        if not matches:
            # Boundary tokenization can alter the first/last token. Try the interior sequence.
            if len(q_ids) > 2:
                core = q_ids[1:-1]
                for i in range(0, len(input_ids) - len(core) + 1):
                    if input_ids[i : i + len(core)] == core:
                        return max(0, i - 1), min(len(input_ids), i + len(core) + 1)
            raise ValueError("Could not locate question tokens with a slow tokenizer")
        start = matches[0]
        return start, start + qn

    def build(
        self,
        question: str,
        shared_text: str = "",
        control_text: str = "",
        device: str | torch.device | None = None,
    ) -> PromptEncoding:
        base = self._render_base(question)
        full_text = base + shared_text + control_text
        encoded = self._tokenize_text(full_text, offsets=False)
        input_ids_list = list(encoded["input_ids"])
        if not input_ids_list:
            raise ValueError("Prompt tokenization produced no tokens")

        question_start, question_end = self._question_span(
            full_text, question, input_ids_list, base_char_end=len(base)
        )

        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        sink: set[int] = {
            i for i, token_id in enumerate(input_ids_list) if token_id in special_ids
        }
        sink.update(range(min(self.sink_prefix_tokens, len(input_ids_list))))
        # Never remove question tokens from the denominator.
        sink.difference_update(range(question_start, question_end))

        base_tokens = len(self._tokenize_text(base, offsets=False)["input_ids"])
        base_plus_shared = len(
            self._tokenize_text(base + shared_text, offsets=False)["input_ids"]
        )
        shared_tokens = max(0, base_plus_shared - base_tokens)
        control_tokens = max(0, len(input_ids_list) - base_plus_shared)

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        return PromptEncoding(
            input_ids=input_ids,
            rendered_text=full_text,
            question_start=question_start,
            question_end=question_end,
            sink_positions=tuple(sorted(sink)),
            base_prompt_tokens=base_tokens,
            shared_text_tokens=shared_tokens,
            control_tokens=control_tokens,
        )
