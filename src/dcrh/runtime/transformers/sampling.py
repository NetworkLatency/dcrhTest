from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SamplingParameters:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float


class TokenSampler:
    """Applies generation penalties and truncation after boundary signals use raw logits."""

    def __init__(
        self,
        vocab_size: int,
        device: torch.device,
        parameters: SamplingParameters,
        seed: int,
    ) -> None:
        self.parameters = parameters
        self.seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))

    def _apply_repetition_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        penalty = float(self.parameters.repetition_penalty)
        if penalty == 1.0 or not bool(self.seen.any()):
            return logits
        values = logits[self.seen]
        values = torch.where(values < 0, values * penalty, values / penalty)
        logits[self.seen] = values
        return logits

    def _apply_presence_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        penalty = float(self.parameters.presence_penalty)
        if penalty != 0.0 and bool(self.seen.any()):
            logits[self.seen] -= penalty
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.numel():
            return logits
        threshold = torch.topk(logits, k=top_k).values[-1]
        return logits.masked_fill(logits < threshold, -torch.inf)

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        if top_p <= 0.0:
            raise ValueError("top_p must be positive")
        finite_indices = torch.nonzero(torch.isfinite(logits), as_tuple=False).flatten()
        finite_logits = logits.index_select(0, finite_indices)
        sorted_logits, order = torch.sort(finite_logits, descending=True)
        sorted_indices = finite_indices.index_select(0, order)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > top_p
        if remove.numel() > 1:
            remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(logits, -torch.inf)
        filtered.scatter_(0, sorted_indices, sorted_logits)
        return filtered

    def sample(self, raw_logits: torch.Tensor) -> int:
        if raw_logits.ndim != 1:
            raise ValueError(f"Expected 1D logits, got {tuple(raw_logits.shape)}")
        logits = raw_logits.float().clone()
        logits = self._apply_presence_penalty(logits)
        logits = self._apply_repetition_penalty(logits)

        if not self.parameters.do_sample:
            token_id = int(torch.argmax(logits).item())
            self.seen[token_id] = True
            return token_id

        temperature = float(self.parameters.temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0 when do_sample=True")
        logits /= temperature
        logits = self._apply_top_k(logits, int(self.parameters.top_k))
        logits = self._apply_top_p(logits, float(self.parameters.top_p))
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0.0:
            raise RuntimeError("Sampling distribution became invalid")
        token_id = int(
            torch.multinomial(probabilities, num_samples=1, generator=self.generator).item()
        )
        self.seen[token_id] = True
        return token_id
