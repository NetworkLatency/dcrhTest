import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from dcrh.runtime.transformers.attention_backend import (
    BACKEND_NAME,
    GroundingCollector,
    register_probe_attention_backend,
)


def test_selective_online_grounding_probe():
    register_probe_attention_backend()
    config = Qwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(config).eval()
    model.set_attn_implementation(BACKEND_NAME)
    prefix = torch.randint(0, 100, (1, 10))
    with torch.inference_mode():
        first = model(prefix, use_cache=True, logits_to_keep=1)
        collector = GroundingCollector(
            selected_layers=frozenset({1, 2}),
            question_start=2,
            question_end=5,
            sink_positions=(0,),
            head_chunk_size=2,
        )
        collector.begin_token()
        model(
            torch.randint(0, 100, (1, 1)),
            past_key_values=first.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            dcrh_grounding_collector=collector,
        )
        summary = collector.end_token()
    assert summary.observed_layers == 2
    assert summary.observed_heads == 8
    assert torch.isfinite(torch.tensor(summary.grounding))


def test_probe_matches_eager_attention_aggregation():
    register_probe_attention_backend()
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(config).eval()
    prefix = torch.randint(0, 100, (1, 10))
    token = torch.randint(0, 100, (1, 1))
    selected = (0,)
    q_start, q_end = 2, 5
    sink = 0

    model.set_attn_implementation("eager")
    with torch.inference_mode():
        eager_prefill = model(prefix, use_cache=True, logits_to_keep=1)
        eager_decode = model(
            token,
            past_key_values=eager_prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            output_attentions=True,
        )
    key_length = eager_decode.attentions[0].shape[-1]
    base_rate = (q_end - q_start) / (key_length - 1)
    base_logit = torch.logit(torch.tensor(base_rate, dtype=torch.float64))
    eager_values = []
    for layer in selected:
        weights = eager_decode.attentions[layer].float()
        question_mass = weights[..., q_start:q_end].sum(dim=-1)
        valid_mass = (1.0 - weights[..., sink : sink + 1].sum(dim=-1)).clamp_min(1e-6)
        relative = (question_mass / valid_mass).clamp(1e-6, 1.0 - 1e-6)
        eager_values.append(torch.logit(relative) - base_logit)
    eager_grounding = torch.cat([x.reshape(-1) for x in eager_values]).mean().item()

    model.set_attn_implementation(BACKEND_NAME)
    with torch.inference_mode():
        probe_prefill = model(prefix, use_cache=True, logits_to_keep=1)
        collector = GroundingCollector(
            selected_layers=frozenset(selected),
            question_start=q_start,
            question_end=q_end,
            sink_positions=(sink,),
            head_chunk_size=2,
        )
        collector.begin_token()
        model(
            token,
            past_key_values=probe_prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            dcrh_grounding_collector=collector,
        )
        probe_grounding = collector.end_token().grounding

    assert abs(probe_grounding - eager_grounding) < 1e-5
