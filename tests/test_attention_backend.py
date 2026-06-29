import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
Qwen3Config = getattr(transformers, "Qwen3Config", None)
Qwen3ForCausalLM = getattr(transformers, "Qwen3ForCausalLM", None)
if Qwen3Config is None or Qwen3ForCausalLM is None:
    pytest.skip("Qwen3 classes require the pinned Transformers version", allow_module_level=True)

from dcrh.runtime.transformers.attention_backend import (
    BACKEND_NAME,
    RouteCollector,
    register_probe_attention_backend,
    select_mdrv_route_layer,
)
from dcrh.core.router_signals import compute_attention_route


def test_select_mdrv_route_layer_safeguards():
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
    config.layer_types = [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert select_mdrv_route_layer(config) == 3
    config.layer_types = []
    assert select_mdrv_route_layer(config) is None

    config.layer_types = None
    config.use_sliding_window = False
    config.sliding_window = None
    assert select_mdrv_route_layer(config) == 3

    config.use_sliding_window = True
    config.sliding_window = 16
    config.max_window_layers = 2
    assert select_mdrv_route_layer(config) == 1

    config.max_window_layers = 0
    assert select_mdrv_route_layer(config) is None


def test_route_probe_matches_eager_last_row_distribution():
    register_probe_attention_backend()
    torch.manual_seed(11)
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
    spans = {"A": (0, 3), "O": (3, 5), "P": (5, 8), "C": (8, 10)}
    route_layer = 3

    model.set_attn_implementation("eager")
    with torch.inference_mode():
        eager = model(
            prefix,
            use_cache=False,
            logits_to_keep=1,
            output_attentions=True,
        )
    expected = compute_attention_route(eager.attentions[route_layer], spans)

    model.set_attn_implementation(BACKEND_NAME)
    collector = RouteCollector(
        route_layer=route_layer,
        region_spans=spans,
        head_chunk_size=2,
    )
    with torch.inference_mode():
        collector.begin_boundary()
        model(
            prefix,
            use_cache=False,
            logits_to_keep=1,
            dcrh_route_collector=collector,
        )
        observed = collector.end_boundary().route_distribution

    for name in ("A", "O", "P", "C"):
        assert abs(observed[name] - expected[name]) < 1e-5
