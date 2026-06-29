from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
PreTrainedTokenizerFast = transformers.PreTrainedTokenizerFast
Qwen3Config = getattr(transformers, "Qwen3Config", None)
Qwen3ForCausalLM = getattr(transformers, "Qwen3ForCausalLM", None)
if Qwen3Config is None or Qwen3ForCausalLM is None:
    pytest.skip("Qwen3 classes require the pinned Transformers version", allow_module_level=True)

from dcrh.config import CostConfig, GenerationConfig, ModelConfig, PromptConfig, SignalConfig
from dcrh.core.costs import CostLedger
from dcrh.runtime.transformers.model_runner import LocalQwen3Runner


def test_local_only_runner_one_step(tmp_path: Path):
    vocab = {
        "[UNK]": 0,
        "<bos>": 1,
        "<eos>": 2,
        "system": 3,
        "user": 4,
        "assistant": 5,
        "You": 6,
        "are": 7,
        "helpful": 8,
        "What": 9,
        "is": 10,
        "1": 11,
        "+": 12,
        "?": 13,
        "2": 14,
        "x": 15,
        "</think>": 16,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}assistant "
    )
    tokenizer.save_pretrained(tmp_path)

    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        eos_token_id=2,
        bos_token_id=1,
    )
    model = Qwen3ForCausalLM(config)
    model.generation_config.eos_token_id = 2
    model.save_pretrained(tmp_path)

    runner = LocalQwen3Runner(
        role="slm",
        model_cfg=ModelConfig(path=str(tmp_path), device="cpu", dtype="float32"),
        prompt_cfg=PromptConfig(system_prompt="You are helpful", enable_thinking=False),
        generation_cfg=GenerationConfig(do_sample=False, max_initial_slm_tokens=2),
        signal_cfg=SignalConfig(),
        cost_cfg=CostConfig(synchronize_cuda_for_timing=False),
    )
    ledger = CostLedger()
    session = runner.create_session(
        question="What is 1 + 1 ?",
        shared_text="",
        control_text="",
        purpose="test",
        ledger=ledger,
    )
    step = session.step()
    assert step.observation.token_id >= 0
    assert isinstance(step.observation.text_piece, str)
    assert ledger.summary()["counters"]["prefill_calls"] == 1
    session.close()

    runner.thinking_end_marker = "</think>"
    marker_ledger = CostLedger()
    marker_session = runner.create_session(
        question="What is 1 + 1 ?",
        shared_text="",
        control_text="",
        purpose="marker-test",
        ledger=marker_ledger,
    )
    forced = torch.full((len(vocab),), -1000.0)
    forced[vocab["</think>"]] = 1000.0
    marker_session._next_logits = forced
    marker_step = marker_session.step()
    assert marker_step.ended
    assert marker_step.end_reason == "thinking_end"
    marker_session.close()
