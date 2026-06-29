import math

import pytest

torch = pytest.importorskip("torch")

from dcrh.core.router_signals import (
    build_region_spans_from_token_ids,
    build_region_spans,
    compute_attention_route,
    compute_jsd_route_velocity,
    compute_tpm_from_logits,
    find_first_content_token,
    token_is_content_token,
)


class TinyTokenizer:
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(char) for char in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        pieces = []
        for token_id in ids:
            if token_id == 0:
                pieces.append("<eos>")
            else:
                pieces.append(chr(int(token_id)))
        return "".join(pieces)


def test_tpm_margin_uses_full_softmax_top2_gap():
    logits = torch.tensor([0.0, 2.0, 1.0, -1.0])
    tpm = compute_tpm_from_logits(logits)
    probs = torch.softmax(logits, dim=-1)
    assert tpm.top1_token_id == 1
    assert tpm.top2_token_id == 2
    assert math.isclose(tpm.tpm_margin, float((probs[1] - probs[2]).item()))


def test_region_spans_keep_empty_o_and_p_for_first_steps():
    tokenizer = TinyTokenizer()
    first = build_region_spans("qq", ["a\n\n"], tokenizer)
    assert first.A == (0, 2)
    assert first.O == (2, 2)
    assert first.P == (2, 2)
    assert first.C == (2, 5)
    assert first.input_ids == tuple(ord(c) for c in "qqa\n\n")

    second = build_region_spans("qq", ["a\n\n", "b\n\n"], tokenizer)
    assert second.O == (2, 2)
    assert second.P == (2, 5)
    assert second.C == (5, 8)


def test_region_spans_from_token_ids_can_include_skipped_action_tokens():
    spans = build_region_spans_from_token_ids(
        prompt_ids=[1, 2],
        chunk_ids=[[3, 4], [5, 6]],
        input_ids=[1, 2, 3, 4, 5, 6, 7],
    )
    assert spans.A == (0, 2)
    assert spans.P == (2, 4)
    assert spans.C == (4, 6)
    assert spans.input_ids == (1, 2, 3, 4, 5, 6, 7)


def test_attention_route_is_length_normalized_density():
    spans = {"A": (0, 2), "O": (2, 2), "P": (2, 4), "C": (4, 5)}
    row = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.2])
    route = compute_attention_route(row, spans)
    # Densities: A=.3, O=0, P=.1, C=.2 => normalized by .6.
    assert math.isclose(route["A"], 0.5)
    assert route["O"] == 0.0
    assert math.isclose(route["P"], 1.0 / 6.0)
    assert math.isclose(route["C"], 1.0 / 3.0)
    assert math.isclose(sum(route.values()), 1.0)


def test_attention_route_can_exclude_query_self_position():
    spans = {"A": (0, 1), "O": (1, 1), "P": (1, 2), "C": (2, 4)}
    row = torch.tensor([0.2, 0.2, 0.2, 0.4])
    route = compute_attention_route(row, spans, exclude_positions=(3,))
    assert math.isclose(route["A"], 1.0 / 3.0)
    assert math.isclose(route["P"], 1.0 / 3.0)
    assert math.isclose(route["C"], 1.0 / 3.0)


def test_jsd_route_velocity_zero_for_identical_routes():
    route = {"A": 0.2, "O": 0.3, "P": 0.1, "C": 0.4}
    assert compute_jsd_route_velocity(route, route) == 0.0


def test_first_content_token_skips_whitespace_special_and_punctuation():
    tokenizer = TinyTokenizer()
    assert not token_is_content_token(ord(" "), " ", tokenizer)
    assert not token_is_content_token(0, "<eos>", tokenizer)
    assert not token_is_content_token(ord(","), ",", tokenizer)
    assert token_is_content_token(ord("A"), " A", tokenizer)

    token_ids = [ord(" "), 0, ord(","), ord("A")]
    assert find_first_content_token(token_ids, tokenizer) == (3, ord("A"), "A")
