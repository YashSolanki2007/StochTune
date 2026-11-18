import pytest
import torch

from stochastic_llm.train import GracefulStopCallback
from stochastic_llm.training_data import (
    CompletionCollator,
    TargetTrie,
    encode_example,
    soft_kl_loss,
)


def test_padding_does_not_mask_real_eos_when_pad_equals_eos():
    batch = CompletionCollator(9)(
        [
            {"input_ids": [1, 2, 9], "labels": [-100, 2, 9]},
            {"input_ids": [1, 9], "labels": [-100, 9]},
        ]
    )
    assert batch["labels"].tolist() == [[-100, 2, 9], [-100, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_sequence_kl_matches_chain_rule_and_penalizes_invalid_mass():
    # Target: P(1, EOS)=.25, P(2, EOS)=.75. The model assigns .1 invalid mass.
    trie = TargetTrie([[1, 3], [2, 3]], [0.25, 0.75])
    examples = []
    for path in trie.paths:
        examples.append(
            {
                "input_ids": [0] + path,
                "labels": [-100] + path,
                "soft_targets": [(i, ids, p) for i, (ids, p) in enumerate(trie.targets(path))],
            }
        )
    losses = []
    for example in examples:
        batch = CompletionCollator(0)([example])
        logits = (
            torch.tensor(
                [[[0.1, 0.3, 0.6, 1e-20], [1e-20, 1e-20, 1e-20, 1.0], [0.25, 0.25, 0.25, 0.25]]]
            )
            .log()
            .requires_grad_()
        )
        loss = soft_kl_loss(logits, batch["soft_targets"])
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        losses.append(loss.item())
    expected = 0.25 * torch.log(torch.tensor(0.25 / 0.3)) + 0.75 * torch.log(
        torch.tensor(0.75 / 0.6)
    )
    assert 0.25 * losses[0] + 0.75 * losses[1] == pytest.approx(expected.item(), abs=1e-6)
    assert expected > 0


def test_overlong_training_examples_are_rejected():
    class Tokenizer:
        eos_token_id = 9

        def encode(self, text, **kwargs):
            return list(range(len(text)))

    with pytest.raises(ValueError, match="refusing to truncate"):
        encode_example(Tokenizer(), "prompt", "answer", max_length=5)


def test_time_limit_signal_requests_checkpoint_and_stop():
    from transformers import TrainerControl

    callback = GracefulStopCallback()
    callback.request_stop(None, None)
    control = callback.on_step_end(None, None, TrainerControl())
    assert control.should_save
    assert control.should_training_stop
    assert not control.should_evaluate


def test_soft_kl_handles_underflowed_tail_probabilities():
    logits = torch.zeros((1, 1, 3), requires_grad=True)
    targets = {
        "batch": torch.tensor([0, 0]),
        "position": torch.tensor([0, 0]),
        "token": torch.tensor([0, 1]),
        "probability": torch.tensor([1.0, 0.0]),
    }
    loss = soft_kl_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
