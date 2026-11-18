import pytest

from stochastic_llm.train import GracefulStopCallback
from stochastic_llm.training_data import CompletionCollator, encode_example


def test_padding_does_not_mask_real_eos_when_pad_equals_eos():
    batch = CompletionCollator(9)(
        [
            {"input_ids": [1, 2, 9], "labels": [-100, 2, 9]},
            {"input_ids": [1, 9], "labels": [-100, 9]},
        ]
    )
    assert batch["labels"].tolist() == [[-100, 2, 9], [-100, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


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
