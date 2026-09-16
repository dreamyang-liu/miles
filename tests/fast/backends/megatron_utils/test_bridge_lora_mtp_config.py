from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("requested,expected", [(None, 1), (0, 0), (2, 2)])
def test_bridge_lora_honors_explicit_mtp_zero(monkeypatch, requested, expected):
    import megatron.bridge
    import megatron.bridge.training.config as bridge_config

    from miles.backends.megatron_utils import bridge_lora_helpers, lora_utils

    provider = MagicMock(mtp_num_layers=1)
    bridge = SimpleNamespace(to_megatron_provider=lambda **_: provider)
    monkeypatch.setattr(megatron.bridge.AutoBridge, "from_hf_pretrained", lambda *_, **__: bridge)
    monkeypatch.setattr(bridge_config, "DistributedDataParallelConfig", MagicMock())
    monkeypatch.setattr(
        bridge_lora_helpers,
        "load_hf_config",
        lambda _: SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
    )
    monkeypatch.setattr(lora_utils, "create_lora_instance", lambda _: MagicMock())
    args = Namespace(
        hf_checkpoint="fixture",
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        gradient_accumulation_fusion=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=["core_attn"],
        distribute_saved_activations=False,
        attention_backend="flash",
        mtp_num_layers=requested,
        optimizer="adam",
        offload_train=False,
        multi_lora=False,
        make_vocab_size_divisible_by=64,
        accumulate_allreduce_grads_in_fp32=True,
    )
    bridge_lora_helpers._setup_lora_model_via_bridge(args)
    assert provider.mtp_num_layers == expected
    assert provider.make_vocab_size_divisible_by == 64
    provider.finalize.assert_called_once()
