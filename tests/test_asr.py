from __future__ import annotations

from transkriptor.asr import VibeVoiceASRBackend


def test_context_info_combines_context_and_hotwords() -> None:
    context_info = VibeVoiceASRBackend._context_info(
        hotwords=["Karl Lauterbach", "SPD"],
        context="Moderation: Markus Lanz.",
    )

    assert context_info == (
        "Moderation: Markus Lanz.\n\n"
        "Hotwords: Karl Lauterbach, SPD"
    )


def test_context_info_returns_none_for_empty_input() -> None:
    assert VibeVoiceASRBackend._context_info(hotwords=[], context=" ") is None


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_quantization_config_none() -> None:
    backend = VibeVoiceASRBackend(quantization="none")

    assert backend._quantization_config(FakeBitsAndBytesConfig, compute_dtype="bf16") is None


def test_quantization_config_8bit() -> None:
    backend = VibeVoiceASRBackend(quantization="8bit")

    config = backend._quantization_config(FakeBitsAndBytesConfig, compute_dtype="bf16")

    assert config.kwargs == {"load_in_8bit": True}


def test_quantization_config_4bit_alias_uses_nf4() -> None:
    backend = VibeVoiceASRBackend(quantization="4bit")

    config = backend._quantization_config(FakeBitsAndBytesConfig, compute_dtype="bf16")

    assert config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bf16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }


def test_quantization_config_4bit_fp4() -> None:
    backend = VibeVoiceASRBackend(quantization="4bit-fp4")

    config = backend._quantization_config(FakeBitsAndBytesConfig, compute_dtype="bf16")

    assert config.kwargs["bnb_4bit_quant_type"] == "fp4"
