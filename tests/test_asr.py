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
