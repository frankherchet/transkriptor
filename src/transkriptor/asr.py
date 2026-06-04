from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL_PATH = "microsoft/VibeVoice-ASR"
DEFAULT_QUANTIZATION = "none"
SUPPORTED_QUANTIZATIONS = ("none", "8bit", "4bit", "4bit-nf4", "4bit-fp4")


class ASRBackend(Protocol):
    model_path: str

    def transcribe(
        self,
        media_path: Path,
        *,
        hotwords: list[str],
        context: str | None,
    ) -> Any:
        pass


@dataclass
class VibeVoiceASRBackend:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    quantization: str = DEFAULT_QUANTIZATION
    max_new_tokens: int = 32768
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    repetition_penalty: float = 1.0
    attn_implementation: str = "sdpa"

    _processor: Any = None
    _model: Any = None
    _torch: Any = None

    def transcribe(
        self,
        media_path: Path,
        *,
        hotwords: list[str],
        context: str | None,
    ) -> dict[str, Any]:
        self._load()

        inputs = self._processor(
            audio=str(media_path),
            sampling_rate=None,
            context_info=self._context_info(hotwords=hotwords, context=context),
            return_tensors="pt",
            add_generation_prompt=True,
        )
        device = self._resolved_device()
        inputs = {
            key: value.to(device) if isinstance(value, self._torch.Tensor) else value
            for key, value in inputs.items()
        }

        do_sample = self.temperature > 0
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if do_sample else None,
            "top_p": self.top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self._processor.pad_id,
            "eos_token_id": self._processor.tokenizer.eos_token_id,
        }
        generation_config = {
            key: value for key, value in generation_config.items() if value is not None
        }

        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_config)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = self._processor.decode(generated_ids, skip_special_tokens=True)
        segments = self._processor.post_process_transcription(generated_text)
        return {"raw_text": generated_text, "segments": segments}

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_asr import (
                VibeVoiceASRForConditionalGeneration,
            )
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "VibeVoice-ASR dependencies are not installed. "
                "Install with `uv sync --extra asr`."
            ) from exc

        self._torch = torch
        dtype = self._dtype(torch)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": True,
        }
        quantization_config = self._quantization_config(
            BitsAndBytesConfig,
            compute_dtype=dtype,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto" if self.device == "auto" else {"": self.device}
        elif self.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self.device != "mps":
            model_kwargs["device_map"] = self.device

        self._processor = VibeVoiceASRProcessor.from_pretrained(self.model_path)
        self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if self.device != "auto" and quantization_config is None:
            self._model.to(self.device)
        self._model.eval()

    def _quantization_config(
        self,
        bitsandbytes_config: Any,
        *,
        compute_dtype: Any,
    ) -> Any | None:
        quantization = self.quantization.lower()
        if quantization not in SUPPORTED_QUANTIZATIONS:
            supported = ", ".join(SUPPORTED_QUANTIZATIONS)
            raise ValueError(
                f"unsupported quantization: {self.quantization!r}. "
                f"Supported values: {supported}"
            )
        if quantization == "none":
            return None
        if quantization == "8bit":
            return bitsandbytes_config(load_in_8bit=True)

        quant_type = "nf4" if quantization in {"4bit", "4bit-nf4"} else "fp4"
        return bitsandbytes_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=True,
        )

    def _dtype(self, torch: Any) -> Any:
        if self.device in {"cpu", "mps", "xpu"}:
            return torch.float32
        return torch.bfloat16

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if self._model is None:
            return "cpu"
        try:
            return str(next(self._model.parameters()).device)
        except StopIteration:
            return "cpu"

    @staticmethod
    def _context_info(*, hotwords: list[str], context: str | None) -> str | None:
        parts: list[str] = []
        if context is not None and context.strip():
            parts.append(context.strip())

        cleaned = [word.strip() for word in hotwords if word.strip()]
        if cleaned:
            parts.append("Hotwords: " + ", ".join(cleaned))

        return "\n\n".join(parts) if parts else None
