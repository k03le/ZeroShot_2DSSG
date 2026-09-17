"""
VLM wrapper for Qwen2.5-VL-7B-Instruct.

Single responsibility: (image + prompt) -> text response.
No parsing, no visualization — those are handled by other modules.

Design decisions:
  - Lazy loading:  Model (~16 GB VRAM) is only loaded on the first call.
  - Singleton:     Multiple imports still load the model only once.
  - dataclass:     All hyperparameters in one central VLMConfig.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


# ─── Configuration (all hyperparameters in one place) ─────────────
@dataclass(frozen=True)
class VLMConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "auto"           # uses bf16 on RTX 3090
    device_map: str = "auto"            # automatically places layers on GPU
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.2     # > 1.0 penalizes repeated tokens


# ─── VLM Wrapper ──────────────────────────────────────────────────
class VLM:
    """
    Singleton wrapper around Qwen2.5-VL.

    Usage:
        from vlm import VLM
        response = VLM.ask(image, "Describe this scene.")
    """

    _model: ClassVar[Qwen2_5_VLForConditionalGeneration | None] = None
    _processor: ClassVar[AutoProcessor | None] = None
    _config: ClassVar[VLMConfig] = VLMConfig()

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Loads model and processor exactly once into GPU memory."""
        if cls._model is not None:
            return

        print(f"Loading {cls._config.model_id} ...")
        cls._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cls._config.model_id,
            torch_dtype=cls._config.torch_dtype,
            device_map=cls._config.device_map,
        )
        cls._processor = AutoProcessor.from_pretrained(cls._config.model_id)
        print("Model ready.")

    @classmethod
    def ask(cls, image: Image.Image, prompt: str) -> str:
        """
        Sends (image, prompt) to the model and returns the text response.

        Four steps of inference:
            1. Apply chat template  -> structured prompt string
            2. Tokenize and encode  -> tensor inputs on the GPU
            3. Autoregressive generation -> token IDs (input + new)
            4. Decode ONLY the new tokens -> pure model response
        """
        cls._ensure_loaded()

        # After _ensure_loaded, both are guaranteed to be non-None.
        model = cls._model
        processor = cls._processor
        assert model is not None and processor is not None

        # 1. Qwen expects role-based messages with mixed content types
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ],
        }]

        # 2. Build prompt string and tensor inputs
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # 3. Generate tokens (input tokens + newly generated tokens)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cls._config.max_new_tokens,
            repetition_penalty=cls._config.repetition_penalty,
        )

        # 4. Strip input tokens, decode only the model's answer
        new_token_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(
            new_token_ids, skip_special_tokens=True,
        )[0]