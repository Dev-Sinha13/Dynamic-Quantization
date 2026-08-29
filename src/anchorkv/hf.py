"""Optional Hugging Face integration used by the Colab experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .artifacts import AttentionTrace, TraceMetadata, save_trace
from .segmentation import token_spans_from_offsets
from .types import TokenSpan


@dataclass(frozen=True, slots=True)
class HFTraceConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "main"
    max_sequence_length: int = 1024
    max_new_tokens: int = 192
    seed: int = 7
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if self.max_sequence_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("sequence limits must be positive")


@dataclass(frozen=True, slots=True)
class HFTraceResult:
    artifact_path: Path
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    sequence_length: int
    peak_gpu_bytes: int


def extract_hf_trace(
    prompt: str,
    *,
    sample_id: str,
    output_path: str | Path,
    config: HFTraceConfig = HFTraceConfig(),
) -> HFTraceResult:
    """Generate a reasoning trace and replay it once with eager attention.

    PyTorch and Transformers are imported lazily so the model-agnostic package
    and its unit tests remain lightweight. This function intentionally supports
    batch size one and rejects traces longer than the configured bound.
    """

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not sample_id:
        raise ValueError("sample_id is required")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face extraction requires `pip install -e .[research]`"
        ) from error

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    rendered_prompt = _render_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    allowed_new_tokens = min(
        config.max_new_tokens,
        config.max_sequence_length - prompt_tokens,
    )
    if allowed_new_tokens <= 0:
        raise ValueError(
            f"prompt uses {prompt_tokens} tokens and leaves no room under the "
            f"{config.max_sequence_length}-token trace limit"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(config.device)
    model.eval()
    device_inputs = {key: value.to(config.device) for key, value in inputs.items()}

    if config.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        generated_ids = model.generate(
            **device_inputs,
            max_new_tokens=allowed_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids = generated_ids[:, : config.max_sequence_length]
    generated_token_ids = full_ids[0, prompt_tokens:].tolist()
    generated_text, generated_offsets = decoded_token_offsets(tokenizer, generated_token_ids)
    generated_spans = token_spans_from_offsets(generated_text, generated_offsets)
    spans = tuple(
        TokenSpan(
            start=span.start + prompt_tokens,
            end=span.end + prompt_tokens,
            text=span.text,
        )
        for span in generated_spans
    )
    if not spans:
        raise RuntimeError("the generated trace contained no sentence-like reasoning steps")

    if not hasattr(model, "set_attn_implementation"):
        raise RuntimeError(
            "the installed Transformers version cannot switch attention backends; "
            "install a current release"
        )
    model.set_attn_implementation("eager")

    with torch.inference_mode():
        replay = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    if replay.attentions is None:
        raise RuntimeError("the model did not return attention tensors in eager mode")
    vertical_scores = reduce_attention_layers(replay.attentions, spans)

    model_config = model.config
    layers = int(model_config.num_hidden_layers)
    query_heads = int(model_config.num_attention_heads)
    kv_heads = int(getattr(model_config, "num_key_value_heads", query_heads))
    head_dim = int(getattr(model_config, "head_dim", model_config.hidden_size // query_heads))
    if vertical_scores.shape[:2] != (layers, query_heads):
        raise RuntimeError(
            "returned attention dimensions do not match the model configuration: "
            f"{vertical_scores.shape[:2]} versus {(layers, query_heads)}"
        )

    metadata = TraceMetadata.create(
        model_id=config.model_id,
        sample_id=sample_id,
        prompt=prompt,
        seed=config.seed,
        dtype="float16",
        sequence_length=int(full_ids.shape[-1]),
        layers=layers,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        model_revision=config.model_revision,
    )
    artifact = AttentionTrace(metadata, spans, vertical_scores)
    artifact_path = save_trace(artifact, output_path)
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated()) if config.device == "cuda" else 0
    )
    return HFTraceResult(
        artifact_path=artifact_path,
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated_token_ids),
        sequence_length=int(full_ids.shape[-1]),
        peak_gpu_bytes=peak_gpu_bytes,
    )


def decoded_token_offsets(tokenizer: Any, token_ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decode tokens individually and return offsets into the joined text."""

    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(piece)
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def reduce_attention_layers(
    attention_layers: Sequence[Any],
    spans: Sequence[TokenSpan],
) -> NDArray[np.float32]:
    """Reduce one layer at a time to avoid an additional full-attention stack."""

    reduced_layers: list[NDArray[np.float32]] = []
    expected_heads: int | None = None
    for layer_attention in attention_layers:
        layer = layer_attention[0]
        if hasattr(layer, "detach"):
            layer = layer.detach().float().cpu().numpy()
        values = np.asarray(layer, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != values.shape[-2]:
            raise ValueError("each attention layer must have shape [heads, tokens, tokens]")
        if expected_heads is None:
            expected_heads = values.shape[0]
        elif values.shape[0] != expected_heads:
            raise ValueError("attention layers must have the same number of heads")

        sentence_scores = np.zeros((values.shape[0], len(spans)), dtype=np.float32)
        for sentence_index, span in enumerate(spans):
            if span.end > values.shape[-1]:
                raise ValueError("sentence span exceeds the attention sequence length")
            if span.end == values.shape[-1]:
                continue
            future = values[:, span.end :, span.start : span.end]
            sentence_scores[:, sentence_index] = (
                future.sum(axis=-1).mean(axis=-1) / span.length
            )
        reduced_layers.append(sentence_scores)
    if not reduced_layers:
        raise ValueError("at least one attention layer is required")
    return np.stack(reduced_layers, axis=0)


def _render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

