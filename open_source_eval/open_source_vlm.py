"""Open-source VLM client for the local vLLM OpenAI-compatible server."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


API_KEY = "dummy"
IMAGE_DETAIL = "auto"
MODEL_IDS = {
    "gemma_4_12B_it": {
        "provider": "responses",
        "base_url": "http://172.16.68.130:8001/v1",
        "model": "gemma-4-12b-it",
    },
    "qwen3_vl_8B_instruct": {
        "provider": "responses",
        "base_url": "http://172.16.68.130:8001/v1",
        "model": "qwen3-vl-8b-instruct",
    },
    "gemma_4_26B_A4B_it": {
        "provider": "responses",
        "base_url": "http://172.16.68.130:8002/v1",
        "model": "gemma-4-26b-a4b-it",
    },
    "qwen3_6_27B": {
        "provider": "responses",
        "base_url": "http://172.16.68.130:8002/v1",
        "model": "qwen3.6-27b",
    },
    "glm_4_6v_flash": {
        "provider": "responses",
        "base_url": "http://172.16.68.130:8002/v1",
        "model": "glm-4.6v-flash",
    },
    "gemini_3_1_pro_preview": {
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
    },
    "anthropic_claude_sonnet_5": {
        "provider": "responses",
        "base_url": "https://a6api.com/v1",
        "model": "claude-sonnet-5",
        "api_key_env": "A6API_KEY",
    },
    "anthropic_claude_opus_4_8": {
        "provider": "responses",
        "base_url": "https://a6api.com/v1",
        "model": "claude-opus-4-8",
        "api_key_env": "A6API_KEY",
    },
    "kimi_k2_6": {
        "provider": "chat_completions",
        "base_url": "https://a6api.com/v1",
        "model": "kimi-k2.6",
        "api_key_env": "A6API_KEY",
    },
    "qwen3_vl_235b_a22b": {
        "provider": "bedrock_converse",
        "region": "us-east-1",
        "model": "qwen.qwen3-vl-235b-a22b",
    },
    "qwen3_6_plus_cun_ai": {
        "provider": "curl_chat_completions",
        "base_url": "https://www.cun.ai/v1",
        "model": "qwen3.6-plus",
        "api_key_env": "CUN_AI_KEY",
    },
}

ImageInput = str | Path | Image.Image


def _image_to_data_url(image: ImageInput) -> str:
    if isinstance(image, str | Path):
        path = Path(image)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image_format = image.format or "PNG"
        image.save(buffer, format=image_format)
        mime_type = Image.MIME.get(image_format.upper(), "image/png")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    raise TypeError("image must be a path or PIL.Image")


def _ensure_images(images: ImageInput | Sequence[ImageInput] | None) -> list[ImageInput]:
    if images is None:
        return []
    if isinstance(images, str | Path | Image.Image):
        return [images]
    return list(images)


def _post_responses(base_url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_chat_completions(base_url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_curl_chat_completions(base_url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    result = subprocess.run(
        [
            "curl",
            "-sS",
            f"{base_url}/chat/completions",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    response = json.loads(result.stdout)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def _post_gemini(payload: dict[str, Any], model: str, timeout: float) -> dict[str, Any]:
    api_key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_bedrock_converse(payload: dict[str, Any], region: str, model: str, timeout: float) -> dict[str, Any]:
    token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"empty VLM response text: {response}")
    return text


def _extract_chat_completions_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in response.get("choices", []):
        text = choice.get("message", {}).get("content")
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"empty chat completions response text: {response}")
    return text


def _extract_gemini_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"empty Gemini response text: {response}")
    return text


def _extract_bedrock_converse_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for content in response.get("output", {}).get("message", {}).get("content", []):
        text = content.get("text")
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if not text:
        raise RuntimeError(f"empty Bedrock Converse response text: {response}")
    return text


def model_config_for_id(model_id: str) -> dict[str, str]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model_id {model_id!r}; expected one of {sorted(MODEL_IDS)}")
    return MODEL_IDS[model_id]


def codex(
    prompt: str,
    images: ImageInput | Sequence[ImageInput] | None = None,
    *,
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    timeout: float = 300,
) -> str:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image in _ensure_images(images):
        content.append(
            {
                "type": "input_image",
                "detail": IMAGE_DETAIL,
                "image_url": _image_to_data_url(image),
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    api_key = os.environ[api_key_env] if api_key_env is not None else API_KEY
    return _extract_text(_post_responses(base_url, payload, api_key, timeout))


def chat_completions(
    prompt: str,
    images: ImageInput | Sequence[ImageInput] | None = None,
    *,
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    timeout: float = 300,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in _ensure_images(images):
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    api_key = os.environ[api_key_env] if api_key_env is not None else API_KEY
    return _extract_chat_completions_text(_post_chat_completions(base_url, payload, api_key, timeout))


def curl_chat_completions(
    prompt: str,
    images: ImageInput | Sequence[ImageInput] | None = None,
    *,
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    timeout: float = 300,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in _ensure_images(images):
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    api_key = os.environ[api_key_env] if api_key_env is not None else API_KEY
    return _extract_chat_completions_text(_post_curl_chat_completions(base_url, payload, api_key, timeout))


def gemini(
    prompt: str,
    images: ImageInput | Sequence[ImageInput] | None = None,
    *,
    model: str,
    api_key_env: str | None = None,
    timeout: float = 300,
) -> str:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for image in _ensure_images(images):
        data_url = _image_to_data_url(image)
        header, data = data_url.split(",", 1)
        mime_type = header.removeprefix("data:").removesuffix(";base64")
        parts.append({"inlineData": {"mimeType": mime_type, "data": data}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 2048,
        },
    }
    return _extract_gemini_text(_post_gemini(payload, model, timeout))


def bedrock_converse(
    prompt: str,
    images: ImageInput | Sequence[ImageInput] | None = None,
    *,
    region: str,
    model: str,
    timeout: float = 3600,
) -> str:
    content: list[dict[str, Any]] = [{"text": prompt}]
    for image in _ensure_images(images):
        data_url = _image_to_data_url(image)
        header, data = data_url.split(",", 1)
        mime_type = header.removeprefix("data:").removesuffix(";base64")
        image_format = mime_type.rsplit("/", 1)[-1]
        if image_format == "jpg":
            image_format = "jpeg"
        content.append(
            {
                "image": {
                    "format": image_format,
                    "source": {"bytes": data},
                }
            }
        )

    payload = {
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {
            "temperature": 0,
            "maxTokens": 2048,
        },
    }
    return _extract_bedrock_converse_text(_post_bedrock_converse(payload, region, model, timeout))
