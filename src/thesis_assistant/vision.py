"""Vision fallback: read content that text extractors cannot see.

PyPDFLoader and Docx2txtLoader only read the native text layer, so scanned
PDF pages and images embedded in .docx files would be invisible to the RAG
pipeline. This module sends such images to a vision model (through
OpenRouter) and returns plain text that can be indexed or used as context.

Shared by every call site so they behave identically:
    - ingestion.py     sparse PDF pages and images embedded in corpus .docx files
    - rag.py           images embedded in a student's uploaded .docx
    - telegram_bot.py  photos sent directly to the bot

Every public function degrades gracefully: when the fallback is disabled,
the API key is missing or the API call fails, callers receive ``None`` or an
empty result instead of an exception.
"""

import base64
from typing import Any, List, Optional

import docx as python_docx
from openai import OpenAI

from thesis_assistant import config

VISION_PROMPT = (
    "This is a page or image from an academic/university document (thesis "
    "template, methodology guide, or similar). Transcribe any visible text "
    "verbatim. If the page contains a diagram, flowchart, chart, or table "
    "rendered as an image, describe its structure and content in plain "
    "text (e.g. steps in a flowchart, rows/columns of a table, axes/values "
    "of a chart) so the information is searchable. Respond with plain text "
    "only — no markdown, no commentary about what you're doing."
)

_vision_client: Optional[OpenAI] = None  # created lazily; see _get_vision_client


def _get_vision_client() -> Optional[OpenAI]:
    """Return a cached OpenRouter client, or None when vision is unavailable.

    The client is built lazily so a missing API key only disables the
    fallback instead of crashing the process at import time.
    """
    global _vision_client
    if not config.VISION_FALLBACK_ENABLED:
        return None
    if _vision_client is None:
        if not config.OPENROUTER_API_KEY:
            return None
        _vision_client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)
    return _vision_client


def guess_image_mime(image_bytes: bytes) -> str:
    """Identify an image's MIME type from its magic bytes (defaults to PNG)."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # most Office-embedded images are PNG


def describe_image_bytes(
    image_bytes: bytes,
    mime_type: str = "image/png",
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Transcribe or describe an image with the vision model.

    Args:
        image_bytes: Raw image data.
        mime_type: MIME type of ``image_bytes``.
        prompt: Overrides the default indexing-oriented ``VISION_PROMPT``
            (the Telegram bot passes the student's own caption).
        system_prompt: Optional system message for conversational use.

    Returns:
        The model's text, or None if the fallback is disabled, no API key is
        configured, the call failed, or the model returned nothing.
    """
    client = _get_vision_client()
    if client is None:
        return None

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    messages: List[Any] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
            ],
        }
    )

    try:
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=messages,
            max_tokens=config.MAX_VISION_OUTPUT_TOKENS,
        )
    except Exception:
        return None
    return (response.choices[0].message.content or "").strip() or None


def extract_docx_images(docx_path: str) -> List[bytes]:
    """Return the raw bytes of every image embedded in a .docx file.

    A .docx is a zip archive; python-docx exposes images as related parts.
    Order is not guaranteed to match reading order, which is acceptable
    because each image is described independently.
    """
    document = python_docx.Document(str(docx_path))
    images: List[bytes] = []
    for rel in document.part.rels.values():
        if "image" in rel.reltype:
            try:
                images.append(rel.target_part.blob)
            except Exception:
                continue
    return images


def describe_docx_images(docx_path: str) -> List[str]:
    """Describe every image embedded in a .docx file.

    Returns one text snippet per successfully described image, or an empty
    list when the fallback is disabled or the file cannot be read.
    """
    if not config.VISION_FALLBACK_ENABLED:
        return []
    try:
        images = extract_docx_images(docx_path)
    except Exception:
        return []

    snippets = []
    for image_bytes in images:
        description = describe_image_bytes(image_bytes, mime_type=guess_image_mime(image_bytes))
        if description:
            snippets.append(description)
    return snippets


def format_image_snippets(snippets: List[str]) -> str:
    """Join image descriptions into a labelled block appended to document text."""
    return "\n\n".join(
        f"[Embedded image {i}]\n{snippet}" for i, snippet in enumerate(snippets, start=1)
    )
