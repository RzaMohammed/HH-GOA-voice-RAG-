"""
ElevenLabs Speech-to-Text client.

Alternative STT provider using ElevenLabs' transcription API (Scribe v1).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import httpx
from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import STTResult


# ═══════════════════════════════════════════════════════════════════════════
# ElevenLabs STT API Client
# ═══════════════════════════════════════════════════════════════════════════

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def _detect_audio_mimetype(data: bytes) -> tuple[str, str]:
    """Detect filename extension and MIME type from audio magic bytes."""
    if data.startswith(b"RIFF") and b"WAVE" in data[:12]:
        return "audio.wav", "audio/wav"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio.webm", "audio/webm"
    if data.startswith(b"OggS"):
        return "audio.ogg", "audio/ogg"
    if data.startswith(b"ID3") or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio.mp3", "audio/mpeg"
    return "audio.wav", "audio/wav"


class ElevenLabsSTT:
    """ElevenLabs Speech-to-Text transcription client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "scribe_v1",
    ):
        cfg = get_settings()
        self.api_key = api_key or cfg.elevenlabs_api_key
        self.model = model

        if not self.api_key:
            logger.warning("ElevenLabs API key not set — STT calls will fail or fall back")

    async def transcribe(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio using ElevenLabs STT.

        Args:
            audio_data: Raw audio bytes or path to an audio file.
            language:   Language code hint (e.g., 'en', 'hi', 'bn').

        Returns:
            STTResult with transcribed text.
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs STT")

        # Read file if path provided
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_data = audio_path.read_bytes()

        filename, mime_type = _detect_audio_mimetype(audio_data)
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                data = {"model_id": self.model}
                if language and language != "auto" and language != "unknown":
                    # Normalize language code to short 2-letter (e.g. hi-IN -> hi)
                    data["language_code"] = language.split("-")[0] if "-" in language else language

                response = await client.post(
                    ELEVENLABS_STT_URL,
                    headers={
                        "xi-api-key": self.api_key,
                    },
                    files={"file": (filename, audio_data, mime_type)},
                    data=data,
                )
                response.raise_for_status()
                result = response.json()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            transcript = result.get("text", "")
            detected_lang = result.get("language_code", language or "en")

            return STTResult(
                text=transcript.strip(),
                language=detected_lang,
                confidence=result.get("confidence", 0.95),
                provider="elevenlabs",
                duration_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"ElevenLabs STT API error: {exc.response.status_code} — {exc.response.text}"
            )
            raise
        except Exception as exc:
            logger.error(f"ElevenLabs STT failed: {exc}")
            raise

    def transcribe_sync(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """Synchronous wrapper for transcribe()."""
        import asyncio
        return asyncio.run(self.transcribe(audio_data, language))
