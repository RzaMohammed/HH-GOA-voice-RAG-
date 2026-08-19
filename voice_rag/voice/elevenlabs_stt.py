"""
ElevenLabs Speech-to-Text client.

Alternative STT provider using ElevenLabs' transcription API.
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
            logger.warning("ElevenLabs API key not set — STT calls will fail")

    async def transcribe(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio using ElevenLabs STT.

        Args:
            audio_data: Raw audio bytes or path to an audio file.
            language:   Language code hint (e.g., 'en', 'hi').

        Returns:
            STTResult with transcribed text.
        """
        # Read file if path provided
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_data = audio_path.read_bytes()

        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                data = {"model_id": self.model}
                if language:
                    data["language_code"] = language

                response = await client.post(
                    ELEVENLABS_STT_URL,
                    headers={
                        "xi-api-key": self.api_key,
                    },
                    files={"file": ("audio.wav", audio_data, "audio/wav")},
                    data=data,
                )
                response.raise_for_status()
                result = response.json()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            transcript = result.get("text", "")
            detected_lang = result.get("language_code", language or "en")

            return STTResult(
                text=transcript,
                language=detected_lang,
                confidence=result.get("confidence", 0.9),
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
