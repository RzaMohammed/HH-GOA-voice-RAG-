"""
Sarvam AI Speech-to-Text client.

Integrates with Sarvam AI's STT API (saaras:v1 / saaras:v2) supporting
Hindi, English, and other Indic language audio transcription.
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
# Sarvam STT API Client
# ═══════════════════════════════════════════════════════════════════════════

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"


class SarvamSTT:
    """Sarvam AI Speech-to-Text transcription client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v2",
        language: str = "hi-IN",
    ):
        cfg = get_settings()
        self.api_key = api_key or cfg.sarvam_api_key
        self.model = model
        self.language = language

        if not self.api_key:
            logger.warning("Sarvam API key not set — STT calls will fail")

    async def transcribe(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio using Sarvam AI STT.

        Args:
            audio_data: Raw audio bytes or path to an audio file.
            language:   Language code (e.g., 'hi-IN', 'en-IN').

        Returns:
            STTResult with transcribed text.
        """
        language = language or self.language

        # Read file if path provided
        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_data = audio_path.read_bytes()

        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    SARVAM_STT_URL,
                    headers={
                        "api-subscription-key": self.api_key,
                    },
                    files={"file": ("audio.wav", audio_data, "audio/wav")},
                    data={
                        "model": self.model,
                        "language_code": language,
                    },
                )
                response.raise_for_status()
                result = response.json()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            transcript = result.get("transcript", "")
            # Some Sarvam responses have 'translated_text' for cross-lingual
            if not transcript:
                transcript = result.get("translated_text", "")

            return STTResult(
                text=transcript,
                language=language.split("-")[0] if "-" in language else language,
                confidence=result.get("confidence", 0.9),
                provider="sarvam",
                duration_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as exc:
            logger.error(f"Sarvam STT API error: {exc.response.status_code} — {exc.response.text}")
            raise
        except Exception as exc:
            logger.error(f"Sarvam STT failed: {exc}")
            raise

    def transcribe_sync(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """Synchronous wrapper for transcribe()."""
        import asyncio
        return asyncio.run(self.transcribe(audio_data, language))


# ═══════════════════════════════════════════════════════════════════════════
# Mock STT for offline testing
# ═══════════════════════════════════════════════════════════════════════════

class MockSTT:
    """Deterministic mock STT for testing without API keys."""

    def __init__(self, default_text: str = "What is the capital of India?"):
        self._default_text = default_text

    async def transcribe(
        self,
        audio_data: Union[bytes, Path, str, None] = None,
        language: Optional[str] = None,
    ) -> STTResult:
        return STTResult(
            text=self._default_text,
            language=language or "en",
            confidence=1.0,
            provider="mock",
            duration_ms=0.1,
        )

    def transcribe_sync(
        self,
        audio_data: Union[bytes, Path, str, None] = None,
        language: Optional[str] = None,
    ) -> STTResult:
        import asyncio
        return asyncio.run(self.transcribe(audio_data, language))


def get_stt(provider: str = "auto") -> Union[SarvamSTT, MockSTT]:
    """
    Factory for STT providers.

    'auto' uses Sarvam if API key is set, otherwise falls back to mock.
    """
    cfg = get_settings()
    if provider == "sarvam" or (provider == "auto" and cfg.sarvam_api_key):
        return SarvamSTT()
    return MockSTT()
