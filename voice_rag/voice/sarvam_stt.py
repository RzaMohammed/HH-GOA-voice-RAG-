"""
Sarvam AI Speech-to-Text client.

Integrates with Sarvam AI's STT API (saaras:v2 / saaras:v1) supporting
Hindi, English, and other Indic language audio transcription.
"""

from __future__ import annotations

import asyncio
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

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

# Language normalization mapping
SARVAM_LANG_MAP = {
    "hi": "hi-IN",
    "en": "en-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "od": "od-IN",
    "auto": "unknown",
}


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
            logger.warning("Sarvam API key not set — STT calls will fall back to browser transcript")

    async def transcribe(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio using Sarvam AI STT.

        Args:
            audio_data: Raw audio bytes or path to an audio file.
            language:   Language code (e.g., 'hi-IN', 'en-IN', 'hi', 'auto').

        Returns:
            STTResult with transcribed text.
        """
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not configured")

        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_data = audio_path.read_bytes()

        lang = language or self.language
        if lang in SARVAM_LANG_MAP:
            lang = SARVAM_LANG_MAP[lang]
        elif lang and "-" not in lang and lang != "unknown":
            lang = f"{lang}-IN"

        filename, mime_type = _detect_audio_mimetype(audio_data)
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                data_payload = {
                    "model": self.model,
                }
                if lang and lang != "unknown":
                    data_payload["language_code"] = lang
                else:
                    data_payload["language_code"] = "unknown"

                # Try speech-to-text endpoint first, fallback to translate
                response = await client.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": self.api_key},
                    files={"file": (filename, audio_data, mime_type)},
                    data=data_payload,
                )

                if response.status_code == 404:
                    response = await client.post(
                        SARVAM_STT_TRANSLATE_URL,
                        headers={"api-subscription-key": self.api_key},
                        files={"file": (filename, audio_data, mime_type)},
                        data=data_payload,
                    )

                response.raise_for_status()
                result = response.json()

            elapsed_ms = (time.perf_counter() - t0) * 1000

            transcript = result.get("transcript", "")
            if not transcript:
                transcript = result.get("translated_text", "")

            detected_lang = result.get("language_code", lang)
            if "-" in detected_lang:
                detected_lang = detected_lang.split("-")[0]

            return STTResult(
                text=transcript.strip(),
                language=detected_lang,
                confidence=1.0,
                provider="sarvam",
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            cfg = get_settings()
            if cfg.elevenlabs_api_key:
                logger.warning(f"Sarvam STT failed ({exc}), falling back to ElevenLabs STT...")
                from voice_rag.voice.elevenlabs_stt import ElevenLabsSTT
                eleven_stt = ElevenLabsSTT(api_key=cfg.elevenlabs_api_key)
                return await eleven_stt.transcribe(audio_data, language=language)
            raise

    def transcribe_sync(
        self,
        audio_data: Union[bytes, Path, str],
        language: Optional[str] = None,
    ) -> STTResult:
        return asyncio.run(self.transcribe(audio_data, language))


# ═══════════════════════════════════════════════════════════════════════════
# Mock STT for offline testing
# ═══════════════════════════════════════════════════════════════════════════

class MockSTT:
    """Deterministic mock STT for testing without API keys."""

    def __init__(self, default_text: str = ""):
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
        return asyncio.run(self.transcribe(audio_data, language))


def get_stt(provider: str = "auto") -> Union[SarvamSTT, MockSTT]:
    """
    Factory for STT providers.
    """
    cfg = get_settings()
    prov = (provider or "auto").lower()

    if prov == "sarvam":
        return SarvamSTT()
    if prov == "elevenlabs":
        from voice_rag.voice.elevenlabs_stt import ElevenLabsSTT
        return ElevenLabsSTT()
    if prov == "mock":
        return MockSTT()

    if cfg.sarvam_api_key:
        return SarvamSTT()
    if cfg.elevenlabs_api_key:
        from voice_rag.voice.elevenlabs_stt import ElevenLabsSTT
        return ElevenLabsSTT()
    return MockSTT()
