"""
Multi-provider Text-to-Speech (TTS) interface.

Supports:
  - Sarvam AI TTS (bulbul:v1) — specialized for Indic languages (Hindi, Bengali, Tamil, Telugu, etc.)
  - ElevenLabs TTS (eleven_multilingual_v2) — high-fidelity multilingual voice synthesis
  - MockTTS — deterministic fallback for offline testing
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from voice_rag.config import get_settings


# ═══════════════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════════════

class TTSResult(BaseModel):
    """Output from text-to-speech synthesis."""
    audio_base64: str = ""
    mime_type: str = "audio/wav"
    provider: str = "mock"
    language: str = "en"
    duration_ms: float = 0.0
    is_fallback: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Base Class
# ═══════════════════════════════════════════════════════════════════════════

class BaseTTS(ABC):
    """Abstract base for Text-to-Speech providers."""

    provider_name: str = "base"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text into audio."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Sarvam AI TTS (Bulbul)
# ═══════════════════════════════════════════════════════════════════════════

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Mapping language code prefixes to Sarvam standard codes
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
}


class SarvamTTS(BaseTTS):
    """Sarvam AI Bulbul Text-to-Speech client."""

    provider_name = "sarvam"

    def __init__(
        self,
        api_key: Optional[str] = None,
        speaker: str = "meera",
        model: str = "bulbul:v1",
    ):
        cfg = get_settings()
        self.api_key = api_key or cfg.sarvam_api_key
        self.speaker = speaker
        self.model = model

        if not self.api_key:
            logger.warning("Sarvam API key not set — TTS will fail or fall back")

    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required for Sarvam TTS")

        lang = language or "hi-IN"
        # Normalize language code to full Sarvam format
        if lang in SARVAM_LANG_MAP:
            lang = SARVAM_LANG_MAP[lang]
        elif "-" not in lang:
            lang = f"{lang}-IN"

        speaker = voice or self.speaker

        # Truncate if text exceeds single-request limit (approx 500 chars)
        clean_text = text.strip()
        if len(clean_text) > 480:
            clean_text = clean_text[:480] + "..."

        payload = {
            "inputs": [clean_text],
            "target_language_code": lang,
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": self.model,
        }

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(SARVAM_TTS_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            elapsed_ms = (time.perf_counter() - t0) * 1000
            audios = data.get("audios", [])
            audio_b64 = audios[0] if audios else ""

            return TTSResult(
                audio_base64=audio_b64,
                mime_type="audio/wav",
                provider="sarvam",
                language=lang,
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            cfg = get_settings()
            if cfg.elevenlabs_api_key:
                logger.warning(f"Sarvam TTS failed ({exc}), falling back to ElevenLabs TTS...")
                eleven_tts = ElevenLabsTTS(api_key=cfg.elevenlabs_api_key)
                return await eleven_tts.synthesize(text=text, language=language, voice=voice)
            raise


# ═══════════════════════════════════════════════════════════════════════════
# ElevenLabs TTS
# ═══════════════════════════════════════════════════════════════════════════

ELEVENLABS_TTS_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" voice


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs Multilingual Text-to-Speech client."""

    provider_name = "elevenlabs"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
    ):
        cfg = get_settings()
        self.api_key = api_key or cfg.elevenlabs_api_key
        self.voice_id = voice_id or DEFAULT_ELEVENLABS_VOICE_ID
        self.model_id = model_id

        if not self.api_key:
            logger.warning("ElevenLabs API key not set — TTS will fail or fall back")

    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs TTS")

        voice_id = voice or self.voice_id
        url = f"{ELEVENLABS_TTS_BASE}/{voice_id}"

        clean_text = text.strip()
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."

        payload = {
            "text": clean_text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
            },
        }

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                audio_bytes = response.content

            elapsed_ms = (time.perf_counter() - t0) * 1000
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return TTSResult(
                audio_base64=audio_b64,
                mime_type="audio/mpeg",
                provider="elevenlabs",
                language=language or "en",
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            logger.warning(f"ElevenLabs TTS synthesis failed ({exc}), falling back to silent result")
            return TTSResult(
                audio_base64="",
                mime_type="audio/mpeg",
                provider="mock",
                language=language or "en",
                duration_ms=(time.perf_counter() - t0) * 1000,
                is_fallback=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Mock TTS
# ═══════════════════════════════════════════════════════════════════════════

class MockTTS(BaseTTS):
    """Deterministic Mock TTS returning empty/placeholder audio."""

    provider_name = "mock"

    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        return TTSResult(
            audio_base64="",
            mime_type="audio/wav",
            provider="mock",
            language=language or "en",
            duration_ms=0.1,
            is_fallback=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_tts(provider: Optional[str] = None) -> BaseTTS:
    """
    Get TTS provider instance.
    'auto' selects Sarvam if available, else ElevenLabs, else Mock.
    """
    cfg = get_settings()
    prov = (provider or "auto").lower()

    if prov == "sarvam":
        return SarvamTTS()
    if prov == "elevenlabs":
        return ElevenLabsTTS()
    if prov == "mock":
        return MockTTS()

    # Auto mode: prioritize available API keys
    if cfg.sarvam_api_key:
        return SarvamTTS()
    if cfg.elevenlabs_api_key:
        return ElevenLabsTTS()
    return MockTTS()
