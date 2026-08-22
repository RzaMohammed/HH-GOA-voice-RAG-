"""
Tests for Voice STT and TTS modules.
"""

import pytest
from voice_rag.voice.sarvam_stt import SarvamSTT, MockSTT, get_stt, _detect_audio_mimetype
from voice_rag.voice.elevenlabs_stt import ElevenLabsSTT
from voice_rag.voice.tts import SarvamTTS, ElevenLabsTTS, MockTTS, get_tts, TTSResult


class TestVoiceSTT:
    def test_mock_stt(self):
        stt = MockSTT(default_text="Testing voice query")
        res = stt.transcribe_sync()
        assert res.text == "Testing voice query"
        assert res.provider == "mock"

    def test_get_stt_factory(self):
        stt = get_stt("mock")
        assert isinstance(stt, MockSTT)

    def test_audio_mimetype_detection(self):
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt "
        filename, mime = _detect_audio_mimetype(wav_header)
        assert filename == "audio.wav"
        assert mime == "audio/wav"

        webm_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
        filename, mime = _detect_audio_mimetype(webm_header)
        assert filename == "audio.webm"
        assert mime == "audio/webm"


class TestVoiceTTS:
    @pytest.mark.asyncio
    async def test_mock_tts(self):
        tts = MockTTS()
        res = await tts.synthesize("Hello world", language="en")
        assert isinstance(res, TTSResult)
        assert res.provider == "mock"

    def test_get_tts_factory(self):
        tts = get_tts("mock")
        assert isinstance(tts, MockTTS)
