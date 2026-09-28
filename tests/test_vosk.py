"""Tests for wyoming-vosk"""

import asyncio
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import List, Optional

import pytest
from wyoming.asr import Transcript, TranscriptChunk, TranscriptStart, TranscriptStop
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024

# Need to give time for the model to download
_TRANSCRIBE_TIMEOUT = 60

_TEST_PHRASE = {
    "ar": "تتتلمذ اللغة العربية",
    "en": "turn on the living room lamp",
    "uk": "ви розмовляєте українською",
}


@pytest.mark.parametrize("language", ["en", "uk", "ar"])
@pytest.mark.asyncio
async def test_vosk(language: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_vosk",
        "--uri",
        "stdio://",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        language,
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Check info
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=1)
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert len(info.asr) == 1, "Expected one asr service"
        asr = info.asr[0]
        assert len(asr.models) > 0, "Expected at least one model"
        assert any(
            language in m.languages for m in asr.models
        ), f"Expected a model for {language}"
        break

    # Test known WAV
    with wave.open(str(_DIR / f"{language}.wav"), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None

        if not Transcript.is_type(event.type):
            continue

        transcript = Transcript.from_event(event)
        text = transcript.text.lower().strip()
        assert text == _TEST_PHRASE[language]
        break

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()


@pytest.mark.asyncio
async def test_streaming() -> None:
    """Partial transcripts are streamed as transcript chunks."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_vosk",
        "--uri",
        "stdio://",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        "en",
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Streaming must be advertised
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=1)
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert info.asr[0].supports_transcript_streaming
        break

    with wave.open(str(_DIR / "en.wav"), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    started = False
    chunk_texts: List[str] = []
    text: Optional[str] = None
    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None

        if TranscriptStart.is_type(event.type):
            started = True
        elif TranscriptChunk.is_type(event.type):
            assert started, "transcript-start must come first"
            chunk_texts.append(TranscriptChunk.from_event(event).text)
        elif Transcript.is_type(event.type):
            text = Transcript.from_event(event).text
        elif TranscriptStop.is_type(event.type):
            assert text is not None, "transcript must come before transcript-stop"
            break

    assert chunk_texts, "Expected at least one transcript chunk"

    # Chunks are appended, so they must join back into the transcript
    assert "".join(chunk_texts).lower().strip() == _TEST_PHRASE["en"]
    assert text is not None
    assert text.lower().strip() == _TEST_PHRASE["en"]

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()


@pytest.mark.asyncio
async def test_no_streaming() -> None:
    """No transcript chunks are sent with --no-streaming."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_vosk",
        "--uri",
        "stdio://",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        "en",
        "--no-streaming",
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(async_read_event(proc.stdout), timeout=1)
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert not info.asr[0].supports_transcript_streaming
        break

    with wave.open(str(_DIR / "en.wav"), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None
        assert not TranscriptStart.is_type(event.type)
        assert not TranscriptChunk.is_type(event.type)

        if not Transcript.is_type(event.type):
            continue

        assert Transcript.from_event(event).text.lower().strip() == _TEST_PHRASE["en"]
        break

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()
