#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from vosk import KaldiRecognizer, Model, SetLogLevel
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .download import MODELS, download_model
from .sentences import correct_sentence, load_sentences_for_language

_LOGGER = logging.getLogger()
_DIR = Path(__file__).parent


class State:
    """State of system"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.models: Dict[str, Model] = {}

    def get_model(self, language: str, model_name: str) -> Optional[Model]:
        model = self.models.get(model_name)
        if model is not None:
            return model

        # Check if model is already downloaded
        for data_dir in self.args.data_dir:
            model_dir = Path(data_dir) / model_name
            if model_dir.is_dir():
                _LOGGER.debug("Found %s at %s", model_name, model_dir)
                model = Model(str(model_dir))
                self.models[model_name] = model
                return model

        model_dir = download_model(language, model_name, self.args.download_dir)
        model = Model(str(model_dir))
        self.models[model_name] = model
        return model


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument("--language", default="en")
    #
    parser.add_argument(
        "--sentences-dir", help="Directory with YAML files for each language"
    )
    parser.add_argument(
        "--correct-sentences",
        nargs="?",
        type=float,
        const=0,
        help="Enable sentence correction with optional score cutoff (0=strict, 100=relaxed)",
    )
    parser.add_argument("--limit-sentences", action="store_true")
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    if (args.correct_sentences is not None) or args.limit_sentences:
        if not args.sentences_dir:
            _LOGGER.fatal(
                "--sentences-dir is required with --correct-sentences or --limit-sentences"
            )
            sys.exit(1)

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    SetLogLevel(0)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="vosk",
                description="A speech recognition toolkit",
                attribution=Attribution(
                    name="Alpha Cephei",
                    url="https://alphacephei.com/vosk/",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Alpha Cephei",
                            url="https://alphacephei.com/vosk/models",
                        ),
                        installed=True,
                        languages=[language],
                    )
                    for language, model_names in MODELS.items()
                    for model_name in model_names
                ],
            )
        ],
    )

    state = State(args)

    _LOGGER.info("Ready")

    # Start server
    server = AsyncServer.from_uri(args.uri)

    try:
        await server.run(partial(VoskEventHandler, wyoming_info, args, state))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class VoskEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        state: State,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.client_id = str(time.monotonic_ns())
        self.state = state
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.audio_buffer = bytes()
        self.language: Optional[str] = None
        self.model_name: Optional[str] = None
        self.recognizer: Optional[KaldiRecognizer] = None

        _LOGGER.debug("Client connected: %s", self.client_id)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info to client: %s", self.client_id)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self.language = transcribe.language
            self.model_name = transcribe.name
        elif AudioStart.is_type(event.type):
            pass
        elif AudioChunk.is_type(event.type):
            if self.recognizer is None:
                self.language = self.language or self.cli_args.language
                if not self.model_name:
                    self.model_name = MODELS[self.language][0]

                _LOGGER.debug(
                    "Loading %s for language %s", self.model_name, self.language
                )
                model = self.state.get_model(self.language, self.model_name)
                assert model is not None, f"No model named: {self.model_name}"

                self.recognizer = self._load_recognizer(model)

            assert self.recognizer is not None

            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)
            self.recognizer.AcceptWaveform(chunk.audio)

        elif AudioStop.is_type(event.type):
            assert self.recognizer is not None
            result = json.loads(self.recognizer.FinalResult())
            text = result["text"]
            _LOGGER.debug("Transcript for client %s: %s", self.client_id, text)

            if self.cli_args.correct_sentences is not None:
                original_text = text
                text = self._fix_transcript(original_text)
                if text != original_text:
                    _LOGGER.debug("Corrected transcript: %s", text)

            await self.write_event(Transcript(text=text).event())

            return False
        else:
            _LOGGER.debug("Unexpected event: type=%s, data=%s", event.type, event.data)

        return True

    async def disconnect(self) -> None:
        _LOGGER.debug("Client disconnected: %s", self.client_id)

    def _load_recognizer(self, model: Model) -> KaldiRecognizer:
        if not self.cli_args.limit_sentences:
            # Open-ended
            return KaldiRecognizer(model, 16000)

        assert self.language, "Language not set"
        sentences = load_sentences_for_language(
            self.cli_args.sentences_dir, self.language
        )

        _LOGGER.debug("Limiting to %s possible sentence(s)", len(sentences))

        limited_sentences_str = json.dumps(list(sentences.keys()))
        return KaldiRecognizer(model, 16000, limited_sentences_str)

    def _fix_transcript(self, text: str) -> str:

        assert self.language, "Language not set"
        sentences = load_sentences_for_language(
            self.cli_args.sentences_dir, self.language
        )

        return correct_sentence(
            text, sentences, score_cutoff=self.cli_args.correct_sentences
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
