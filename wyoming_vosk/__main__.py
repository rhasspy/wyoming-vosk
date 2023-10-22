#!/usr/bin/env python
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
_CASING = {
    "casefold": lambda s: s.casefold(),
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "keep": lambda s: s,
}
_UNK = "[unk]"


class State:
    """State of system"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.models: Dict[str, Model] = {}

    def get_model(
        self, language: str, model_name: Optional[str] = None
    ) -> Optional[Model]:
        # Allow override
        model_name = self.args.model_for_language.get(language, model_name)

        if not model_name:
            # Use model matching --model-index
            available_models = MODELS[language]
            model_name = available_models[
                min(self.args.model_index, len(available_models) - 1)
            ]

        assert model_name is not None

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
    parser.add_argument("--language", default="en", help="Set default model language")
    parser.add_argument(
        "--preload-language",
        action="append",
        default=[],
        help="Preload model for language(s)",
    )
    parser.add_argument(
        "--model-for-language",
        nargs=2,
        metavar=("language", "model"),
        action="append",
        default=[],
        help="Override default model for language",
    )
    parser.add_argument(
        "--casing-for-language",
        nargs=2,
        metavar=("language", "casing"),
        action="append",
        default=[],
        help="Override casing for language (casefold, lower, upper, keep)",
    )
    parser.add_argument(
        "--model-index",
        default=0,
        type=int,
        help="Index of model to use when name is not specified",
    )
    #
    parser.add_argument(
        "--sentences-dir", help="Directory with YAML files for each language"
    )
    parser.add_argument(
        "--correct-sentences",
        nargs="?",
        type=float,
        const=0,
        help="Enable sentence correction with optional score cutoff (0=strict, higher=less strict)",
    )
    parser.add_argument(
        "--limit-sentences",
        action="store_true",
        help="Only sentences in --sentences-dir can be spoken",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Return empty transcript when unknown words are spoken",
    )
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

    # Convert to dict of language -> model
    args.model_for_language = dict(args.model_for_language)

    # Convert to dict of language -> casing
    args.casing_for_language = {
        language: _CASING.get(casing, _CASING["keep"])
        for language, casing in args.casing_for_language
    }

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if args.debug:
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
                        description=model_name.replace("vosk-model-", ""),
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
    for language in args.preload_language:
        _LOGGER.debug("Preloading model for %s", language)
        state.get_model(language)

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
            # Request to transcribe: set language/model
            transcribe = Transcribe.from_event(event)
            self.language = transcribe.language
            self.model_name = transcribe.name
        elif AudioStart.is_type(event.type):
            # Recognized, but we don't do anything until we get an audio chunk
            pass
        elif AudioChunk.is_type(event.type):
            if self.recognizer is None:
                # Load recognizer on first audio chunk
                self.language = self.language or self.cli_args.language
                model = self.state.get_model(self.language, self.model_name)
                assert model is not None, f"No model named: {self.model_name}"

                self.recognizer = self._load_recognizer(model)

            assert self.recognizer is not None

            # Process audio chunk
            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)
            self.recognizer.AcceptWaveform(chunk.audio)

        elif AudioStop.is_type(event.type):
            # Get transcript
            assert self.recognizer is not None
            result = json.loads(self.recognizer.FinalResult())
            casing_func = self.cli_args.casing_for_language.get(
                self.language, _CASING["keep"]
            )
            text = casing_func(result["text"])
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
        """Loads Kaldi recognizer for the model, optionally limited by user-provided sentences."""
        if self.cli_args.limit_sentences:
            assert self.language, "Language not set"
            lang_config = load_sentences_for_language(
                self.cli_args.sentences_dir, self.language
            )
            if (lang_config is not None) and lang_config.sentences:
                _LOGGER.debug(
                    "Limiting to %s possible sentence(s)", len(lang_config.sentences)
                )
                limited_sentences = list(lang_config.sentences.keys())
                if self.cli_args.allow_unknown:
                    # Enable unknown words (will return empty transcript)
                    limited_sentences.append(_UNK)

                limited_sentences_str = json.dumps(
                    limited_sentences, ensure_ascii=False
                )
                return KaldiRecognizer(model, 16000, limited_sentences_str)

        # Open-ended
        return KaldiRecognizer(model, 16000)

    def _fix_transcript(self, text: str) -> str:
        """Corrects a transcript using user-provided sentences."""
        if self.cli_args.allow_unknown and (text == _UNK):
            return ""

        assert self.language, "Language not set"
        lang_config = load_sentences_for_language(
            self.cli_args.sentences_dir, self.language
        )

        if lang_config is None:
            # Can't fix
            return text

        return correct_sentence(
            text, lang_config, score_cutoff=self.cli_args.correct_sentences
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
