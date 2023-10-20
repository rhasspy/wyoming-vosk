"""Utility for downloading faster-whisper models."""
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Union
from urllib.request import urlopen
from zipfile import ZipFile

URL_FORMAT = "https://huggingface.co/rhasspy/vosk-models/resolve/main/{language}/{model_name}.zip"

_LOGGER = logging.getLogger(__name__)


MODELS: Dict[str, List[str]] = {
    "ar": ["vosk-model-ar-mgb2-0.4"],
    "br": ["vosk-model-br-0.8"],
    "ca": ["vosk-model-small-ca-0.4"],
    "cs": ["vosk-model-small-cs-0.4-rhasspy"],
    "de": ["vosk-model-small-de-0.15"],
    "en": ["vosk-model-small-en-us-0.15", "vosk-model-en-us-0.22-lgraph"],
    "eo": ["vosk-model-small-eo-0.42"],
    "es": ["vosk-model-small-es-0.42"],
    "fa": ["vosk-model-small-fa-0.5"],
    "fr": ["vosk-model-small-fr-0.22"],
    "hi": ["vosk-model-small-hi-0.22"],
    "it": ["vosk-model-small-it-0.22"],
    "ja": ["vosk-model-small-ja-0.22"],
    "ko": ["vosk-model-small-ko-0.22"],
    "kz": ["vosk-model-small-kz-0.15"],
    "nl": ["vosk-model-small-nl-0.22", "vosk-model-nl-spraakherkenning-0.6-lgraph"],
    "pl": ["vosk-model-small-pl-0.22"],
    "pt": ["vosk-model-small-pt-0.3"],
    "ru": ["vosk-model-small-ru-0.22"],
    "sv": ["vosk-model-small-sv-rhasspy-0.15"],
    "tl": ["vosk-model-tl-ph-generic-0.6"],
    "tr": ["vosk-model-small-tr-0.3"],
    "uk": ["vosk-model-small-uk-v3-small"],
    "uz": ["vosk-model-small-uz-0.22"],
    "vi": ["vosk-model-small-vn-0.4", "vosk-model-vn-0.4"],
    "zh": ["vosk-model-small-cn-0.22"],
}


def download_model(language: str, model_name: str, dest_dir: Union[str, Path]) -> Path:
    """
    Downloads/extracts model directly to destination directory.

    Returns directory of downloaded model.
    """
    dest_dir = Path(dest_dir)
    model_dir = dest_dir / model_name

    if model_dir.is_dir():
        # Remove model directory if it already exists
        shutil.rmtree(model_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    model_url = URL_FORMAT.format(language=language, model_name=model_name)
    _LOGGER.debug("Downloading: %s", model_url)

    with urlopen(model_url) as response:
        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".zip") as temp_model_file:
            shutil.copyfileobj(response, temp_model_file)
            temp_model_file.seek(0)
            with ZipFile(temp_model_file.name, mode="r") as model_file:
                model_file.extractall(dest_dir)

    return model_dir
