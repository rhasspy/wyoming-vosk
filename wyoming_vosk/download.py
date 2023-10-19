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


MODELS: Dict[str, List[str]] = {"en": ["vosk-model-small-en-us-0.15"]}


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
