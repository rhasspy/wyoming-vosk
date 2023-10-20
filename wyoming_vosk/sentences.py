import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# language -> input text -> output text
_SENTENCES_CACHE: Dict[str, Dict[str, str]] = {}

_LOGGER = logging.getLogger()


def load_sentences_for_language(
    sentences_dir: Union[str, Path], language: str
) -> Optional[Dict[str, str]]:
    """Load YAML file for language with sentence templates."""
    if language in _SENTENCES_CACHE:
        # Cache hit
        return _SENTENCES_CACHE[language]

    sentences: Dict[str, str] = {}
    sentences_path = Path(sentences_dir) / f"{language}.yaml"

    if not sentences_path.exists():
        return None

    try:
        import hassil.parse_expression
        import hassil.sample
        import yaml
        from hassil.intents import SlotList, TextChunk, TextSlotList, TextSlotValue
    except ImportError as exc:
        raise Exception("pip3 install wyoming-vosk[limited]") from exc

    # sentences:
    #   - same text in and out
    #   - in: text in
    #     out: different text out
    #   - in:
    #       - multiple text
    #       - multiple text in
    #     out: different text out
    # lists:
    #   <name>:
    #     - value 1
    #     - value 2
    with open(sentences_path, "r", encoding="utf-8") as sentences_file:
        sentences_yaml = yaml.safe_load(sentences_file)
        templates = sentences_yaml.get("sentences")
        if not templates:
            raise ValueError(f"No sentences for {language}")

        slot_lists: Dict[str, SlotList] = {
            slot_name: TextSlotList(
                [
                    TextSlotValue(TextChunk(value), value_out=value)
                    for value in slot_values
                ]
            )
            for slot_name, slot_values in sentences_yaml.get("lists", {}).items()
        }

        for template in templates:
            if isinstance(template, str):
                input_templates: List[str] = [template]
                output_text: Optional[str] = None
            else:
                input_str_or_list = template["in"]
                if isinstance(input_str_or_list, str):
                    # One template
                    input_templates = [input_str_or_list]
                else:
                    # Multiple templates
                    input_templates = input_str_or_list

                output_text = template.get("out")

            for input_template in input_templates:
                if hassil.intents.is_template(input_template):
                    # Generate possible texts
                    input_expression = hassil.parse_expression.parse_sentence(
                        input_template
                    )
                    for input_text in hassil.sample.sample_expression(
                        input_expression, slot_lists=slot_lists
                    ):
                        sentences[input_text] = output_text or input_text
                else:
                    # Not a template
                    sentences[input_template] = output_text or input_template

    _SENTENCES_CACHE[language] = sentences

    return sentences


def correct_sentence(
    text: str, sentences: Dict[str, str], score_cutoff: float = 0.0
) -> str:
    """Correct a sentence using rapidfuzz."""
    if not sentences:
        return text

    try:
        from rapidfuzz.process import extractOne
    except ImportError as exc:
        raise Exception("pip3 install wyoming-vosk[limited]") from exc

    fixed_text, score, _key = extractOne(text, sentences.keys())
    _LOGGER.debug("score=%s, original=%s, fixed=%s", score, text, fixed_text)

    if score >= score_cutoff:
        # Map to output text
        return sentences[fixed_text]

    return text
