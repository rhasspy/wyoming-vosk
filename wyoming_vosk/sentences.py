import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

_LOGGER = logging.getLogger()


@dataclass
class LanguageConfig:
    sentences_mtime_ns: int
    sentences_file_size: int
    sentences: Dict[str, str] = field(default_factory=dict)
    no_correct_patterns: List[re.Pattern] = field(default_factory=list)


# language -> config
_CONFIG_CACHE: Dict[str, LanguageConfig] = {}


def load_sentences_for_language(
    sentences_dir: Union[str, Path], language: str
) -> Optional[LanguageConfig]:
    """Load YAML file for language with sentence templates."""
    sentences_path = Path(sentences_dir) / f"{language}.yaml"
    if not sentences_path.is_file():
        return None

    sentences_stats = sentences_path.stat()
    config = _CONFIG_CACHE.get(language)

    # We will reload if the file modification time or size has changed
    if (
        (config is not None)
        and (sentences_stats.st_mtime_ns == config.sentences_mtime_ns)
        and (sentences_stats.st_size == config.sentences_file_size)
    ):
        # Cache hit
        return config

    # Load YAML
    _LOGGER.debug("Loading %s", sentences_path)
    config = LanguageConfig(
        sentences_mtime_ns=sentences_stats.st_mtime_ns,
        sentences_file_size=sentences_stats.st_size,
    )

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
    # expansion_rules:
    #   <name>: sentence template
    with open(sentences_path, "r", encoding="utf-8") as sentences_file:
        sentences_yaml = yaml.safe_load(sentences_file)
        templates = sentences_yaml.get("sentences")
        if not templates:
            raise ValueError(f"No sentences for {language}")

        sentences = config.sentences

        # Load slot lists
        slot_lists: Dict[str, SlotList] = {}
        for slot_name, slot_info in sentences_yaml.get("lists", {}).items():
            slot_values = slot_info.get("values")
            if not slot_values:
                _LOGGER.warning("No values for list %s, skipping", slot_name)
                continue

            slot_list_values: List[TextSlotValue] = []
            for slot_value in slot_values:
                if isinstance(slot_value, str):
                    value_in: str = slot_value
                    value_out: str = slot_value
                else:
                    # - in: text to say
                    #   out: text to output
                    value_in = slot_value["in"]
                    value_out = slot_value["out"]

                slot_list_values.append(
                    TextSlotValue(TextChunk(value_in), value_out=value_out)
                )

            slot_lists[slot_name] = TextSlotList(slot_list_values)

        # Load expansion rules
        expansion_rules: Dict[str, hassil.Sentence] = {}
        for rule_name, rule_text in sentences_yaml.get("expansion_rules", {}).items():
            expansion_rules[rule_name] = hassil.parse_sentence(rule_text)

        # Generate possible sentences
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
                        input_expression,
                        slot_lists=slot_lists,
                        expansion_rules=expansion_rules,
                    ):
                        sentences[input_text] = output_text or input_text
                else:
                    # Not a template
                    sentences[input_template] = output_text or input_template

        # Load "no correct" patterns
        no_correct_patterns = sentences_yaml.get("no_correct_patterns", [])
        for pattern_text in no_correct_patterns:
            config.no_correct_patterns.append(re.compile(pattern_text))

    _CONFIG_CACHE[language] = config

    return config


def correct_sentence(
    text: str, config: LanguageConfig, score_cutoff: float = 0.0
) -> str:
    """Correct a sentence using rapidfuzz."""
    if not config.sentences:
        return text

    # Don't correct transcripts that match a "no correct" pattern
    for pattern in config.no_correct_patterns:
        if pattern.match(text):
            return text

    try:
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import extractOne
    except ImportError as exc:
        raise Exception("pip3 install wyoming-vosk[limited]") from exc

    result = extractOne(
        text,
        config.sentences.keys(),
        scorer=Levenshtein.distance,
        scorer_kwargs={"weights": (1, 1, 3)},
    )
    fixed_text, score = result[0], result[1]
    _LOGGER.debug(
        "score=%s/%s, original=%s, fixed=%s", score, score_cutoff, text, fixed_text
    )

    if (score_cutoff <= 0) or (score <= score_cutoff):
        # Map to output text
        return config.sentences[fixed_text]

    return text
