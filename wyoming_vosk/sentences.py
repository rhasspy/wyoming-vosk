import itertools
import logging
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from hassil.expression import Expression, Sentence
    from hassil.intents import SlotList

_LOGGER = logging.getLogger()


@dataclass
class LanguageConfig:
    sentences_mtime_ns: int
    sentences_file_size: int
    sentences: Dict[str, str] = field(default_factory=dict)
    no_correct_patterns: List[re.Pattern] = field(default_factory=list)
    unknown_text: Optional[str] = None


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
                values_in: List[str] = []

                if isinstance(slot_value, str):
                    values_in.append(slot_value)
                    value_out: str = slot_value
                else:
                    # - in: text to say
                    #   out: text to output
                    value_in = slot_value["in"]
                    value_out = slot_value["out"]

                    if hassil.intents.is_template(value_in):
                        input_expression = hassil.parse_expression.parse_sentence(
                            value_in
                        )
                        for input_text in hassil.sample.sample_expression(
                            input_expression,
                        ):
                            values_in.append(input_text)
                    else:
                        values_in.append(value_in)

                for value_in in values_in:
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
                    for input_text, output_text in sample_expression_with_output(
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

        # Load text to use for unknown sentences
        config.unknown_text = sentences_yaml.get("unknown_text")

    _CONFIG_CACHE[language] = config

    return config


def sample_expression_with_output(
    expression: "Expression",
    slot_lists: "Optional[Dict[str, SlotList]]" = None,
    expansion_rules: "Optional[Dict[str, Sentence]]" = None,
) -> Iterable[Tuple[str, Optional[str]]]:
    """Sample possible text strings from an expression."""
    from hassil.expression import (
        ListReference,
        RuleReference,
        Sequence,
        SequenceType,
        TextChunk,
    )
    from hassil.intents import TextSlotList
    from hassil.recognize import MissingListError, MissingRuleError
    from hassil.util import normalize_whitespace

    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression
        yield (chunk.original_text, chunk.original_text)
    elif isinstance(expression, Sequence):
        seq: Sequence = expression
        if seq.type == SequenceType.ALTERNATIVE:
            for item in seq.items:
                yield from sample_expression_with_output(
                    item,
                    slot_lists,
                    expansion_rules,
                )
        elif seq.type == SequenceType.GROUP:
            seq_sentences = map(
                partial(
                    sample_expression_with_output,
                    slot_lists=slot_lists,
                    expansion_rules=expansion_rules,
                ),
                seq.items,
            )
            sentence_texts = itertools.product(*seq_sentences)
            for sentence_words in sentence_texts:
                yield (
                    normalize_whitespace("".join(w[0] for w in sentence_words)),
                    normalize_whitespace(
                        "".join(w[1] for w in sentence_words if w[1] is not None)
                    ),
                )
        else:
            raise ValueError(f"Unexpected sequence type: {seq}")
    elif isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression
        if (not slot_lists) or (list_ref.list_name not in slot_lists):
            raise MissingListError(f"Missing slot list {{{list_ref.list_name}}}")

        slot_list = slot_lists[list_ref.list_name]
        if isinstance(slot_list, TextSlotList):
            text_list: TextSlotList = slot_list

            if not text_list.values:
                # Not necessarily an error, but may be a surprise
                _LOGGER.warning("No values for list: %s", list_ref.list_name)

            for text_value in text_list.values:
                if text_value.value_out:
                    is_first_text = True
                    for input_text, output_text in sample_expression_with_output(
                        text_value.text_in,
                        slot_lists,
                        expansion_rules,
                    ):
                        if is_first_text:
                            output_text = (
                                text_value.value_out
                                if isinstance(text_value.value_out, str)
                                else ""
                            )
                            is_first_text = False
                        else:
                            output_text = None

                        yield (input_text, output_text)
                else:
                    yield from sample_expression_with_output(
                        text_value.text_in,
                        slot_lists,
                        expansion_rules,
                    )
        else:
            raise ValueError(f"Unexpected slot list type: {slot_list}")
    elif isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression
        if (not expansion_rules) or (rule_ref.rule_name not in expansion_rules):
            raise MissingRuleError(f"Missing expansion rule <{rule_ref.rule_name}>")

        rule_body = expansion_rules[rule_ref.rule_name]
        yield from sample_expression_with_output(
            rule_body,
            slot_lists,
            expansion_rules,
        )
    else:
        raise ValueError(f"Unexpected expression: {expression}")


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

    if (score_cutoff <= 0) or (score <= score_cutoff):
        # Map to output text
        final_text = config.sentences[fixed_text]
    else:
        final_text = text

    _LOGGER.debug(
        "score=%s/%s, original=%s, final=%s", score, score_cutoff, text, final_text
    )

    return final_text
