import argparse
import itertools
import logging
import re
import sqlite3
import time
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from hassil.expression import Expression, Sentence
    from hassil.intents import SlotList
    from unicode_rbnf import RbnfEngine

_LOGGER = logging.getLogger()


class MissingLimitedDependencyError(Exception):
    """Raised when an optional dependency for limited sentences is missing."""

    def __init__(self) -> None:
        super().__init__("pip3 install wyoming-vosk[limited]")


@dataclass
class LanguageConfig:
    sentences_mtime_ns: int
    sentences_file_size: int
    database_path: Path
    no_correct_patterns: List[re.Pattern] = field(default_factory=list)
    unknown_text: Optional[str] = None


@dataclass
class Substitutions:
    """Text that was actually sampled for each {list} and <rule> reference.

    Used to resolve back references in a sentence's output text, so that
    "turn on <device>" can produce "turn on the kitchen light".
    """

    lists: Dict[str, Any] = field(default_factory=dict)
    rules: Dict[str, str] = field(default_factory=dict)

    def merged(self, *others: "Substitutions") -> "Substitutions":
        """Return a copy with the values from others applied on top."""
        merged = Substitutions(dict(self.lists), dict(self.rules))
        for other in others:
            merged.lists.update(other.lists)
            merged.rules.update(other.rules)

        return merged

    def apply(self, text: str) -> str:
        """Replace {list} and <rule> back references in output text."""
        if self.lists:
            text = text.format(**self.lists)

        for rule_name, rule_text in self.rules.items():
            text = text.replace(f"<{rule_name}>", rule_text)

        return text


# language -> config
_CONFIG_CACHE: Dict[str, LanguageConfig] = {}


def load_sentences_for_language(
    sentences_dir: Union[str, Path], language: str, database_dir: Union[str, Path]
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

    try:
        import yaml
    except ImportError as exc:
        raise MissingLimitedDependencyError() from exc

    # Load and verify YAML
    _LOGGER.debug("Loading %s", sentences_path)
    with open(sentences_path, "r", encoding="utf-8") as sentences_file:
        sentences_yaml = yaml.safe_load(sentences_file)
        if not sentences_yaml:
            _LOGGER.warning("Empty YAML file: %s", sentences_path)
            return None

        if not sentences_yaml.get("sentences"):
            _LOGGER.warning("No sentences in %s", sentences_path)
            return None

    database_dir = Path(database_dir)
    database_dir.mkdir(parents=True, exist_ok=True)
    database_path = database_dir / f"{language}.db"

    # Continue loading
    config = LanguageConfig(
        sentences_mtime_ns=sentences_stats.st_mtime_ns,
        sentences_file_size=sentences_stats.st_size,
        database_path=database_path,
    )

    # Load "no correct" patterns
    no_correct_patterns = sentences_yaml.get("no_correct_patterns", [])
    for pattern_text in no_correct_patterns:
        config.no_correct_patterns.append(re.compile(pattern_text))

    # Load text to use for unknown sentences
    config.unknown_text = sentences_yaml.get("unknown_text")

    # Remove existing database
    database_path.unlink(missing_ok=True)

    # Create new database
    db_conn = sqlite3.connect(str(database_path))
    with db_conn:
        db_conn.execute(
            "CREATE TABLE sentences "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, input_text TEXT, "
            "output_text TEXT, input_sounds_like TEXT);"
        )
        db_conn.commit()
        generate_sentences_db(sentences_yaml, db_conn, get_number_engine(language))

    _CONFIG_CACHE[language] = config

    return config


def get_number_engine(language: str) -> "Optional[RbnfEngine]":
    """Get an engine for spelling out numbers, if the language is supported."""
    try:
        from unicode_rbnf import RbnfEngine
    except ImportError as exc:
        raise MissingLimitedDependencyError() from exc

    try:
        return RbnfEngine.for_language(language)
    except Exception:
        _LOGGER.debug("No number engine for language: %s", language)
        return None


def generate_sentences_db(
    sentences_yaml: Dict[str, Any],
    db_conn: sqlite3.Connection,
    number_engine: "Optional[RbnfEngine]" = None,
) -> None:
    """Write every possible sentence from the YAML templates into the database."""
    start_time = time.monotonic()

    num_sentences = 0
    for input_text, output_text in generate_sentences(sentences_yaml, number_engine):
        if not input_text:
            continue

        db_conn.execute(
            "INSERT INTO sentences (input_text, output_text, input_sounds_like) "
            "VALUES (?, ?, ?)",
            (input_text, output_text, sounds_like(input_text)),
        )
        num_sentences += 1

    db_conn.commit()
    end_time = time.monotonic()

    _LOGGER.info(
        "Generated %s sentence(s) in %0.2f second(s)",
        num_sentences,
        end_time - start_time,
    )


def sounds_like(text: str) -> str:
    """Return a phonetic representation of text for fuzzy matching."""
    try:
        from pyphonetics import Metaphone
    except ImportError as exc:
        raise MissingLimitedDependencyError() from exc

    phonetics_algorithm = Metaphone()
    codes: List[str] = []
    for word in text.split():
        try:
            codes.append(phonetics_algorithm.phonetics(word))
        except Exception:
            # Not all scripts can be encoded (Metaphone is Latin-only)
            codes.append(word)

    return "".join(codes)


def generate_sentences(
    sentences_yaml: Dict[str, Any], number_engine: "Optional[RbnfEngine]" = None
) -> Iterable[Tuple[str, str]]:
    """Generate (input text, output text) for every sentence in the YAML."""
    try:
        from hassil.expression import TextChunk
        from hassil.intents import SlotList, TextSlotList, TextSlotValue
        from hassil.parse_expression import parse_sentence
        from hassil.sample import sample_expression
        from hassil.util import is_template
    except ImportError as exc:
        raise MissingLimitedDependencyError() from exc

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
    templates = sentences_yaml["sentences"]

    # Load slot lists
    slot_lists: Dict[str, SlotList] = {}
    for slot_name, slot_info in sentences_yaml.get("lists", {}).items():
        if isinstance(slot_info, ABCSequence):
            slot_info = {"values": slot_info}

        slot_list_values: List[TextSlotValue] = []

        slot_range = slot_info.get("range")
        if slot_range:
            assert (
                number_engine is not None
            ), "Can't expand ranges without a number engine"
            slot_from = int(slot_range["from"])
            slot_to = int(slot_range["to"])
            slot_step = int(slot_range.get("step", 1))
            for number in range(slot_from, slot_to + 1, slot_step):
                # Use all available words for a number (all genders, cases, etc.)
                format_result = number_engine.format_number(number)
                number_strs = {
                    s.replace("-", " ") for s in format_result.text_by_ruleset.values()
                }
                slot_list_values.extend(
                    TextSlotValue(text_in=TextChunk(number_str), value_out=number)
                    for number_str in number_strs
                )

            slot_lists[slot_name] = TextSlotList(
                name=slot_name, values=slot_list_values
            )
            continue

        slot_values = slot_info.get("values")
        if not slot_values:
            _LOGGER.warning("No values for list %s, skipping", slot_name)
            continue

        for slot_value in slot_values:
            values_in: List[str] = []
            values_out: List[str] = []

            if isinstance(slot_value, str):
                slot_value = {"in": slot_value}

            # - in: text to say
            #   out: text to output
            value_in = str(slot_value["in"])
            if not value_in:
                # Skip slot value
                continue

            value_out = slot_value.get("out")
            value_context = slot_value.get("context")

            if is_template(value_in):
                input_expression = parse_sentence(value_in).expression
                for input_text in sample_expression(input_expression):
                    values_in.append(input_text)
                    values_out.append(value_out or input_text)
            else:
                values_in.append(value_in)
                values_out.append(value_out or value_in)

            for value_in, value_out in zip(values_in, values_out):
                slot_list_values.append(
                    TextSlotValue(
                        TextChunk(value_in), value_out=value_out, context=value_context
                    )
                )

        slot_lists[slot_name] = TextSlotList(name=slot_name, values=slot_list_values)

    # Load expansion rules
    expansion_rules: Dict[str, "Sentence"] = {}
    for rule_name, rule_text in sentences_yaml.get("expansion_rules", {}).items():
        expansion_rules[rule_name] = parse_sentence(rule_text)

    # Generate possible sentences
    for template in templates:
        requires_context: Optional[Dict[str, Any]] = None
        excludes_context: Optional[Dict[str, Any]] = None

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
            requires_context = template.get("requires_context")
            excludes_context = template.get("excludes_context")

        for input_template in input_templates:
            if not is_template(input_template):
                # Not a template
                # output_text may be empty on purpose
                yield _strip(
                    input_template,
                    input_template if output_text is None else output_text,
                )
                continue

            # Generate possible texts
            input_expression = parse_sentence(input_template).expression
            for (
                input_text,
                maybe_output_text,
                substitutions,
            ) in sample_expression_with_output(
                input_expression,
                slot_lists=slot_lists,
                expansion_rules=expansion_rules,
                requires_context=requires_context,
                excludes_context=excludes_context,
            ):
                if output_text is None:
                    # Sampled text already has lists and rules expanded
                    final_output_text = maybe_output_text or input_text
                else:
                    # May be empty.
                    # Resolve {list} and <rule> back references.
                    final_output_text = substitutions.apply(output_text)

                yield _strip(input_text, final_output_text)


def _strip(input_text: str, output_text: str) -> Tuple[str, str]:
    """Remove whitespace left over from optional template parts."""
    return (input_text.strip(), output_text.strip())


def sample_expression_with_output(
    expression: "Expression",
    slot_lists: "Optional[Dict[str, SlotList]]" = None,
    expansion_rules: "Optional[Dict[str, Sentence]]" = None,
    substitutions: Optional[Substitutions] = None,
    requires_context: Optional[Dict[str, Any]] = None,
    excludes_context: Optional[Dict[str, Any]] = None,
) -> Iterable[Tuple[str, Optional[str], Substitutions]]:
    """Sample possible text strings from an expression."""
    try:
        from hassil.errors import MissingListError, MissingRuleError
        from hassil.expression import (
            Alternative,
            Group,
            ListReference,
            Permutation,
            RuleReference,
            Sequence,
            TextChunk,
        )
        from hassil.intents import TextSlotList
        from hassil.util import (
            check_excluded_context,
            check_required_context,
            normalize_whitespace,
        )
    except ImportError as exc:
        raise MissingLimitedDependencyError() from exc

    if substitutions is None:
        substitutions = Substitutions()

    sample = partial(
        sample_expression_with_output,
        slot_lists=slot_lists,
        expansion_rules=expansion_rules,
        substitutions=substitutions,
        requires_context=requires_context,
        excludes_context=excludes_context,
    )

    if isinstance(expression, TextChunk):
        chunk: TextChunk = expression
        yield (chunk.original_text, chunk.original_text, substitutions)
    elif isinstance(expression, Group):
        grp: Group = expression
        if isinstance(grp, Alternative):
            # Only one item is used
            for item in grp.items:
                yield from sample(item)
        elif isinstance(grp, (Sequence, Permutation)):
            # Each item is sampled, then the samples are combined.
            # sampled_items = [(input_text, output_text, substitutions), ...]
            is_permutation = isinstance(grp, Permutation)
            if is_permutation:
                # Lists are needed because itertools makes multiple passes
                item_samples = [list(sample(item)) for item in grp.items]
                orderings: Iterable[Iterable[Any]] = itertools.permutations(
                    item_samples
                )
            else:
                orderings = [map(sample, grp.items)]

            for ordering in orderings:
                for sampled_items in itertools.product(*ordering):
                    item_substitutions = substitutions.merged(
                        *(item[2] for item in sampled_items)
                    )

                    input_text = normalize_whitespace(
                        "".join(i[0] for i in sampled_items)
                    )
                    output_text = normalize_whitespace(
                        "".join(str(i[1]) for i in sampled_items if i[1] is not None)
                    )

                    if is_permutation:
                        # Strip whitespace added between permuted items
                        input_text = input_text.strip()
                        output_text = output_text.strip()

                    yield (input_text, output_text, item_substitutions)
        else:
            raise ValueError(f"Unexpected group type: {grp}")
    elif isinstance(expression, ListReference):
        # {list}
        list_ref: ListReference = expression
        if (not slot_lists) or (list_ref.list_name not in slot_lists):
            raise MissingListError(f"Missing slot list {{{list_ref.list_name}}}")

        slot_list = slot_lists[list_ref.list_name]
        if not isinstance(slot_list, TextSlotList):
            # Range lists are expanded into words earlier.
            # Wildcards are not supported.
            raise ValueError(f"Unexpected slot list type: {slot_list}")

        text_list: TextSlotList = slot_list

        if requires_context or excludes_context:
            # Filtered values
            filtered_values = [
                v
                for v in text_list.values
                if (
                    (not requires_context)
                    or check_required_context(
                        requires_context, v.context, allow_missing_keys=True
                    )
                )
                and (
                    (not excludes_context)
                    or check_excluded_context(excludes_context, v.context)
                )
            ]
        else:
            filtered_values = text_list.values

        if not filtered_values:
            # Not necessarily an error, but may be a surprise
            _LOGGER.warning("No values for list: %s", list_ref.list_name)

        for text_value in filtered_values:
            for (
                value_input_text,
                value_output_text,
                value_substitutions,
            ) in sample(text_value.text_in):
                if text_value.value_out is not None:
                    value_output_text = str(text_value.value_out)

                yield (
                    value_input_text,
                    value_output_text,
                    value_substitutions.merged(
                        Substitutions(lists={list_ref.list_name: value_output_text})
                    ),
                )
    elif isinstance(expression, RuleReference):
        # <rule>
        rule_ref: RuleReference = expression
        if (not expansion_rules) or (rule_ref.rule_name not in expansion_rules):
            raise MissingRuleError(f"Missing expansion rule <{rule_ref.rule_name}>")

        for (
            rule_input_text,
            rule_output_text,
            rule_substitutions,
        ) in sample(expansion_rules[rule_ref.rule_name].expression):
            # Record the complete text sampled for this rule, so <rule> can be
            # used as a back reference in output text. Whitespace is kept as
            # sampled, so a rule like "[the ]" still spaces itself correctly.
            # Nested rules are recorded too: the inner reference is reached
            # first, so both <light> and <room> are available in "the <room>
            # light".
            yield (
                rule_input_text,
                rule_output_text,
                rule_substitutions.merged(
                    Substitutions(rules={rule_ref.rule_name: rule_output_text or ""})
                ),
            )
    else:
        raise ValueError(f"Unexpected expression: {expression}")


def correct_sentence(
    text: str, config: LanguageConfig, score_cutoff: float = 0.2
) -> str:
    """Correct a sentence using rapidfuzz."""
    if not config.database_path.is_file():
        # Can't correct without a database
        return text

    text = text.strip()
    if not text:
        # Can't correct empty text
        return text

    # Don't correct transcripts that match a "no correct" pattern
    for pattern in config.no_correct_patterns:
        if pattern.match(text):
            return text

    text_sounds_like = sounds_like(text)

    with sqlite3.connect(str(config.database_path)) as db_conn:
        try:
            from rapidfuzz.distance import Levenshtein
            from rapidfuzz.process import extractOne
        except ImportError as exc:
            raise MissingLimitedDependencyError() from exc

        cursor = db_conn.execute("SELECT input_sounds_like, output_text from sentences")
        result = extractOne(
            [text_sounds_like],  # critical that this is a list
            cursor,
            processor=lambda s: s[0],
            scorer=Levenshtein.distance,
            scorer_kwargs={"weights": (1, 1, 3)},
        )
        fixed_row, score = result[0], result[1]

        # Normalize by transcript length so the cutoff does not depend on how
        # long the sentence is.
        norm_score = score / len(text)

        final_text = text
        if norm_score < score_cutoff:
            # Map to output text
            final_text = fixed_row[1]

        _LOGGER.debug(
            "score=%s/%s, original=%s, final=%s",
            norm_score,
            score_cutoff,
            text,
            final_text,
        )

        return final_text


# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-dir", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--database-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    load_sentences_for_language(args.sentences_dir, args.language, args.database_dir)


if __name__ == "__main__":
    main()
