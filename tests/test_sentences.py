"""Tests for sentence template generation."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from wyoming_vosk.sentences import (
    _CONFIG_CACHE,
    LanguageConfig,
    correct_sentence,
    generate_sentences,
    load_sentences_for_language,
)


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Keep the language cache from leaking between tests.

    It is keyed on language alone, so two tests writing the same en.yaml could
    otherwise share a config.
    """
    _CONFIG_CACHE.clear()
    yield
    _CONFIG_CACHE.clear()


def generate(sentences_yaml: str) -> List[Tuple[str, str]]:
    """Generate (input, output) pairs from a YAML string."""
    parsed: Dict[str, Any] = yaml.safe_load(sentences_yaml)
    return list(generate_sentences(parsed))


def test_list_back_reference() -> None:
    """A {list} in output text is replaced by the value that was matched."""
    assert (
        generate("""
        sentences:
          - in: turn on [the] {device}
            out: "ON:{device}"
        lists:
          device:
            values:
              - in: tv
                out: living room tv
        """)
        == [
            ("turn on the tv", "ON:living room tv"),
            ("turn on tv", "ON:living room tv"),
        ]
    )


def test_expansion_rule_back_reference() -> None:
    """An <expansion rule> in output text is replaced by the text sampled for it.

    This is the example from issue #4.
    """
    assert (
        generate("""
        sentences:
          - in: dim <light>
            out: set <light> brightness to 25
        expansion_rules:
          light: "(livingroom light|kitchen light)"
        """)
        == [
            ("dim livingroom light", "set livingroom light brightness to 25"),
            ("dim kitchen light", "set kitchen light brightness to 25"),
        ]
    )


def test_expansion_rule_with_multiple_chunks() -> None:
    """A rule made of more than one text chunk is captured in full."""
    assert (
        generate("""
        sentences:
          - in: dim <light>
            out: "DIM:<light>"
        expansion_rules:
          light: "the (kitchen|living room) light"
        """)
        == [
            ("dim the kitchen light", "DIM:the kitchen light"),
            ("dim the living room light", "DIM:the living room light"),
        ]
    )


def test_nested_expansion_rules() -> None:
    """Both the outer and inner rule are available as back references."""
    assert (
        generate("""
        sentences:
          - in: dim <light>
            out: "<light>|<room>"
        expansion_rules:
          light: "the <room> light"
          room: "(kitchen|office)"
        """)
        == [
            ("dim the kitchen light", "the kitchen light|kitchen"),
            ("dim the office light", "the office light|office"),
        ]
    )


def test_expansion_rule_containing_list() -> None:
    """A rule that references a list captures the list's output value."""
    assert generate("""
        sentences:
          - in: turn on <device>
            out: "ON:<device>"
        expansion_rules:
          device: "{name}"
        lists:
          name:
            values:
              - in: tv
                out: living room tv
        """) == [("turn on tv", "ON:living room tv")]


def test_back_reference_used_twice() -> None:
    """Every occurrence of a back reference is replaced, not just the first."""
    assert (
        generate("""
        sentences:
          - in: dim <light>
            out: <light> and <light>
        expansion_rules:
          light: "(kitchen|office) light"
        """)
        == [
            ("dim kitchen light", "kitchen light and kitchen light"),
            ("dim office light", "office light and office light"),
        ]
    )


def test_optional_expansion_rule() -> None:
    """An unmatched optional part leaves no stray whitespace behind."""
    assert generate("""
        sentences:
          - in: turn on <the> light
            out: "ON:<the>light"
        expansion_rules:
          the: "[the ]"
        """) == [("turn on the light", "ON:the light"), ("turn on light", "ON:light")]


def test_no_back_references() -> None:
    """Output text without back references is used as-is."""
    assert (
        generate("""
        sentences:
          - plain sentence
          - in: lou mo ss
            out: turn on all the lights
          - in: nevermind
            out: ""
        """)
        == [
            ("plain sentence", "plain sentence"),
            ("lou mo ss", "turn on all the lights"),
            ("nevermind", ""),
        ]
    )


def build_config(tmp_path: Path, sentences_yaml: str) -> LanguageConfig:
    """Write a sentences file and build its database."""
    (tmp_path / "en.yaml").write_text(sentences_yaml, encoding="utf-8")
    config = load_sentences_for_language(tmp_path, "en", tmp_path)
    assert config is not None
    return config


_CORRECTION_SENTENCES = """
sentences:
  - turn on the kitchen light
  - set the lamp to red
no_correct_patterns:
  - "^draw me .*"
"""


def test_cutoff_corrects_near_misses_only(tmp_path: Path) -> None:
    """A higher cutoff corrects more, which is the opposite of the old docs."""
    config = build_config(tmp_path, _CORRECTION_SENTENCES)

    # Sounds close to a template
    assert (
        correct_sentence("turn on the kichen lite", config, score_cutoff=0.2)
        == "turn on the kitchen light"
    )

    # Nothing like a template
    assert (
        correct_sentence("play some jazz music please", config, score_cutoff=0.2)
        == "play some jazz music please"
    )


def test_cutoff_of_zero_never_corrects(tmp_path: Path) -> None:
    """0 disables correction. In 1.5.0 it meant the opposite: always correct."""
    config = build_config(tmp_path, _CORRECTION_SENTENCES)
    assert (
        correct_sentence("turn on the kichen lite", config, score_cutoff=0)
        == "turn on the kichen lite"
    )


def test_always_correct_ignores_cutoff(tmp_path: Path) -> None:
    """Limited mode forces a template even for a transcript that is far away."""
    config = build_config(tmp_path, _CORRECTION_SENTENCES)
    assert correct_sentence(
        "play some jazz music please",
        config,
        score_cutoff=0,
        always_correct=True,
    ) in ("turn on the kitchen light", "set the lamp to red")


def test_no_correct_pattern_beats_always_correct(tmp_path: Path) -> None:
    """An explicit no-correct pattern still passes text through untouched."""
    config = build_config(tmp_path, _CORRECTION_SENTENCES)
    assert (
        correct_sentence("draw me a picture of a cat", config, always_correct=True)
        == "draw me a picture of a cat"
    )


def test_empty_transcript_is_not_forced(tmp_path: Path) -> None:
    """An empty transcript stays empty, even in limited mode."""
    config = build_config(tmp_path, _CORRECTION_SENTENCES)
    assert correct_sentence("", config, always_correct=True) == ""
