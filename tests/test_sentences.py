"""Tests for sentence template generation."""

from typing import Any, Dict, List, Tuple

import yaml

from wyoming_vosk.sentences import generate_sentences


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
