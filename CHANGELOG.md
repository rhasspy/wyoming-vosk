# Changelog

## 1.6.0

- Limited mode restricts vosk to whole sentences instead of individual words
- Match transcripts to templates phonetically (Metaphone) instead of by spelling
- `--correct-sentences` cutoff is now normalized by transcript length and defaults to 0.2 (0 disables correction)
- Sentence templates support number ranges, `requires_context`/`excludes_context`, and `{list}` references in `out`
- Move packaging from `setup.py`/`requirements*.txt` to `pyproject.toml`
- Bump wyoming to 1.10.2, hassil to 3.12.1, rapidfuzz to 3.14.6
- Fix crash on audio that needs resampling

## 1.5.0

- Restore Arabic and Ukrainian models

## 1.4.0

- Add tests and Github actions
- Bump wyoming to 1.5.2

## 1.3.0

- Public release
