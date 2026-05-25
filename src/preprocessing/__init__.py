"""Phase 2 — preprocessing modules.

Submodules:
- ``tanglish_detector``: code-switching ratio + language tagging.
- ``text_cleaner``: URL/HTML strip, dedup, length filter, tech-keyword gate.
- ``qa_formatter``: Alpaca + ChatML formatting, stratified train/val/test split.
- ``language_filter``: thin wrapper around :mod:`tanglish_detector` for CLI use.
"""
