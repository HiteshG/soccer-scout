"""Web-app surface for eventGPT.

`web/lib/*` are pure-Python helpers (take a pre-loaded ``Assets`` and return
JSON-able dicts). They back both the Modal HTTP endpoints in
``modal_endpoints.py`` and the typer CLI scripts under ``probes/`` and
``cases/`` (which now thin-wrap these helpers).
"""
