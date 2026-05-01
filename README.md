# EventGPT Scouting Engine — Streamlit app

A scouting + transfer-reasoning companion for football clubs, built on top
of the [eventGPT](../eventGPT) event-level language model. The model lives
on Modal; this repo is just the UI + LLM orchestration.

## Run locally

1. Deploy the Modal service from the eventGPT repo:
   ```sh
   cd ../eventGPT
   modal deploy src/eventgpt/web/modal_endpoints.py
   # copy the printed URL — looks like https://<workspace>--eventgpt-web-webapi.modal.run
   ```
2. Set up env:
   ```sh
   cp .env.example .env
   # edit .env: paste MODAL_URL, paste OPENAI_API_KEY
   ```
3. Install + run:
   ```sh
   python3.11 -m venv .venv
   .venv/bin/pip install -e .
   .venv/bin/streamlit run app/streamlit_app.py
   ```

## What it does

Two modes, single-page app:

- **Scout** — search any player, see their style fingerprint (action mix +
  pitch heatmap + archetype), find similar players under filters, run a
  what-if swap (replace X with Y, see expected impact), generate an AI
  scouting one-pager.
- **Strategy** — pick a target club, score a candidate's fit to that club,
  browse the data-driven archetype taxonomy on a 2D map, generate an AI
  board memo for a shortlist.

All numerical signals are translated into plain English first
(`app/services/explainer.py`); the OpenAI calls receive the *already-
explained* context so the prose stays football-native, not ML-native.
