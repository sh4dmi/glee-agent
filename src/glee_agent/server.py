"""HTTP server implementing GLEE's player interface: POST /chat and GET /health.

    python -m glee_agent.server --port 5001
    python -m glee_agent.server --port 5001 --key MY_GLEE_KEY   # require X-API-Key

The key can also come from the GLEE_KEY environment variable (or a .env file).
"""
from __future__ import annotations

import argparse
import hmac
import logging
import os

from flask import Flask, jsonify, request

from .graph import Agent


def create_app(agent: Agent | None = None, api_key: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["AGENT"] = agent or Agent()
    app.config["GLEE_KEY"] = api_key if api_key is not None else os.getenv("GLEE_KEY")

    def authorised() -> bool:
        expected = app.config["GLEE_KEY"]
        return not expected or hmac.compare_digest(request.headers.get("X-API-Key", ""), expected)

    @app.post("/chat")
    def chat():
        if not authorised():
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return jsonify({"error": "expected JSON with a 'messages' list"}), 400
        return jsonify({"response": app.config["AGENT"].respond(payload)})

    @app.get("/health")
    def health():
        key = request.headers.get("X-API-Key")
        if key is not None and app.config["GLEE_KEY"] and not authorised():
            return jsonify({"status": "alive", "auth": "failed"}), 401
        body = {"status": "alive", "llm": app.config["AGENT"].llm.enabled}
        if key is not None and app.config["GLEE_KEY"]:
            body["auth"] = "ok"
        return jsonify(body), 200

    return app


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="GLEE agent server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--key", default=None, help="require this X-API-Key (default: $GLEE_KEY)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    app = create_app(api_key=args.key)
    if not app.config["GLEE_KEY"]:
        logging.warning("no GLEE_KEY set: the server accepts requests without authentication")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
