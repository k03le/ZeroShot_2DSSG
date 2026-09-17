"""
Script: vlm_server.py
Stage:  8A-server — VLM HTTP Server
Purpose: Wraps Qwen2.5-VL-7B in a minimal stdlib HTTP server.
         wasti-new sends crop images (base64 JSON), receives text descriptions.
Input:   POST /describe  {"image_b64": "...", "label": "..."}
Output:  {"description": "..."}
Usage:   python vlm_server.py [--port 8765] [--host 0.0.0.0]

Run on:  wasti-old (Port 6180) | conda activate mmc
Keep running while describe_objects.py runs on wasti-new.
Stop with Ctrl+C when done.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

sys.path.insert(0, "/data/mmc-user/projects/src")
from vlm import VLM  # noqa: E402

# ─── Prompt ───────────────────────────────────────────────────────────────────
DESCRIBE_PROMPT = """\
You are annotating a detected object for an indoor 3D scene graph.
The object category is: "{label}"

Describe this object in 1–2 sentences.
Focus on: color, material, shape, and any distinctive features visible in the image.
Do not repeat the category label. Output only the description, no preamble."""


# ─── Request handler ──────────────────────────────────────────────────────────
class VLMHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ("/describe", "/ask"):
            self._respond(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))

            # Both endpoints require an image (VLM.ask() signature).
            image_bytes = base64.b64decode(data["image_b64"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if self.path == "/describe":
                prompt = DESCRIBE_PROMPT.format(label=data["label"])
            else:  # /ask — caller supplies the complete prompt
                prompt = data["prompt"]

            response = VLM.ask(image, prompt)
            # "description" kept for backward compat with describe_objects.py;
            # "response" is the canonical field for /ask callers.
            self._respond(200, {"description": response, "response": response})

        except KeyError as e:
            self._respond(400, {"error": f"Missing field: {e}"})
        except Exception as e:
            logging.error(f"Inference error: {e}", exc_info=True)
            self._respond(500, {"error": str(e)})

    def _respond(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # Route HTTP access log through Python logging
        logging.info("HTTP %s %s %s", *args)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM HTTP server — wraps Qwen2.5-VL-7B for Stage 8A."
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to listen on (default: 8765)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Pre-warm model before accepting requests — avoids 24s timeout on first call
    logging.info("Pre-warming Qwen2.5-VL-7B (this takes ~24s) ...")
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype="uint8"))
    VLM.ask(dummy, "test")
    logging.info("Model ready.")

    server = HTTPServer((args.host, args.port), VLMHandler)
    logging.info(f"Listening on {args.host}:{args.port} — ready for requests.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Ctrl+C received — shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()