"""Kleine lokale Web-Oberfläche für die Demo.

Aufnahme hochladen oder Demo-Besprechung wählen, dann laufen Diarization und deutsche
Spracherkennung auf diesem Rechner. Das Ergebnis erscheint als Sprecherspuren und Transkript;
auf Wunsch schreibt das lokale Sprachmodell in LM Studio das Protokoll.

Der Server lauscht nur auf 127.0.0.1. Nichts verlässt den Rechner.

Aufruf:  python3 scripts/server.py [--port 8790]
"""

import argparse
import json
import mimetypes
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
JOBS = ROOT / "jobs"
FIXTURES = ROOT / "fixtures"
MAX_UPLOAD = 500 * 1024 * 1024
DEMOS = {
    "supertonic": "besprechung-supertonic.wav",
    "supertonic-aehnlich": "besprechung-supertonic-aehnlich.wav",
    "macos": "besprechung-macos.wav",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[server] " + fmt % args + "\n")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            return self.send_file(WEB / "index.html")
        if route == "/api/demos":
            spec = json.loads((FIXTURES / "besprechung.json").read_text())
            available = [k for k, f in DEMOS.items() if (FIXTURES / f).exists()]
            return self.send_json(200, {"demos": available, "titel": spec["titel"], "hinweis": spec["hinweis"]})
        if route.startswith("/jobs/"):
            job, _, name = route[len("/jobs/"):].partition("/")
            if job.isalnum() and name == "audio.wav":
                return self.send_file(JOBS / job / "audio.wav")
        self.send_error(404)

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        try:
            if route == "/api/analyze":
                return self.analyze()
            if route == "/api/protocol":
                return self.protocol()
        except Exception as exc:  # Fehler sichtbar an die Oberfläche geben, Details ins Log
            traceback.print_exc()
            return self.send_json(500, {"fehler": f"{type(exc).__name__}: {exc}"})
        self.send_error(404)

    def analyze(self):
        demo = self.headers.get("X-Demo")
        job = uuid.uuid4().hex[:12]
        job_dir = JOBS / job
        job_dir.mkdir(parents=True)
        if demo:
            if demo not in DEMOS:
                return self.send_json(400, {"fehler": "Unbekannte Demo"})
            source = FIXTURES / DEMOS[demo]
        else:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD:
                return self.send_json(400, {"fehler": "Datei fehlt oder ist größer als 500 MB"})
            suffix = Path(self.headers.get("X-Filename", "upload.wav")).suffix.lower()[:8] or ".wav"
            source = job_dir / f"upload{suffix}"
            source.write_bytes(self.rfile.read(length))
        result = pipeline.run(source, job_dir)
        result["job"] = job
        result["audio_url"] = f"/jobs/{job}/audio.wav"
        return self.send_json(200, result)

    def protocol(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        job = str(body.get("job", ""))
        if not job.isalnum() or not (JOBS / job / "transkript.json").exists():
            return self.send_json(400, {"fehler": "Unbekannter Auftrag"})
        result = json.loads((JOBS / job / "transkript.json").read_text())
        names = {k: str(v)[:60] for k, v in body.get("names", {}).items() if str(v).strip()}
        try:
            prot = pipeline.protocol(result, names, body.get("model") or pipeline.DEFAULT_LLM)
        except OSError as exc:
            return self.send_json(503, {"fehler": f"LM Studio nicht erreichbar ({exc}). Server und Modell laden: lms server start; lms load {pipeline.DEFAULT_LLM}"})
        prot["namen"] = names
        (JOBS / job / "protokoll.json").write_text(json.dumps(prot, ensure_ascii=False, indent=1))
        return self.send_json(200, prot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    port = ap.parse_args().port
    JOBS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Demo läuft auf http://127.0.0.1:{port}  (nur lokal, Strg+C beendet)")
    server.serve_forever()


if __name__ == "__main__":
    main()
