"""audio.cpp und die Modelle laden, jede Datei gegen ihre SHA256-Prüfsumme prüfen.

Aufruf:  python3 scripts/install.py [--mit-tts]
Ohne --mit-tts wird die Sprachsynthese (nur zum Neuerzeugen der Testbesprechung) übersprungen.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch(url: str, path: Path, sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Lade {path.name} …", flush=True)
        urllib.request.urlretrieve(url, path)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != sha256:
        path.unlink()
        raise RuntimeError(f"Prüfsumme stimmt nicht: {path.name}. Datei gelöscht, bitte erneut starten.")
    print(f"Geprüft: {path.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-tts", action="store_true")
    with_tts = ap.parse_args().mit_tts
    spec = json.loads((ROOT / "docs" / "downloads.json").read_text())

    ac = spec["audio_cpp"]
    archive = ROOT / "downloads" / ac["url"].rsplit("/", 1)[1]
    fetch(ac["url"], archive, ac["sha256"])
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(runtime, filter="data")
    # Der sichere Entpackfilter entfernt Ausführungsrechte; für die Programme wiederherstellen.
    for name in ("audiocpp_cli", "audiocpp_server", "audiocpp_gguf"):
        p = runtime / name
        p.chmod(p.stat().st_mode | 0o111)

    for m in spec["modelle"]:
        if m.get("optional") and not with_tts:
            continue
        fetch(m["url"], ROOT / "models" / m["path"], m["sha256"])
    print("Fertig. Start: python3 scripts/server.py")


if __name__ == "__main__":
    main()
