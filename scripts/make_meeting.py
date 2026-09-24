"""Synthetische deutsche Besprechung aus fixtures/besprechung.json bauen.

Jeder Beitrag wird mit einer macOS-Stimme gesprochen und auf eine gemeinsame Zeitachse
gelegt. Ein negativer `versatz_s` lässt den Beitrag vor dem Ende des vorherigen beginnen,
so entstehen echte Überlappungen. Weil wir die Spuren selbst mischen, ist die Soll-Zuordnung
(wer spricht wann) exakt bekannt und wird vor jedem Modelllauf als RTTM festgeschrieben.

Zwei Stimmsätze, beide synthetisch:
  macos       macOS-Systemstimmen (Anna, Reed, Shelley, Eddy); die beiden Männerstimmen
              stammen aus derselben Synthese und klingen sehr ähnlich (Stresstest)
  supertonic  Supertonic 3 über audio.cpp, neuronale Stimmen F1, M2, F3, M4

Aufruf:  uv run --with numpy python scripts/make_meeting.py --engine macos|supertonic
"""

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
SR = 16000
TRIM_DB = -45.0


def speak(text: str, voice: str, out: Path, engine: str) -> np.ndarray:
    if engine == "macos":
        subprocess.run(
            ["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", "--file-format=WAVE", text],
            check=True,
        )
    else:
        raw = out.with_suffix(".44k.wav")
        subprocess.run(
            [str(ROOT / "runtime" / "audiocpp_cli"), "--task", "tts", "--family", "supertonic",
             "--model", str(ROOT / "models" / "Supertonic-3-GGUF" / "supertonic-3-orig.gguf"),
             "--backend", "metal", "--language", "de", "--voice-id", voice, "--text", text, "--out", str(raw)],
            check=True, capture_output=True,
        )
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", str(out)], check=True)
    with wave.open(str(out)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return trim(pcm)


def trim(pcm: np.ndarray) -> np.ndarray:
    """Stille am Anfang und Ende entfernen, damit die Soll-Grenzen der hörbaren Sprache entsprechen."""
    frame = SR // 100
    n = len(pcm) // frame
    rms = np.sqrt(np.mean(pcm[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    voiced = np.where(20 * np.log10(rms) > TRIM_DB)[0]
    if len(voiced) == 0:
        raise ValueError("Beitrag ohne hörbare Sprache")
    return pcm[voiced[0] * frame : (voiced[-1] + 1) * frame]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["macos", "supertonic"], required=True)
    engine = ap.parse_args().engine
    name = f"besprechung-{engine}"
    spec = json.loads((FIXTURES / "besprechung.json").read_text())
    speakers = spec["sprecher"]
    gap = spec["abstand_standard_s"]

    placed = []
    cursor = 0.5
    with tempfile.TemporaryDirectory() as tmp:
        for item in spec["beitraege"]:
            voice = speakers[item["sprecher"]][f"stimme_{engine}"]
            pcm = speak(item["text"], voice, Path(tmp) / f"{item['id']}.wav", engine)
            start = cursor + item.get("versatz_s", gap) if placed else cursor
            start = max(start, placed[-1]["start"] + 0.3) if placed else start
            placed.append({**item, "start": round(start, 3), "end": round(start + len(pcm) / SR, 3), "pcm": pcm})
            cursor = max(cursor, placed[-1]["end"])

    total = int((cursor + 0.8) * SR)
    mix = np.zeros(total, dtype=np.float32)
    for p in placed:
        s = int(p["start"] * SR)
        mix[s : s + len(p["pcm"])] += p["pcm"] * 0.7
    mix /= max(1.0, float(np.max(np.abs(mix))) / 0.95)

    with wave.open(str(FIXTURES / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    rttm = [
        f"SPEAKER {name} 1 {p['start']:.3f} {p['end'] - p['start']:.3f} <NA> <NA> {p['sprecher']} <NA> <NA>"
        for p in placed
    ]
    (FIXTURES / f"{name}.rttm").write_text("\n".join(rttm) + "\n")

    overlap = sum(
        max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        for i, a in enumerate(placed)
        for b in placed[i + 1 :]
        if a["sprecher"] != b["sprecher"]
    )
    soll = {
        "engine": engine,
        "audio": f"fixtures/{name}.wav",
        "dauer_s": round(total / SR, 2),
        "ueberlappung_s": round(overlap, 2),
        "segmente": [{k: p[k] for k in ("id", "sprecher", "start", "end", "text")} for p in placed],
    }
    (FIXTURES / f"{name}.soll.json").write_text(json.dumps(soll, ensure_ascii=False, indent=1))
    print(f"{name}.wav: {soll['dauer_s']} s, {len(placed)} Beiträge, Überlappung {soll['ueberlappung_s']} s")


if __name__ == "__main__":
    main()
