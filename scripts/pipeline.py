"""Wer hat was gesagt: Diarization, deutsche Spracherkennung und Protokoll, alles lokal.

1. Nemotron 3 Diarization (audio.cpp, Metal) findet, wer wann spricht.
2. Qwen3-ASR 0.6B (audio.cpp) transkribiert jeden erkannten Abschnitt auf Deutsch.
3. Stellen sich alle zu Beginn vor („Ich bin Petra Lang, Produktionsleitung“), werden die Namen
   aus dem Transkript gelesen und den Stimmen zugeordnet. Kein Stimmprofil, keine Biometrie.
4. Optional: ein lokales Sprachmodell in LM Studio schreibt daraus das Protokoll
   mit Aufgaben, Verantwortlichen und Terminen, jeweils mit Beleg aus dem Transkript.

Aufruf:
  python scripts/pipeline.py fixtures/besprechung-supertonic.wav --out results/lauf-supertonic
  python scripts/pipeline.py <audio> --out <ordner> --names '{"speaker_0": "Petra"}' --protocol
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "runtime" / "audiocpp_cli"
DIAR_MODEL = ROOT / "models" / "Nemotron-3-Diarization-GGUF" / "nemotron-3-diarization-bf16.gguf"
ASR_MODEL = ROOT / "models" / "Qwen3-ASR-0.6B-GGUF" / "qwen3-asr-0.6b-q8_0.gguf"
LMSTUDIO_URL = "http://127.0.0.1:1234/api/v1/chat"
DEFAULT_LLM = "qwen/qwen3.8-27b"
SR = 16000
MERGE_GAP_S = 0.6
PAD_S = 0.1


def to_wav16k(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def diarize(wav: Path, out_dir: Path) -> list[dict]:
    turns_path = out_dir / "turns.json"
    subprocess.run(
        [str(CLI), "--task", "diar", "--family", "nemotron_3_diar", "--model", str(DIAR_MODEL),
         "--backend", "metal", "--audio", str(wav), "--turns-out", str(turns_path)],
        check=True, capture_output=True,
    )
    return [
        {"speaker": t["speaker_id"], "start": t["start_sample"] / SR, "end": t["end_sample"] / SR,
         "confidence": t.get("confidence")}
        for t in json.loads(turns_path.read_text())
    ]


def merge_turns(turns: list[dict]) -> list[dict]:
    """Aufeinanderfolgende Abschnitte derselben Person zusammenlegen, damit die ASR ganze Sätze hört."""
    merged: list[dict] = []
    for t in sorted(turns, key=lambda t: t["start"]):
        last = merged[-1] if merged else None
        if last and last["speaker"] == t["speaker"] and t["start"] - last["end"] <= MERGE_GAP_S:
            last["end"] = max(last["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def transcribe(wav: Path, segments: list[dict]) -> None:
    """Jeden Abschnitt ausschneiden und in einer einzigen ASR-Sitzung transkribieren."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, seg in enumerate(segments):
            start = max(0.0, seg["start"] - PAD_S)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
                 "-t", f"{seg['end'] + PAD_S - start:.3f}", "-i", str(wav), str(Path(tmp) / f"{i:04d}.wav")],
                check=True,
            )
        proc = subprocess.run(
            [str(CLI), "--task", "asr", "--family", "qwen3_asr", "--model", str(ASR_MODEL),
             "--language", "de", "--batch-audio-dir", tmp, "--out-dir", str(Path(tmp) / "out")],
            check=True, capture_output=True, text=True,
        )
    texts: dict[int, str] = {}
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith("request_id="):
            current = int(line.split("=", 1)[1])
        elif line.startswith("text_output=") and current is not None:
            texts[current] = line.split("=", 1)[1].strip()
    for i, seg in enumerate(segments):
        seg["text"] = texts.get(i, "")


INTRO_WINDOW_S = 60.0
_NAME = r"([A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?\s+(?:(?:von|van|de|zu)\s+)?[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?)"
# Auslöser ohne Rücksicht auf Groß-/Kleinschreibung, der Name selbst muss großgeschrieben sein.
_INTRO_TRIGGER = re.compile(
    r"\b(?i:ich bin|ich heiße|mein name ist|hier ist|hier spricht|am apparat ist)\s+" + _NAME + r"(.*)"
)
# „Jonas Weber, Einkauf.“ oder ohne Komma „Jonas Weber Einkauf“: nur als erster Satz des ersten
# Beitrags einer Stimme, sonst läse sich „Zwölf Teile, das ist viel.“ wie ein Name.
_INTRO_BARE = re.compile(r"^(?:[Uu]nd\s+)?" + _NAME + r"(?:[\s,]+(.*))?$")
# Großgeschriebene Satzanfänge, die sonst wie ein Vorname aussehen („Zu Brenner, …“).
_NOT_A_FIRST_NAME = {
    "Zu", "Und", "Dann", "Gut", "Nein", "Ja", "Moment", "Bei", "Ich", "Wir", "Das", "Die", "Der", "Den",
    "Letzter", "Soll", "Kurze", "Kann", "Guten", "Hallo", "Danke", "Bis", "Also", "Okay", "Sag", "Mit",
}
# Rollen statt Namen („Ich bin Teamleiterin Vertrieb“). Bewusst eine Wortliste statt Endungen,
# damit Namen wie Dominik, Marion oder Jung nicht herausfallen.
_ROLE_WORDS = {
    "vertrieb", "einkauf", "qualität", "produktion", "buchhaltung", "marketing", "personal", "logistik",
    "service", "technik", "controlling", "lager", "werkstatt", "entwicklung", "konstruktion", "versand",
    "geschäftsführer", "geschäftsführerin", "geschäftsführung", "leitung", "leiter", "leiterin",
}


def _looks_like_role(name: str) -> bool:
    return any(t.lower() in _ROLE_WORDS or t.lower().endswith(("leiter", "leiterin", "leitung")) for t in name.split())


def detect_names(segments: list[dict]) -> dict:
    """Namen aus einer Vorstellungsrunde am Anfang lesen („Ich bin Petra Lang, Produktionsleitung“).

    Der Name stammt aus dem, was die Person selbst sagt, nicht aus einem Stimmprofil. Nennt eine
    Stimmspur zwei verschiedene Namen, wurden vermutlich zwei Personen zusammengelegt.
    """
    found: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for seg in segments:
        if seg["start"] > INTRO_WINDOW_S:
            break
        first_turn = seg["speaker"] not in seen
        seen.add(seg["speaker"])
        sentences = [x.strip().strip("„“\"") for x in re.split(r"(?<=[.!?])\s+", seg["text"]) if x.strip()]
        for i, sentence in enumerate(sentences):
            m = _INTRO_TRIGGER.search(sentence) or (_INTRO_BARE.match(sentence) if first_turn and i == 0 else None)
            if not m or m.group(1).split()[0] in _NOT_A_FIRST_NAME or _looks_like_role(m.group(1)):
                continue
            name = m.group(1)
            role = re.sub(r"^[\s,]+|[\s.]+$", "", m.group(2) or "")[:60]
            entries = found.setdefault(seg["speaker"], [])
            if name not in {e["name"] for e in entries}:
                entries.append({"name": name, "rolle": role, "zeit": round(seg["start"], 1), "zitat": sentence})
    names = {sp: entries[0] for sp, entries in found.items()}
    warnings = [
        f"Eine Stimmspur ({sp}) nennt {len(e)} Namen: {', '.join(x['name'] for x in e)}. "
        "Vermutlich wurden Stimmen zusammengelegt; diese Zuordnung bitte prüfen."
        for sp, e in found.items() if len(e) > 1
    ]
    return {"namen": names, "alle_treffer": found, "warnungen": warnings}


def run(audio: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    to_wav16k(audio, wav)

    t0 = time.perf_counter()
    turns = diarize(wav, out_dir)
    t1 = time.perf_counter()
    segments = merge_turns(turns)
    transcribe(wav, segments)
    t2 = time.perf_counter()

    duration = int(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=duration_ts", "-of", "csv=p=0", str(wav)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()) / SR
    speakers = sorted({s["speaker"] for s in segments}, key=lambda s: int(s.rsplit("_", 1)[-1]))
    stats = {
        sp: {
            "beitraege": sum(1 for s in segments if s["speaker"] == sp),
            "sekunden": round(sum(s["end"] - s["start"] for s in segments if s["speaker"] == sp), 1),
            "woerter": sum(len(s["text"].split()) for s in segments if s["speaker"] == sp),
        }
        for sp in speakers
    }
    result = {
        "quelle": audio.name,
        "dauer_s": round(duration, 2),
        "sprecher": speakers,
        "segmente": [{**s, "start": round(s["start"], 2), "end": round(s["end"], 2)} for s in segments],
        "statistik": stats,
        "vorstellung": detect_names(segments),
        "zeiten_s": {
            "diarization_inkl_laden": round(t1 - t0, 2),
            "spracherkennung_inkl_laden": round(t2 - t1, 2),
        },
        "modelle": {
            "diarization": "nvidia/Nemotron-3-Diarization (audio.cpp GGUF bf16)",
            "asr": "Qwen3-ASR-0.6B (GGUF q8_0)",
            "laufzeit": "audio.cpp 0.8.2, Metal",
        },
    }
    (out_dir / "transkript.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def transcript_text(result: dict, names: dict[str, str]) -> str:
    def stamp(s: float) -> str:
        return f"{int(s // 60):02d}:{s % 60:04.1f}"

    return "\n".join(
        f"[{stamp(s['start'])}] {names.get(s['speaker'], s['speaker'])}: {s['text']}" for s in result["segmente"]
    )


PROTOCOL_SYSTEM = """Du erstellst aus einem automatisch erzeugten Besprechungstranskript ein knappes deutsches Protokoll.
Regeln:
- Nimm nur Aufgaben auf, die im Transkript ausdrücklich zugesagt oder zugewiesen werden.
- Wenn eine Zusage später zurückgenommen, verschoben oder an jemand anderen übergeben wird, gilt nur der letzte Stand.
- Was ausdrücklich nicht getan werden soll, ist keine Aufgabe.
- Erfinde keine Namen, Termine oder Inhalte. Unklares kommt unter offene_fragen.
- Jede Aufgabe braucht ein wörtliches Belegzitat und den Zeitstempel der Zeile, aus der sie stammt.
Antworte nur mit JSON in dieser Form:
{"themen": ["..."], "aufgaben": [{"wer": "...", "was": "...", "bis": "...", "beleg": "...", "zeit": "mm:ss.s"}], "offene_fragen": ["..."]}"""


def protocol(result: dict, names: dict[str, str], model: str = DEFAULT_LLM) -> dict:
    request = {
        "model": model,
        "system_prompt": PROTOCOL_SYSTEM,
        "input": transcript_text(result, names),
        "temperature": 0,
        "max_output_tokens": 3000,
        "reasoning": "off",
        "store": False,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        LMSTUDIO_URL, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        answer = json.load(response)
    content = "\n".join(x["content"] for x in answer["output"] if x["type"] == "message")
    parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip()))
    return {
        "modell": model,
        "sekunden": round(time.perf_counter() - t0, 1),
        "roh": content,
        "protokoll": parsed,
        "eingabe": request["input"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--names", default="{}", help='JSON, z. B. {"speaker_0": "Petra"}')
    ap.add_argument("--protocol", action="store_true")
    ap.add_argument("--model", default=DEFAULT_LLM)
    args = ap.parse_args()

    result = run(args.audio, args.out)
    names = {sp: v["name"] for sp, v in result["vorstellung"]["namen"].items()}
    names.update(json.loads(args.names))
    for w in result["vorstellung"]["warnungen"]:
        print("WARNUNG:", w)
    print(transcript_text(result, names))
    print(json.dumps(result["zeiten_s"]))
    if args.protocol:
        prot = protocol(result, names, args.model)
        (args.out / "protokoll.json").write_text(json.dumps(prot, ensure_ascii=False, indent=1))
        print(json.dumps(prot["protokoll"], ensure_ascii=False, indent=1))
        print(f"Protokoll in {prot['sekunden']} s")


if __name__ == "__main__":
    main()
