"""Diarization-Fehlerrate (DER) gegen die festgeschriebene Soll-Zuordnung.

DER = (verpasste Sprache + Fehlalarm + Sprecherverwechslung) / gesamte Soll-Sprechzeit,
gerechnet auf 10-ms-Frames. Die Modell-Labels (speaker_0 …) werden per bester Zuordnung
auf die Soll-Personen abgebildet, wie in der Standardauswertung üblich. Zusätzlich:
wie viele Beiträge der richtigen Person zugeordnet werden (Mehrheit der Frames).

Aufruf:  uv run --with numpy python eval/der.py fixtures/besprechung-macos.soll.json results/<lauf>/turns.json [--collar 0.25]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STEP = 0.01
SR = 16000


def load_reference(path: Path):
    soll = json.loads(path.read_text())
    return soll, sorted({s["sprecher"] for s in soll["segmente"]})


def load_hypothesis(path: Path):
    turns = json.loads(path.read_text())
    return [
        {"speaker": t["speaker_id"], "start": t["start_sample"] / SR, "end": t["end_sample"] / SR}
        for t in turns
    ]


def activity(segments, labels, key, n_frames):
    mat = np.zeros((len(labels), n_frames), dtype=bool)
    for s in segments:
        a, b = int(round(s["start"] / STEP)), int(round(s["end"] / STEP))
        mat[labels.index(s[key]), a:b] = True
    return mat


def collar_mask(ref_segments, n_frames, collar):
    """Frames um jede Soll-Grenze herum werden nicht gewertet (Standard-Toleranz)."""
    mask = np.ones(n_frames, dtype=bool)
    if collar <= 0:
        return mask
    c = int(round(collar / STEP))
    for s in ref_segments:
        for edge in (s["start"], s["end"]):
            e = int(round(edge / STEP))
            mask[max(0, e - c) : e + c] = False
    return mask


def score(ref, hyp, mask):
    ref_n = ref.sum(0)
    hyp_n = hyp.sum(0)
    correct = np.minimum(ref, hyp).sum(0)
    total = ref_n[mask].sum()
    missed = np.maximum(0, ref_n - hyp_n)[mask].sum()
    false_alarm = np.maximum(0, hyp_n - ref_n)[mask].sum()
    confusion = (np.minimum(ref_n, hyp_n) - correct)[mask].sum()
    return {
        "der": (missed + false_alarm + confusion) / total,
        "verpasst": missed / total,
        "fehlalarm": false_alarm / total,
        "verwechslung": confusion / total,
        "soll_sekunden": total * STEP,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("soll", type=Path)
    ap.add_argument("turns", type=Path)
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    soll, ref_labels = load_reference(args.soll)
    hyp_segments = load_hypothesis(args.turns)
    hyp_labels = sorted({s["speaker"] for s in hyp_segments})
    n = int(soll["dauer_s"] / STEP) + 1

    ref = activity(soll["segmente"], ref_labels, "sprecher", n)
    hyp_raw = activity(hyp_segments, hyp_labels, "speaker", n)

    # Beste Abbildung Modell-Label -> Person (bei höchstens 8 Sprechern per Durchprobieren machbar).
    best = None
    width = max(len(ref_labels), len(hyp_labels))
    for perm in itertools.permutations(range(width), len(hyp_labels)):
        overlap = sum((hyp_raw[i] & ref[j]).sum() for i, j in enumerate(perm) if j < len(ref_labels))
        if best is None or overlap > best[0]:
            best = (overlap, perm)
    mapping = {hyp_labels[i]: (ref_labels[j] if j < len(ref_labels) else None) for i, j in enumerate(best[1])}

    hyp = np.zeros_like(ref)
    for i, label in enumerate(hyp_labels):
        target = mapping[label]
        if target is not None:
            hyp[ref_labels.index(target)] |= hyp_raw[i]
        else:
            hyp = np.vstack([hyp, hyp_raw[i : i + 1]])
            ref = np.vstack([ref, np.zeros((1, n), dtype=bool)])

    mask = collar_mask(soll["segmente"], n, args.collar)
    ueberlappt = ref.sum(0) >= 2
    result = {
        "turns": str(args.turns),
        "collar_s": args.collar,
        "zuordnung_modell_zu_person": mapping,
        "modell_sprecher": len(hyp_labels),
        "soll_sprecher": len(ref_labels),
        "gesamt": score(ref, hyp, mask),
        "nur_ueberlappung": score(ref, hyp, mask & ueberlappt),
        "ohne_ueberlappung": score(ref, hyp, mask & ~ueberlappt),
    }

    beitraege = []
    for seg in soll["segmente"]:
        a, b = int(seg["start"] / STEP), int(seg["end"] / STEP)
        votes = {label: int(hyp_raw[i, a:b].sum()) for i, label in enumerate(hyp_labels)}
        top = max(votes, key=votes.get) if any(votes.values()) else None
        erkannt = mapping.get(top) if top else None
        beitraege.append({"id": seg["id"], "soll": seg["sprecher"], "erkannt": erkannt, "richtig": erkannt == seg["sprecher"]})
    result["beitraege_richtig"] = sum(b["richtig"] for b in beitraege)
    result["beitraege_gesamt"] = len(beitraege)
    result["beitraege"] = beitraege

    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text)
    g = result["gesamt"]
    print(
        f"DER {g['der']:.2%} (verpasst {g['verpasst']:.2%}, Fehlalarm {g['fehlalarm']:.2%}, "
        f"Verwechslung {g['verwechslung']:.2%}) | Modell fand {len(hyp_labels)} von {len(ref_labels)} Personen | "
        f"Beiträge richtig: {result['beitraege_richtig']}/{len(beitraege)} | Zuordnung {mapping}"
    )
    o = result["nur_ueberlappung"]
    print(f"Nur Überlappung ({o['soll_sekunden']:.1f} s Soll-Sprechzeit): DER {o['der']:.2%}")


if __name__ == "__main__":
    main()
