"""Protokoll gegen die vorab festgelegten Erwartungen prüfen.

Eine Aufgabe gilt als getroffen, wenn eine Protokollzeile der richtigen Person zugeordnet ist
und eines der Stichworte enthält. Zusätzlich gezählt: Aufgaben bei der falschen Person,
erfundene Aufgaben und ausdrücklich verworfene Aufgaben (Ersatzwerkzeug).

Aufruf:  python eval/check_protocol.py results/<lauf>/protokoll.json --names '{"speaker_0": "petra"}'
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protokoll", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    spec = json.loads((ROOT / "fixtures" / "besprechung.json").read_text())
    persons = {k: v["name"].split()[0].lower() for k, v in spec["sprecher"].items()}
    tasks = json.loads(args.protokoll.read_text())["protokoll"]["aufgaben"]

    def owner(task):
        w = task.get("wer", "").lower()
        hits = [k for k, first in persons.items() if first in w]
        return hits[0] if len(hits) == 1 else w

    used = set()
    rows = []
    for exp in spec["erwartete_aufgaben"]:
        match = None
        for i, t in enumerate(tasks):
            text = (t.get("was", "") + " " + t.get("beleg", "")).lower()
            if i not in used and any(k in text for k in exp["stichworte"]):
                match = i
                if owner(t) == exp["wer"]:
                    break
        if match is not None:
            used.add(match)
            t = tasks[match]
            ok = owner(t) == exp["wer"]
            rows.append({"erwartet": exp["was"], "soll": exp["wer"], "ist": owner(t), "bis": t.get("bis"), "richtig": ok})
        else:
            rows.append({"erwartet": exp["was"], "soll": exp["wer"], "ist": None, "bis": None, "richtig": False})

    extra = [t for i, t in enumerate(tasks) if i not in used]
    verworfen = [t for t in tasks if any(k in (t.get("was", "") + t.get("beleg", "")).lower() for k in spec["keine_aufgabe_stichworte"])]
    result = {
        "richtig_zugeordnet": sum(r["richtig"] for r in rows),
        "erwartet": len(rows),
        "falsche_person": sum(1 for r in rows if r["ist"] and not r["richtig"]),
        "fehlend": sum(1 for r in rows if r["ist"] is None),
        "zusaetzlich": extra,
        "verworfene_aufgabe_aufgenommen": verworfen,
        "zeilen": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for r in rows:
        mark = "OK " if r["richtig"] else "XX "
        print(f"{mark}{r['erwartet'][:55]:55} soll {r['soll']:7} ist {str(r['ist']):14} bis {r['bis']}")
    print(f"=> {result['richtig_zugeordnet']}/{result['erwartet']} richtig, {result['falsche_person']} falsche Person, "
          f"{result['fehlend']} fehlend, {len(extra)} zusätzlich, {len(verworfen)} verworfene aufgenommen")


if __name__ == "__main__":
    main()
