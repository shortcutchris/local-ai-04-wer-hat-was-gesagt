"""Protokoll gegen die vorab festgelegten Erwartungen prüfen.

Zu jeder erwarteten Aufgabe wird die Protokollzeile gesucht, die das eindeutigste Stichwort
enthält; eine Zeile darf mehrere Aufgaben abdecken. Bewertet werden Person und Termin getrennt. Zusätzlich gezählt: Aufgaben bei der falschen Person,
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

    def keyword_rank(task, exp):
        """Niedrigster Index eines getroffenen Stichworts; die Stichworte sind nach Eindeutigkeit sortiert."""
        text = (task.get("was", "") + " " + task.get("beleg", "")).lower()
        ranks = [i for i, k in enumerate(exp["stichworte"]) if k in text]
        return min(ranks) if ranks else None

    def norm(s):
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    used = set()
    rows = []
    for exp in spec["erwartete_aufgaben"]:
        # Eine Protokollzeile darf mehrere erwartete Aufgaben abdecken (z. B. zwei Aufgaben in einem Satz).
        candidates = [(keyword_rank(t, exp), i) for i, t in enumerate(tasks) if keyword_rank(t, exp) is not None]
        candidates.sort(key=lambda c: (c[0], owner(tasks[c[1]]) != exp["wer"]))
        if candidates:
            match = candidates[0][1]
            used.add(match)
            t = tasks[match]
            wer_ok = owner(t) == exp["wer"]
            bis_ok = norm(exp["bis"]) in norm(t.get("bis")) or norm(t.get("bis")) in norm(exp["bis"])
            rows.append({"erwartet": exp["was"], "soll": exp["wer"], "ist": owner(t), "soll_bis": exp["bis"],
                         "bis": t.get("bis"), "richtig": wer_ok, "termin_richtig": bis_ok and bool(t.get("bis"))})
        else:
            rows.append({"erwartet": exp["was"], "soll": exp["wer"], "ist": None, "soll_bis": exp["bis"], "bis": None,
                         "richtig": False, "termin_richtig": False})

    extra = [t for i, t in enumerate(tasks) if i not in used]
    verworfen = [t for t in tasks if any(k in (t.get("was", "") + t.get("beleg", "")).lower() for k in spec["keine_aufgabe_stichworte"])]
    result = {
        "richtig_zugeordnet": sum(r["richtig"] for r in rows),
        "erwartet": len(rows),
        "termin_richtig": sum(r["termin_richtig"] for r in rows),
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
        term = "Termin ok" if r["termin_richtig"] else f"Termin {r['bis']!r} statt {r['soll_bis']!r}"
        print(f"{mark}{r['erwartet'][:50]:50} soll {r['soll']:7} ist {str(r['ist']):8} {term}")
    print(f"=> {result['richtig_zugeordnet']}/{result['erwartet']} richtige Person, {result['termin_richtig']} Termine richtig, {result['falsche_person']} falsche Person, "
          f"{result['fehlend']} fehlend, {len(extra)} zusätzlich, {len(verworfen)} verworfene aufgenommen")


if __name__ == "__main__":
    main()
