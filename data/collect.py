"""Collect accented Yoruba text: training from MENYO-20k + yoruba-text, test from Global Voices."""
import csv, glob, re, sys, unicodedata
from collections import Counter

RAW = "data/raw/"
RELIGIOUS = "--religious" in sys.argv
MARKS = {"̀", "́", "̣"}  # grave, acute, dot below


def nfc(s):
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s).strip())


def strip(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if c not in MARKS).lower()


def marked_share(s):
    words = s.split()
    return sum(any(c in MARKS for c in unicodedata.normalize("NFD", w)) for w in words) / max(len(words), 1)


def sentences(s):
    return [x for x in re.split(r"(?<=[.!?])\s+", s) if x]


def verdict(s):
    n = len(s.split())
    if n < 3:
        return "too short"
    if n > 80:
        return "too long"
    return "ok" if marked_share(s) >= 0.3 else "unmarked"


def lines(pattern, skip=()):
    for f in sorted(glob.glob(RAW + pattern, recursive=True)):
        if not any(k in f for k in skip):
            yield from open(f, encoding="utf-8", errors="ignore")


def menyo():
    for f in sorted(glob.glob(RAW + "menyo/verified/*.csv")):
        if "global_voices" in f:
            continue
        for row in csv.reader(open(f, encoding="utf-8")):
            if len(row) >= 2 and row[1] != "yoruba":
                yield row[1]


SOURCES = {
    "menyo": menyo(),
    "lagos_nwu": lines("yoruba-text/LagosNWU/**/*.txt", skip=("README",)),
    "proverbs": lines("yoruba-text/Owe/**/*.txt"),
}
if RELIGIOUS:
    SOURCES |= {"bible": lines("yoruba-text/Bibeli_Mimo/**/*.txt"),
                "quran": lines("yoruba-text/Quran_Mimo/**/*.txt"),
                "mormon": lines("yoruba-text/Book_of_Mormon/**/*.txt")}

test = list(dict.fromkeys(nfc(l) for l in open(RAW + "yoruba-text/Iroyin/yoglobalvoices.txt", encoding="utf-8") if l.strip()))
seen = {strip(t) for t in test}
train, origin, kept, total, why, share, dropped = [], [], Counter(), Counter(), Counter(), Counter(), []
for name, src in SOURCES.items():
    for raw in src:
        for s in sentences(nfc(raw)):
            total[name] += 1
            v = verdict(s)
            if v != "ok":
                why[v] += 1
                if v == "unmarked" and len(dropped) < 8:
                    dropped.append((name, s[:90]))
                continue
            if strip(s) in seen:
                why["duplicate"] += 1
                continue
            seen.add(strip(s))
            train.append(s)
            origin.append(name)
            kept[name] += 1
            share[name] += marked_share(s)

open("data/train_raw.txt", "w").write("\n".join(train) + "\n")
open("data/train_sources.txt", "w").write("\n".join(origin) + "\n")
open("data/test_raw.txt", "w").write("\n".join(test) + "\n")
for name in SOURCES:
    print(f"{name:11} kept {kept[name]:6} of {total[name]:6} sentences, "
          f"{share[name] / max(kept[name], 1):.0%} of words marked on average")
print("dropped:", dict(why))
print(f"\nTRAIN {len(train)} sentences, {sum(len(s.split()) for s in train):,} words")
print(f"TEST  {len(test)} sentences, {sum(len(s.split()) for s in test):,} words")
print("\nSample of sentences dropped as unmarked (should be English or unaccented):")
for name, s in dropped:
    print(f"  [{name}] {s}")
