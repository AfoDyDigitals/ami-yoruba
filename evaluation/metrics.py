"""Shared scoring for diacritic restoration: WER, word accuracy, ambiguous- and unseen-word accuracy."""
import re, unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

MARKS = {"̀", "́", "̣"}  # grave, acute, dot below
EDGE = re.compile(r"^([^\w]*)(.*?)([^\ẁ-ͯ]*)$")


def unmark(s):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if c not in MARKS))


def core(token):
    """The word inside a token, without surrounding punctuation, lowercased."""
    return EDGE.match(token).group(2).lower()


def load_pairs(path):
    return [line.rstrip("\n").split("\t") for line in open(path, encoding="utf-8")]


def word_forms(targets):
    """bare word -> Counter of accented forms seen in training."""
    forms = defaultdict(Counter)
    for s in targets:
        for t in s.split():
            w = core(unicodedata.normalize("NFC", t))
            if w:
                forms[unmark(w)][w] += 1
    return forms


def is_ambiguous(counter, min_share=0.1):
    """2+ accented forms, the second one at least 10% as common as all forms together."""
    top = counter.most_common(2)
    return len(top) == 2 and top[1][1] >= min_share * sum(counter.values())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(preds, refs, forms):
    errors = words = 0
    hits = Counter()
    for p, r in zip(preds, refs):
        p, r = unicodedata.normalize("NFC", p).split(), unicodedata.normalize("NFC", r).split()
        errors += edit_distance(p, r)
        words += len(r)
        aligned = p if len(p) == len(r) else _align(p, r)
        for pw, rw in zip(aligned, r):
            bare = unmark(core(rw))
            kind = "unseen" if bare not in forms else "ambiguous" if is_ambiguous(forms[bare]) else "other"
            for k in ("all", kind):
                hits[k, "n"] += 1
                hits[k, "ok"] += pw == rw
    acc = lambda k: hits[k, "ok"] / max(hits[k, "n"], 1)
    return {"WER": errors / words, "word_acc": acc("all"), "ambiguous_acc": acc("ambiguous"),
            "unseen_acc": acc("unseen"), "share_ambiguous": hits["ambiguous", "n"] / hits["all", "n"],
            "share_unseen": hits["unseen", "n"] / hits["all", "n"]}


def _align(p, r):
    """Line prediction words up with reference words when the counts differ (missing ones count as wrong)."""
    out = [None] * len(r)
    sm = SequenceMatcher(a=[unmark(core(x)) for x in p], b=[unmark(core(x)) for x in r], autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        out[j:j + n] = p[i:i + n]
    return out
