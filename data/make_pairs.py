"""Turn accented sentences into (unaccented input, accented answer) pairs; split train/dev; keep test."""
import random, re, unicodedata
from collections import Counter, defaultdict

MARKS = {"\u0300", "\u0301", "\u0323"}  # grave, acute, dot below


def unmark(s):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if c not in MARKS))


def words(s):
    return re.findall(r"[\w\u0300-\u036f]+", s.lower())


def sentences(path):
    out = []
    for line in open(path, encoding="utf-8"):
        out += [x for x in re.split(r"(?<=[.!?])\s+", line.strip()) if x]
    return out


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{unmark(s)}\t{s}\n" for s in rows)


random.seed(0)
train = open("data/train_raw.txt", encoding="utf-8").read().splitlines()
random.shuffle(train)
n_dev = len(train) // 20
dev, train = train[:n_dev], train[n_dev:]
test = sentences("data/test_raw.txt")
for name, rows in [("train", train), ("dev", dev), ("test", test)]:
    write(f"data/{name}.tsv", rows)
    print(f"{name:5} {len(rows):6} pairs, {sum(len(s.split()) for s in rows):7,} words")

forms = defaultdict(Counter)
for s in train:
    for w in words(s):
        forms[unmark(w)][w] += 1
test_words = [w for s in test for w in words(s)]
unseen = sum(unmark(w) not in forms for w in test_words)
ambiguous = sum(len(forms[unmark(w)]) > 1 for w in test_words)
print(f"\nTest words: {len(test_words):,}")
print(f"  bare form never seen in training: {unseen / len(test_words):.1%}")
print(f"  bare form with 2+ accented versions in training: {ambiguous / len(test_words):.1%}")
print("\nMost ambiguous bare forms in training:")
for bare, c in sorted(forms.items(), key=lambda kv: -len(kv[1]))[:5]:
    print(f"  {bare}: " + ", ".join(f"{w} ({n})" for w, n in c.most_common(5)))
print("\nExample pair:")
print("  input: ", unmark(test[0]))
print("  answer:", test[0])
