"""Two baselines: leave the text unchanged, and a most-frequent-form dictionary."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from metrics import EDGE, core, load_pairs, score, unmark, word_forms

train = load_pairs("data/train.tsv")
forms = word_forms(t for _, t in train)
best = {bare: c.most_common(1)[0][0] for bare, c in forms.items()}


def dictionary(sentence):
    out = []
    for token in sentence.split():
        lead, word, trail = EDGE.match(token).groups()
        w = best.get(unmark(word.lower()), word)
        if word[:1].isupper():
            w = w[:1].upper() + w[1:]
        out.append(lead + w + trail)
    return " ".join(out)


results = {}
for split in ("dev", "test"):
    pairs = load_pairs(f"data/{split}.tsv")
    inputs, refs = [i for i, _ in pairs], [t for _, t in pairs]
    for name, fn in (("no change", lambda s: s), ("dictionary", dictionary)):
        results[f"{split}/{name}"] = score([fn(s) for s in inputs], refs, forms)

os.makedirs("results", exist_ok=True)
json.dump(results, open("results/baselines.json", "w"), indent=2)
print(f"Test words that are ambiguous: {results['test/no change']['share_ambiguous']:.1%}, "
      f"unseen in training: {results['test/no change']['share_unseen']:.1%}\n")
print(f"{'':22}{'WER':>7}{'word acc':>10}{'ambiguous':>11}{'unseen':>8}")
for k, r in results.items():
    print(f"{k:22}{r['WER']:7.1%}{r['word_acc']:10.1%}{r['ambiguous_acc']:11.1%}{r['unseen_acc']:8.1%}")
print("\nPublished best on this test set (Orife et al., 2020): WER 20.4%")
ex = load_pairs("data/test.tsv")[0]
print(f"\nExample\n  input:      {ex[0]}\n  dictionary: {dictionary(ex[0])}\n  answer:     {ex[1]}")
