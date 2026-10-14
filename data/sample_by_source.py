import random, sys
from collections import defaultdict
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
by = defaultdict(list)
for src, s in zip(open("data/train_sources.txt").read().split(), open("data/train_raw.txt").read().splitlines()):
    by[src].append(s)
random.seed(2)
for src, items in by.items():
    print(f"\n== {src} ({len(items)} sentences)")
    for i, s in enumerate(random.sample(items, min(n, len(items))), 1):
        print(f"{i}. {s}")
