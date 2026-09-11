"""Validate markdown tables: a header row, a separator, and a stable width.

The previous check only looked for contiguous runs of pipe-prefixed lines, so
it passed a table that a stray blank line had split in two - the second half
rendered as a headerless block. A table is only well-formed if its second line
is a |---|---| separator and every row has the same column count.
"""
import re
import sys

SEP = re.compile(r"^\|[\s:|-]+\|$")


def check(path):
    lines = open(path).read().splitlines()
    blocks, cur = [], []
    for i, line in enumerate(lines, 1):
        if line.startswith("|"):
            cur.append((i, line))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    bad = 0
    for b in blocks:
        first, rows = b[0][0], [l for _, l in b]
        if len(rows) < 2 or not SEP.match(rows[1]):
            print(f"{path}:{first}: table has no |---| separator on its 2nd line "
                  f"(headerless fragment?)")
            bad += 1
            continue
        widths = {r.count("|") for r in rows}
        if len(widths) > 1:
            print(f"{path}:{first}: ragged table, pipe counts {sorted(widths)}")
            bad += 1
    return len(blocks), bad


total = fails = 0
for p in sys.argv[1:]:
    n, b = check(p)
    total += n
    fails += b
    print(f"{p}: {n} tables, {b} malformed")
sys.exit(1 if fails else 0)
