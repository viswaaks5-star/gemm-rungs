#!/usr/bin/env python3
"""Instruction mix of the kernels in a `cuobjdump -sass` listing.

    cuobjdump -sass bin/sgemm > bin/sgemm.sass
    python3 tools/sass_mix.py bin/sgemm.sass --match sgemm_naive [--out sass.txt]

For each kernel whose name contains --match: static instruction counts for the whole function,
then for its hot loop, the backward branch whose body holds the most FFMAs. The hot loop's
FFMA share is the issue-bound ceiling in docs/model.md: each scheduler issues one instruction
per clock, so every non-FFMA instruction in the loop is an issue slot no FFMA got. Counts are
static (each instruction once, however often it runs); Nsight Compute gives dynamic counts.
--out writes the matching kernels' SASS to a file. Standard library only.
"""

import argparse
import collections
import re
import shutil
import subprocess
import sys

FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSN = re.compile(
    r"^\s*/\*([0-9a-f]{4,})\*/\s+"            # /*0080*/
    r"(?:@!?U?P(?:T|\d+)\s+)?"               # optional guard: @P0, @!P1, @PT, @!UP0
    r"([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)"  # opcode and modifiers: LDS.U.128
    r"\s*([^;]*);"                           # operands
)
LABEL = re.compile(r"^\s*(\.L_x_\d+)\s*:")
HEX = re.compile(r"0x([0-9a-f]+)")
LABEL_REF = re.compile(r"(\.L_x_\d+)")

Insn = collections.namedtuple("Insn", "addr op mods operands")


def parse(lines):
    """Return {mangled name: (instructions, raw lines)} in listing order."""
    funcs = collections.OrderedDict()
    name, insns, raw, labels, pending = None, [], [], {}, []

    def close():
        if name is not None:
            resolved = []
            for i in insns:
                if i.op == "BRA" and i.operands.startswith("LABEL:"):
                    target = labels.get(i.operands[6:])
                    resolved.append(i._replace(operands="" if target is None else hex(target)))
                else:
                    resolved.append(i)
            funcs[name] = (resolved, raw)

    for line in lines:
        m = FUNC.match(line)
        if m:
            close()
            name, insns, raw, labels, pending = m.group(1), [], [line], {}, []
            continue
        if name is None:
            continue
        raw.append(line)
        lm = LABEL.match(line)
        if lm:
            pending.append(lm.group(1))
            continue
        im = INSN.match(line)
        if im:
            addr = int(im.group(1), 16)
            for label in pending:
                labels[label] = addr
            pending = []
            op, mods, operands = im.group(2), im.group(3), im.group(4).strip()
            if op == "BRA":
                h, ref = HEX.search(operands), LABEL_REF.search(operands)
                operands = hex(int(h.group(1), 16)) if h else (f"LABEL:{ref.group(1)}" if ref else "")
            insns.append(Insn(addr, op, mods, operands))
    close()
    return funcs


def width(mods):
    for w in ("128", "64"):
        if w in mods.split("."):
            return w
    return "32"


def category(i):
    if i.op == "FFMA":
        return "FFMA"
    if i.op in ("LDS", "STS", "LDG", "STG"):
        return f"{i.op}.{width(i.mods)}"
    if i.op == "BAR":
        return "BAR"
    return i.op


def summarize(insns):
    counts = collections.Counter(category(i) for i in insns)
    total = len(insns)
    ffma = counts.get("FFMA", 0)
    memory = sorted(k for k in counts if k.split(".")[0] in ("LDS", "STS", "LDG", "STG"))
    other = [(k, v) for k, v in counts.most_common() if k != "FFMA" and k not in memory and k != "BAR"]
    parts = [f"FFMA {ffma} ({100.0 * ffma / total:.1f}%)" if total else "FFMA 0"]
    parts += [f"{k} {counts[k]}" for k in memory]
    if counts.get("BAR"):
        parts.append(f"BAR {counts['BAR']}")
    parts += [f"{k} {v}" for k, v in other[:8]]
    rest = sum(v for _, v in other[8:])
    if rest:
        parts.append(f"other {rest}")
    return total, ffma, " · ".join(parts)


def hot_loop(insns):
    """The backward branch whose body holds the most FFMAs (ties: the smallest body)."""
    best = None
    for idx, i in enumerate(insns):
        if i.op != "BRA" or not i.operands:
            continue
        target = int(i.operands, 16)
        if target > i.addr:
            continue
        body = [j for j in insns[: idx + 1] if j.addr >= target]
        ffma = sum(1 for j in body if j.op == "FFMA")
        key = (ffma, -len(body))
        if ffma and (best is None or key > best[0]):
            best = (key, target, i.addr, body)
    return None if best is None else best[1:]


def demangle(names):
    tool = shutil.which("c++filt")
    if not tool or not names:
        return {n: n for n in names}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, out.splitlines()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sass", help="output of cuobjdump -sass")
    p.add_argument("--match", default="", help="only kernels whose name contains this")
    p.add_argument("--out", help="write the matching kernels' SASS to this file")
    args = p.parse_args(argv)

    with open(args.sass, encoding="utf-8", errors="replace") as f:
        funcs = parse(f)
    names = demangle(list(funcs))
    chosen = [n for n in funcs if args.match in n or args.match in names[n]]
    if not chosen:
        print(f"no kernel matching '{args.match}' in {args.sass}", file=sys.stderr)
        return 1

    for n in chosen:
        insns, _ = funcs[n]
        total, _, text = summarize(insns)
        print(names[n] if names[n] == n else f"{names[n]}\n  [{n}]")
        print(f"  whole kernel: {total} instructions; {text}")
        loop = hot_loop(insns)
        if loop is None:
            print("  hot loop: none (no backward branch around an FFMA; fully unrolled?)")
        else:
            start, end, body = loop
            total, ffma, text = summarize(body)
            print(f"  hot loop {start:#06x}-{end:#06x}: {total} instructions; {text}")
            print(f"  -> issue-bound ceiling: {100.0 * ffma / total:.1f}% of fp32 peak "
                  f"(FFMA share of the loop; static count)")
        print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for n in chosen:
                f.writelines(line if line.endswith("\n") else line + "\n" for line in funcs[n][1])
                f.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
