# radiosynth - From MHD to MHz: synthetic radio images of MHD models of the Sun and stars
# Copyright (C) 2026 Soumitra Hazra
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate coderef.tex: line-by-line code reference for radiosynth.

Each module is split into blocks that begin at a *marker* (a substring of the first
line of the block, searched from the start of the previous block).  Commentary items
refer to lines by markers inside the block; they are resolved to real line numbers,
so the document always matches the source.
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "radiosynth")
DST = os.path.join(HERE, "src")
os.makedirs(DST, exist_ok=True)


def esc_all(c):
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "{": r"\{", "}": r"\}", "#": r"\#",
           "%": r"\%", "&": r"\&", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
    return "".join(rep.get(ch, ch) for ch in c)


def tex(s):
    """Light markup -> LaTeX: `code` becomes \\code{...} with escaping."""
    def esc(c):
        rep = {"\\": r"\textbackslash{}", "_": r"\_", "{": r"\{", "}": r"\}", "#": r"\#",
               "%": r"\%", "&": r"\&", "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
        o = "".join(rep.get(ch, ch) + (r"\allowbreak{}" if ch in "*=,(/_." else "") for ch in c)
        return o
    return re.sub(r"`([^`]+)`", lambda m: r"\code{" + esc(m.group(1)) + "}", s)


from coderef_v2 import MODULES  # noqa: E402


def find(lines, marker, start, stop=None):
    stop = len(lines) if stop is None else stop
    for i in range(start, stop):
        if marker in lines[i]:
            return i
    raise ValueError(f"marker {marker!r} not found after line {start + 1}")


out = []
for fname, title, intro, blocks in MODULES:
    os.makedirs(os.path.dirname(os.path.join(DST, fname)), exist_ok=True)
    shutil.copy(os.path.join(SRC, fname), os.path.join(DST, fname))
    lines = open(os.path.join(SRC, fname)).read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n = len(lines)
    starts = []
    pos = 0
    for b in blocks:
        i = find(lines, b[0], pos) if b[0] else 0
        starts.append(i)
        pos = i + 1 if b[0] else 0
    out.append(r"\chapter{" + esc_all(title) + "}")
    out.append(r"\label{mod:" + fname.replace("_", "").replace("/", ":") + "}")
    out.append(r"\textbf{File:} \code{radiosynth/" + esc_all(fname) + "} --- " + str(n) + " lines.\n")
    out.append(tex(intro) + "\n")
    for k, b in enumerate(blocks):
        a = starts[k]
        e = (starts[k + 1] - 1) if k + 1 < len(blocks) else n - 1
        while e > a and lines[e].strip() == "":
            e -= 1
        heading, items = b[1], b[2]
        out.append(r"\section*{\normalsize " + esc_all(heading) +
                   r" \hfill{\small\mdseries\color{gray}lines " + f"{a + 1}--{e + 1}" + "}}")
        out.append(r"\addcontentsline{toc}{section}{" + esc_all(heading) + "}")
        out.append(r"\lstinputlisting[style=py,firstline=" + str(a + 1) + ",lastline=" +
                   str(e + 1) + ",firstnumber=" + str(a + 1) + "]{src/" + fname + "}")
        paras, its = [], []
        for it in items:
            if isinstance(it, str):
                if its:
                    paras.append(("list", its))
                    its = []
                paras.append(("p", it))
            else:
                ref, txt = it
                if isinstance(ref, tuple):
                    i1 = find(lines, ref[0], a, e + 1)
                    i2 = find(lines, ref[1], i1, e + 1)
                    lab = f"L.{i1 + 1}\u2013{i2 + 1}" if i2 > i1 else f"L.{i1 + 1}"
                else:
                    i1 = find(lines, ref, a, e + 1)
                    lab = f"L.{i1 + 1}"
                its.append((lab, txt))
        if its:
            paras.append(("list", its))
        for kind, v in paras:
            if kind == "p":
                out.append(tex(v) + "\n")
            else:
                out.append(r"\begin{lines}")
                for lab, txt in v:
                    out.append(r"\item[" + lab + "] " + tex(txt))
                out.append(r"\end{lines}")
    out.append("")

open(os.path.join(HERE, "coderef.tex"), "w").write("\n".join(out))
print("coderef.tex written")
