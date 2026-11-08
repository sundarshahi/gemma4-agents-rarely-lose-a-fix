"""writeup.md -> paper.tex -> paper.pdf (tectonic). Handles only what writeup.md uses:
headings, paragraphs, **bold**, *italic*, `code`, bullet/numbered lists, pipe tables, images, the key-findings
blockquote and the reference list. Usage: python paper/build_pdf.py && tectonic paper/paper.tex"""
import re
from pathlib import Path

HERE = Path(__file__).parent
md = (HERE / 'writeup.md').read_text()
lines = md.splitlines()
title = lines[0].lstrip('# ').strip()
subtitle = next(l for l in lines if l.startswith('**Subtitle:**')).replace('**Subtitle:**', '').strip()
body = md[md.index('> **Key findings**'):]


def esc(s):
    s = s.replace('\\', r'\textbackslash{}')
    for a, b in (('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    s = s.replace('×10⁻⁸', r'$\times 10^{-8}$').replace('×', r'$\times$').replace('−', '--').replace('—', '---')
    s = s.replace('–', '--').replace('≈', r'$\approx$').replace('·', r'$\cdot$')
    return s


def inline(s):
    parts = re.split(r'(`[^`]+`)', s)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`'):
            out.append(r'\texttt{' + esc(p[1:-1]) + '}')
            continue
        p = esc(p)
        p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\\emph{\1}', p)
        p = re.sub(r'(https?://[^\s)]+)', lambda m: r'\url{' + m.group(1).replace(r'\_', '_').replace(r'\#', '#') + '}', p)
        out.append(p)
    return ''.join(out)


tex, i, L = [], 0, body.splitlines()
fig_n = 0
while i < len(L):
    l = L[i]
    if l.startswith('## References'):
        tex.append(r'\section*{References}\small')
        i += 1
        continue
    if l.startswith('### '):
        tex.append(r'\subsection*{' + inline(re.sub(r'^\d+(\.\d+)*\s+', '', l[4:])) + '}')
    elif l.startswith('## '):
        tex.append(r'\section*{' + inline(re.sub(r'^\d+\.\s+', '', l[3:])) + '}')
    elif l.startswith('>'):
        box = []
        while i < len(L) and L[i].startswith('>'):
            box.append(L[i][1:].strip())
            i += 1
        items = [inline(b[2:]) for b in box if b.startswith('- ')]
        tex.append(r'\begin{center}\fbox{\begin{minipage}{0.95\linewidth}\small\textbf{Key findings}\begin{itemize}\setlength\itemsep{2pt}'
                   + ''.join(r'\item ' + it for it in items) + r'\end{itemize}\end{minipage}}\end{center}')
        continue
    elif l.startswith('!['):
        fig_n += 1
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', l)
        cap = re.sub(r'^Figure \d+:\s*', '', m.group(1))
        tex.append(r'\begin{figure}[h]\centering\includegraphics[width=\linewidth]{fig/' + m.group(2).rsplit('/', 1)[-1]
                   + r'}\caption{' + inline(cap[0].upper() + cap[1:]) + r'.}\end{figure}')
    elif l.startswith('|'):
        rows = []
        while i < len(L) and L[i].startswith('|'):
            if not re.match(r'^\|[-\s|]+\|$', L[i]):
                rows.append([c.strip() for c in L[i].strip('|').split('|')])
            i += 1
        cols = 'l' + 'r' * (len(rows[0]) - 1)
        tex.append(r'\begin{center}\small\begin{tabular}{' + cols + r'}\toprule')
        tex.append(' & '.join(r'\textbf{' + inline(c) + '}' for c in rows[0]) + r' \\ \midrule')
        tex += [' & '.join(inline(c) for c in r) + r' \\' for r in rows[1:]]
        tex.append(r'\bottomrule\end{tabular}\end{center}')
        continue
    elif re.match(r'^(- |\d+\. )', l):
        numbered = bool(re.match(r'^\d+\. ', l))
        env = 'enumerate' if numbered else 'itemize'
        items = []
        while i < len(L) and re.match(r'^(- |\d+\. )', L[i]):
            items.append(inline(re.sub(r'^(- |\d+\. )', '', L[i])))
            i += 1
        tex.append(r'\begin{' + env + r'}\setlength\itemsep{2pt}' + ''.join(r'\item ' + it for it in items) + r'\end{' + env + '}')
        continue
    elif l.startswith('[') and re.match(r'^\[\d+\]', l):
        tex.append(r'\noindent ' + inline(l) + r'\par\smallskip')
    elif l.strip():
        para = [l]
        while i + 1 < len(L) and L[i + 1].strip() and not re.match(r'^(#|>|!\[|\||- |\d+\. |\[\d+\])', L[i + 1]):
            i += 1
            para.append(L[i])
        tex.append(inline(' '.join(para)) + '\n')
    i += 1

doc = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{fontspec}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\usepackage{microtype}
\usepackage[font=small]{caption}
\title{""" + inline(title) + r"""\\[4pt]\large """ + inline(subtitle) + r"""}
\author{Sundar Shahi Thakuri\\ Independent researcher\\ \small Gemma 4 Developer Agent Paper Track}
\date{November 2026}
\begin{document}
\maketitle
""" + '\n'.join(tex) + r"""
\end{document}
"""
(HERE / 'paper.tex').write_text(doc)
print('wrote paper.tex', len(doc), 'chars,', fig_n, 'figures')
