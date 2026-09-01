"""Render the project's Markdown reports to PDF with ReportLab.

No pandoc / LaTeX / wkhtmltopdf on the target machines, so this handles the Markdown
subset the reports actually use: headings, paragraphs, bullet and numbered lists, pipe
tables, images, fenced code blocks, horizontal rules, and inline **bold** / *italic* /
`code`. Anything outside that subset is emitted as plain text rather than silently
dropped.

Run:  python -m pinca_jax.md2pdf docs/PI-NCA_Architectures_and_Results.md
      python -m pinca_jax.md2pdf in.md out.pdf
"""
from __future__ import annotations

import argparse
import html
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable

ACCENT = colors.HexColor("#2C6EA8")
INK = colors.HexColor("#1B1B22")
MUTED = colors.HexColor("#5A5A66")
RULE = colors.HexColor("#D3D3DA")
CODE_BG = colors.HexColor("#F3F3F6")
HEAD_BG = colors.HexColor("#E8EEF6")
ZEBRA = colors.HexColor("#F7F8FA")

PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def _styles():
    ss = getSampleStyleSheet()
    mk = lambda **kw: ParagraphStyle(**kw)
    return {
        "h1": mk(name="h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17,
                 leading=21, spaceBefore=14, spaceAfter=7, textColor=ACCENT),
        "h2": mk(name="h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                 leading=16.5, spaceBefore=13, spaceAfter=5, textColor=ACCENT),
        "h3": mk(name="h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                 leading=14, spaceBefore=10, spaceAfter=4, textColor=INK),
        "h4": mk(name="h4", parent=ss["Heading4"], fontName="Helvetica-Bold", fontSize=10,
                 leading=13, spaceBefore=8, spaceAfter=3, textColor=INK),
        "body": mk(name="body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3,
                   leading=13.4, spaceAfter=5, alignment=TA_LEFT, textColor=INK),
        "bullet": mk(name="bullet", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3,
                     leading=13.2, leftIndent=11, bulletIndent=2, spaceAfter=2.5,
                     textColor=INK),
        "caption": mk(name="caption", fontName="Helvetica-Oblique", fontSize=7.8, leading=10,
                      alignment=1, textColor=MUTED, spaceAfter=8),
        "code": mk(name="code", fontName="Courier", fontSize=8.0, leading=10.6,
                   leftIndent=6, backColor=CODE_BG, borderPadding=5, spaceBefore=3,
                   spaceAfter=7, textColor=INK),
        "cell": mk(name="cell", fontName="Helvetica", fontSize=7.6, leading=9.8,
                   textColor=INK),
        "cellhead": mk(name="cellhead", fontName="Helvetica-Bold", fontSize=7.6, leading=9.8,
                       textColor=INK),
        "title": mk(name="title", fontName="Helvetica-Bold", fontSize=25, leading=30,
                    textColor=ACCENT, spaceAfter=5),
        "sub": mk(name="sub", fontName="Helvetica", fontSize=10.5, leading=15,
                  textColor=MUTED, spaceAfter=3),
    }


def inline(text: str) -> str:
    """Markdown inline markup -> ReportLab mini-HTML. Escapes first, so `<` is safe."""
    out = html.escape(text, quote=False)
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)                    # images handled apart
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", out)              # links -> label
    out = re.sub(r"`([^`]+)`",
                 r'<font face="Courier" size="8">\1</font>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", out)
    return out


def _split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _col_widths(rows, total):
    """Width proportional to the longest cell per column, with a sane floor."""
    ncol = max(len(r) for r in rows)
    longest = [1] * ncol
    for r in rows:
        for i, c in enumerate(r):
            longest[i] = max(longest[i], min(len(re.sub(r"[*`]", "", c)), 46))
    scale = total / sum(longest)
    w = [max(l * scale, 15 * mm) for l in longest]
    over = sum(w) - total
    if over > 0:                       # give the surplus back to the widest column
        w[w.index(max(w))] -= over
    return w


def build_table(rows, styles):
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), styles["cellhead"]) for c in head]]
    data += [[Paragraph(inline(c), styles["cell"]) for c in r] for r in body]
    ncol = len(head)
    data = [r + [Paragraph("", styles["cell"])] * (ncol - len(r)) for r in data]
    t = Table(data, colWidths=_col_widths(rows, CONTENT_W), repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def image_flowable(path, base_dir, styles, caption=None):
    full = os.path.normpath(os.path.join(base_dir, path))
    if not os.path.exists(full):
        return [Paragraph(inline(f"[missing image: {path}]"), styles["caption"])]
    from PIL import Image as PILImage
    with PILImage.open(full) as im:
        iw, ih = im.size
    w = CONTENT_W
    h = w * ih / iw
    max_h = 118 * mm                       # keep any single figure on one page
    if h > max_h:
        h = max_h
        w = h * iw / ih
    flow = [Image(full, width=w, height=h)]
    if caption:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(inline(caption), styles["caption"]))
    return flow


def parse(md: str, base_dir: str, styles):
    story, lines, i = [], md.split("\n"), 0
    pending_img = None                     # image waiting for the italic line beneath it

    def flush_img():
        nonlocal pending_img
        if pending_img:
            story.append(KeepTogether(image_flowable(pending_img, base_dir, styles)))
            pending_img = None

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            # An image's caption is usually separated from it by a blank line, so peek
            # past the blanks before flushing — otherwise the caption renders as body text.
            if pending_img:
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                nxt = lines[j].strip() if j < len(lines) else ""
                if nxt.startswith("*") and nxt.endswith("*") and not nxt.startswith("**"):
                    i = j
                    continue
            flush_img()
            i += 1
            continue

        if stripped.startswith("```"):                       # fenced code
            flush_img()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = html.escape("\n".join(buf)).replace(" ", "&nbsp;").replace("\n", "<br/>")
            story.append(Paragraph(body, styles["code"]))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)         # headings
        if m:
            flush_img()
            level = len(m.group(1))
            if level == 1 and story:
                story.append(PageBreak())
            story.append(Paragraph(inline(m.group(2)), styles[f"h{level}"]))
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):    # horizontal rule
            flush_img()
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            story.append(Spacer(1, 5))
            i += 1
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)   # image
        if m:
            flush_img()
            pending_img = m.group(2)
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|[\s\-:|]+\|?$", lines[i + 1].strip()):
            flush_img()
            rows = [_split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i].strip()))
                i += 1
            story.append(build_table(rows, styles))
            story.append(Spacer(1, 7))
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)             # bullet
        if m:
            flush_img()
            story.append(Paragraph(inline(m.group(1)), styles["bullet"], bulletText="•"))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)          # numbered
        if m:
            flush_img()
            story.append(Paragraph(inline(m.group(2)), styles["bullet"],
                                   bulletText=f"{m.group(1)}."))
            i += 1
            continue

        if stripped.startswith(">"):                          # blockquote
            flush_img()
            story.append(Paragraph(inline(stripped.lstrip("> ")), styles["body"]))
            i += 1
            continue

        # a *italic line* directly under an image is that image's caption
        if pending_img and stripped.startswith("*") and stripped.endswith("*"):
            story.append(KeepTogether(image_flowable(pending_img, base_dir, styles,
                                                     caption=stripped.strip("*"))))
            pending_img = None
            i += 1
            continue

        flush_img()
        para = [stripped]                                    # join wrapped prose lines
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "-", "*", ">", "```", "!["))
                    or re.match(r"^\d+\.\s", nxt)):
                break
            para.append(nxt)
            i += 1
        story.append(Paragraph(inline(" ".join(para)), styles["body"]))

    flush_img()
    return story


def _decorate(canvas, doc, title):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm, title)
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.restoreState()


def convert(md_path, pdf_path=None, title=None, subtitle=None):
    md_path = os.path.abspath(md_path)
    pdf_path = pdf_path or os.path.splitext(md_path)[0] + ".pdf"
    base_dir = os.path.dirname(md_path)
    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    styles = _styles()
    # The first `# ` heading becomes the cover title rather than a body heading.
    m = re.search(r"^#\s+(.*)$", md, flags=re.M)
    doc_title = title or (m.group(1) if m else os.path.basename(md_path))
    if m:
        md = md[:m.start()] + md[m.end():]

    story = [Spacer(1, 46 * mm), Paragraph(html.escape(doc_title), styles["title"]),
             HRFlowable(width="42%", thickness=1.6, color=ACCENT, spaceAfter=9,
                        hAlign="LEFT")]
    if subtitle:
        story.append(Paragraph(html.escape(subtitle), styles["sub"]))
    story.append(PageBreak())
    story += parse(md, base_dir, styles)

    doc = BaseDocTemplate(pdf_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=20 * mm, title=doc_title)
    frame = Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - MARGIN - 20 * mm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=lambda c, d: _decorate(c, d, doc_title))])
    doc.build(story)
    print(f"[md2pdf] {os.path.relpath(md_path)} -> {os.path.relpath(pdf_path)} "
          f"({os.path.getsize(pdf_path) / 1000:.0f} KB)")
    return pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("pdf", nargs="?", default=None)
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()
    convert(args.md, args.pdf, subtitle=args.subtitle)


if __name__ == "__main__":
    main()
