import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "DENOISERL_RECENT_UPDATES.md"
OUTPUT = ROOT / "output" / "pdf" / "DenoiseRL_recent_updates.pdf"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

INK = colors.HexColor("#14213D")
MUTED = colors.HexColor("#5B6475")
ACCENT = colors.HexColor("#2F6BFF")
ACCENT_DARK = colors.HexColor("#1747B8")
PALE_BLUE = colors.HexColor("#EEF4FF")
PALE_GREEN = colors.HexColor("#EEF9F4")
PALE_AMBER = colors.HexColor("#FFF7E8")
LINE = colors.HexColor("#DDE3EE")


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace(r"\(\rho=0.2\)", "ρ=0.2")
    text = text.replace(r"\(\rho\)", "ρ")
    return text


def parse_markdown() -> list[tuple[str, str]]:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    in_formula = False
    formula: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(paragraph)))
            paragraph.clear()

    for raw in lines:
        line = raw.strip()
        if in_formula:
            if line == r"\]":
                blocks.append(("formula", " ".join(formula)))
                formula.clear()
                in_formula = False
            else:
                formula.append(line)
            continue
        if line == r"\[":
            flush()
            in_formula = True
        elif not line:
            flush()
        elif line.startswith("# "):
            flush()
            blocks.append(("title", line[2:]))
        elif line.startswith("## "):
            flush()
            blocks.append(("section", line[3:]))
        elif line.startswith("- "):
            flush()
            blocks.append(("bullet", line[2:]))
        else:
            paragraph.append(line)
    flush()
    return blocks


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName="ArialUnicode",
            fontSize=24, leading=31, textColor=INK, alignment=TA_LEFT,
            spaceAfter=2 * mm, wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=10.5, leading=16, textColor=MUTED, wordWrap="CJK",
        ),
        "section": ParagraphStyle(
            "SectionCN", parent=base["Heading2"], fontName="ArialUnicode",
            fontSize=17, leading=22, textColor=INK, wordWrap="CJK",
        ),
        "badge": ParagraphStyle(
            "Badge", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=12, leading=15, textColor=colors.white, alignment=TA_CENTER,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName="ArialUnicode",
            fontSize=10.6, leading=17.5, textColor=INK, alignment=TA_LEFT,
            spaceAfter=3.2 * mm, wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN", parent=base["BodyText"], fontName="ArialUnicode",
            fontSize=10.2, leading=16.8, leftIndent=5 * mm,
            firstLineIndent=-4 * mm, textColor=INK, spaceAfter=3.2 * mm,
            wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "Equation", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=14.5, leading=23, textColor=ACCENT_DARK,
            alignment=TA_CENTER,
        ),
        "card_title": ParagraphStyle(
            "CardTitle", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=11.2, leading=15, textColor=INK, wordWrap="CJK",
        ),
        "card_body": ParagraphStyle(
            "CardBody", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=9.6, leading=15.5, textColor=MUTED, wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["Normal"], fontName="ArialUnicode",
            fontSize=10, leading=16, textColor=INK, wordWrap="CJK",
        ),
    }


def section_header(number: str, title: str, styles) -> KeepTogether:
    row = Table(
        [[Paragraph(number, styles["badge"]), Paragraph(title, styles["section"])]],
        colWidths=[10 * mm, 148 * mm],
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, 0), ACCENT),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 0), (0, 0), 5),
        ("BOTTOMPADDING", (0, 0), (0, 0), 5),
        ("LEFTPADDING", (1, 0), (1, 0), 9),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    return KeepTogether([Spacer(1, 4 * mm), row, Spacer(1, 3.5 * mm)])


def equation_box(styles) -> Table:
    equation = Paragraph(
        "ρ<sub>i</sub> ← clip(ρ<sub>i</sub> + α(A<sub>i</sub> - A<sub>target</sub>), ρ<sub>min</sub>, ρ<sub>max</sub>)",
        styles["equation"],
    )
    box = Table([[equation]], colWidths=[158 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#BFD0FF")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return box


def comparison_box(pro: str, con: str, styles) -> Table:
    table = Table([
        [Paragraph("优点 · 细粒度", styles["card_title"]),
         Paragraph("缺点 · 多样性受限", styles["card_title"])],
        [Paragraph(inline(pro), styles["card_body"]),
         Paragraph(inline(con), styles["card_body"])],
    ], colWidths=[76 * mm, 76 * mm], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE_GREEN),
        ("BACKGROUND", (1, 0), (1, -1), PALE_AMBER),
        ("BOX", (0, 0), (0, -1), 0.7, colors.HexColor("#CDE8DA")),
        ("BOX", (1, 0), (1, -1), 0.7, colors.HexColor("#F1D9A6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("TOPPADDING", (0, 1), (-1, 1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
    ]))
    return table


def callout(text: str, styles) -> Table:
    table = Table([[Paragraph(inline(text), styles["callout"])]], colWidths=[158 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return table


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    canvas.setFont("ArialUnicode", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 9.2 * mm, "DenoiseRL 近期改进与后续计划")
    canvas.drawRightString(width - 20 * mm, 9.2 * mm, str(doc.page))
    canvas.restoreState()


def build() -> None:
    pdfmetrics.registerFont(TTFont("ArialUnicode", FONT))
    pdfmetrics.registerFontFamily(
        "ArialUnicode", normal="ArialUnicode", bold="ArialUnicode",
        italic="ArialUnicode", boldItalic="ArialUnicode",
    )
    styles = make_styles()
    blocks = parse_markdown()
    output = []
    section_number = 0
    index = 0
    while index < len(blocks):
        kind, text = blocks[index]
        if kind == "title":
            output.extend([
                Spacer(1, 3 * mm),
                Paragraph(inline(text), styles["title"]),
                Paragraph("动态难度控制、跨任务验证与噪声来源分析", styles["subtitle"]),
                Spacer(1, 5 * mm),
                HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=4 * mm),
            ])
        elif kind == "section":
            section_number += 1
            if section_number == 2:
                output.append(PageBreak())
            clean_title = re.sub(r"^\d+\.\s*", "", text)
            output.append(section_header(str(section_number), clean_title, styles))
        elif kind == "formula":
            output.extend([equation_box(styles), Spacer(1, 4 * mm)])
        elif kind == "bullet":
            output.append(Paragraph(inline(text), styles["bullet"], bulletText="•"))
        elif kind == "paragraph":
            if text.startswith("该机制的主要优点") and index + 1 < len(blocks):
                next_kind, next_text = blocks[index + 1]
                if next_kind == "paragraph" and next_text.startswith("它的主要缺点"):
                    pro = text.replace("该机制的主要优点是控制更加细粒度：", "", 1)
                    con = next_text.replace("它的主要缺点是训练多样性可能下降。", "", 1)
                    output.append(comparison_box(pro, con, styles))
                    index += 1
                else:
                    output.append(Paragraph(inline(text), styles["body"]))
            elif text.startswith("如果动态控制没有取得"):
                output.append(callout(text, styles))
            else:
                output.append(Paragraph(inline(text), styles["body"]))
        index += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=26 * mm, rightMargin=26 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="DenoiseRL 近期改进与后续计划", author="DenoiseRL",
    )
    doc.build(output, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
