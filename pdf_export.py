"""Выгрузка маршрута и чек-листа в PDF (reportlab). Русские буквы требуют шрифта TTF, ищем его в системе."""
import io
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from schemas import Itinerary, ItineraryDay
from verified_map_places import is_public_map_point, item_map_attribution
from program_coverage import FREE_DAY_NOTE, program_report_text
from city_estimates import UNPRICED_VISIT_LABEL, city_unpriced_visit_note, is_unpriced_city_visit

_WINDIR = os.environ.get("WINDIR", "C:\\Windows")
# Пары (обычный, жирный). Первый найденный вариант используется. Свой шрифт можно положить в папку fonts.
FONT_CANDIDATES = [
    ("fonts/DejaVuSans.ttf", "fonts/DejaVuSans-Bold.ttf"),
    (os.path.join(_WINDIR, "Fonts", "arial.ttf"), os.path.join(_WINDIR, "Fonts", "arialbd.ttf")),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]
_fonts: Optional[Tuple[str, str]] = None


def _register_fonts() -> Tuple[str, str]:
    global _fonts
    if _fonts:
        return _fonts
    for regular, bold in FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            pdfmetrics.registerFont(TTFont("TripFont", regular))
            pdfmetrics.registerFont(TTFont("TripFont-Bold", bold))
            _fonts = ("TripFont", "TripFont-Bold")
            return _fonts
    raise RuntimeError("Не найден шрифт с русскими буквами. Положите DejaVuSans.ttf и DejaVuSans-Bold.ttf в папку fonts.")


EMOJI_RE = re.compile("[\U00010000-\U0010FFFF\u2190-\u2BFF\uFE00-\uFE0F\u200d]")


def _clean(text: str) -> str:
    """Убирает эмодзи: в шрифте PDF их нет, вместо них были бы квадратики."""
    return EMOJI_RE.sub("", text or "").translate(str.maketrans({"\u2011": "-", "\u2013": "-", "\u2014": "-"})).strip()


def _esc(text: str) -> str:
    return escape(_clean(text))


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_esc(text).replace("\n", "<br/>"), style)


def _amount(value: Any) -> float:
    """Old saved requests may have no fixed-cost fields."""
    try:
        result = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0 else 0.0


def _money(value: float, currency: str) -> str:
    number = f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", " ")
    return f"{number} {currency}"


def _budget_values(request: Dict[str, Any], itinerary: Itinerary) -> Dict[str, float]:
    program = sum(_amount(item.cost) for day in itinerary.days for item in day.items)
    travel = _amount(request.get("travel_cost"))
    lodging = _amount(request.get("lodging_cost"))
    budget = _amount(request.get("budget"))
    total = travel + lodging + program
    return {"budget": budget, "travel": travel, "lodging": lodging, "program": program,
            "program_budget": budget - travel - lodging, "total": total, "remaining": budget - total}


def _verified_route_points(day: ItineraryDay) -> List[Dict[str, Any]]:
    points = []
    for item in day.items:
        if not is_public_map_point(item):
            continue
        points.append({"number": len(points) + 1, "title": item.title, "time": item.time,
                       "lat": item.lat, "lon": item.lon, **item_map_attribution(item)})
    return points


class _RouteScheme(Flowable):
    """Offline vector sketch. Geographic positions, no invented street map."""

    def __init__(self, points: List[Dict[str, Any]], font: str, width: float, height: float = 135):
        super().__init__()
        self.points, self.font = points, font
        self.width, self.height = width, height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#F1F6F8"))
        canvas.setStrokeColor(colors.HexColor("#D0DFE5"))
        canvas.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=1)
        pad = 21
        inner_w, inner_h = self.width - 2 * pad, self.height - 2 * pad
        canvas.setLineWidth(0.35)
        for step in range(1, 4):
            canvas.line(pad + inner_w * step / 4, pad, pad + inner_w * step / 4, self.height - pad)
            canvas.line(pad, pad + inner_h * step / 4, self.width - pad, pad + inner_h * step / 4)

        mean_lat = sum(point["lat"] for point in self.points) / len(self.points)
        longitude_scale = max(0.01, math.cos(math.radians(mean_lat)))
        geographic = [(point["lon"] * longitude_scale, point["lat"]) for point in self.points]
        x_min, x_max = min(x for x, _ in geographic), max(x for x, _ in geographic)
        y_min, y_max = min(y for _, y in geographic), max(y for _, y in geographic)
        scale = min(inner_w / max(x_max - x_min, 0.0001), inner_h / max(y_max - y_min, 0.0001))
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        plotted = [(self.width / 2 + (x - center_x) * scale,
                    self.height / 2 + (y - center_y) * scale) for x, y in geographic]
        canvas.setStrokeColor(colors.HexColor("#78959F"))
        canvas.setLineWidth(1.6)
        canvas.setDash(3, 3)
        for (x1, y1), (x2, y2) in zip(plotted, plotted[1:]):
            canvas.line(x1, y1, x2, y2)
        canvas.setDash()
        # Offset exact/near-exact duplicate points so every visit number stays readable.
        used = []
        for point, (x, y) in zip(self.points, plotted):
            original_x, original_y = x, y
            attempt = 0
            while any(math.hypot(x - other_x, y - other_y) < 19 for other_x, other_y in used) and attempt < 12:
                attempt += 1
                angle = attempt * 2.4
                radius = 12 + 4 * attempt
                x = min(self.width - 10, max(10, original_x + math.cos(angle) * radius))
                y = min(self.height - 10, max(10, original_y + math.sin(angle) * radius))
            if attempt:
                canvas.setStrokeColor(colors.HexColor("#78959F"))
                canvas.line(original_x, original_y, x, y)
            used.append((x, y))
            canvas.setFillColor(colors.HexColor("#286279"))
            canvas.setStrokeColor(colors.white)
            canvas.circle(x, y, 9, fill=1, stroke=1)
            canvas.setFillColor(colors.white)
            canvas.setFont(self.font, 8)
            canvas.drawCentredString(x, y - 3, str(point["number"]))
        canvas.setStrokeColor(colors.HexColor("#617580"))
        canvas.setFillColor(colors.HexColor("#617580"))
        canvas.setFont(self.font, 7)
        canvas.drawString(self.width - 23, self.height - 13, "С")
        canvas.line(self.width - 20, self.height - 31, self.width - 20, self.height - 17)
        canvas.line(self.width - 20, self.height - 17, self.width - 23, self.height - 21)
        canvas.line(self.width - 20, self.height - 17, self.width - 17, self.height - 21)
        canvas.restoreState()


def _day_route(day: ItineraryDay, small: ParagraphStyle, bold: str, width: float) -> List[Any]:
    if day.day_mode == "free":
        return [_p("Свободный день: посещения не запланированы, точки на карте не требуются.", small)]
    points = _verified_route_points(day)
    attractions = sum(item.kind == "attraction" for item in day.items)
    if not points:
        return [_p("Схема маршрута: нет подтверждённых координат. Проверьте места перед поездкой.", small)]
    legend = []
    for point in points:
        url = point["source_url"]
        label = "Источник координат: 2ГИС — карточка организации" if point["provider"] == "2gis" else "Открыть на OpenStreetMap"
        source = (f'<br/><a href="{_esc(url)}" color="#286279">{label}</a>' if url else
                  '<br/>Источник координат требует перепроверки.')
        legend.append(Paragraph(
            f'<font name="{bold}">{point["number"]}. {_esc(point["time"])} - {_esc(point["title"])}</font>'
            + source, small))
        legend.append(Spacer(1, 5))
    table = Table([[_RouteScheme(points, bold, width * 0.43), legend]], colWidths=[width * 0.45, width * 0.55])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (0, 0), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    caption = "Схема расположения мест без географической подложки. Линии показывают порядок посещения, а не пешеходный путь."
    if len(points) < attractions:
        caption += f" Показано {len(points)} из {attractions} мест: у остальных нет подтверждённых координат."
    return [KeepTogether([_p("Схема маршрута дня", small), table, _p(caption, small)])]


def build_trip_pdf(trip: Dict[str, Any], checklist: List[Dict[str, Any]]) -> bytes:
    """Возвращает PDF с маршрутом поездки и (если есть) чек-листом вещей."""
    regular, bold = _register_fonts()
    request = trip.get("request", {})
    itinerary = Itinerary.model_validate(trip["itinerary"])
    currency = request.get("currency", "EUR")
    costs = _budget_values(request, itinerary)
    unpriced_note = city_unpriced_visit_note(itinerary)

    base = ParagraphStyle("base", fontName=regular, fontSize=10, leading=13.5, textColor=colors.black)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12, textColor=colors.HexColor("#444444"))
    title = ParagraphStyle("title", parent=base, fontName=bold, fontSize=18, leading=22, spaceAfter=6)
    heading = ParagraphStyle("heading", parent=base, fontName=bold, fontSize=13, leading=16, spaceBefore=12, spaceAfter=5, keepWithNext=True)
    bold_style = ParagraphStyle("bold", parent=base, fontName=bold)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title=_clean(itinerary.title), author="Family Trip Planner")
    story: list = [_p(_clean(itinerary.title), title)]
    meta = f"{request.get('destination', '')}, {request.get('start_date', '')} - {request.get('end_date', '')}"
    children_ages = request.get("children_ages", [])
    family = f"Взрослых: {request.get('adults', 2)}."
    if children_ages:
        ages = ", ".join(str(age) for age in children_ages)
        family += f" Дети (возраст): {ages}."
    story += [_p(meta, small),
              _p(f"{family} Общий бюджет поездки: {_money(costs['budget'], currency)}", small),
              Spacer(1, 8), _p(itinerary.summary, base)]
    coverage = program_report_text(itinerary)
    if coverage:
        story += [Spacer(1, 8), _p("Программа и свободные дни", heading), _p(coverage, base)]
    if request.get("weather_text"):
        story += [Spacer(1, 6), _p(request["weather_text"], small)]

    budget_rows = [
        ["Дорога туда и обратно" + (" (не указано)" if not costs["travel"] else ""), _money(costs["travel"], currency)],
        ["Жильё за все ночи" + (" (не указано)" if not costs["lodging"] else ""), _money(costs["lodging"], currency)],
        ["Программа на месте (оценка)", f"около {_money(costs['program'], currency)}"],
        ["Общие расходы (оценка)", f"около {_money(costs['total'], currency)}"],
        ["Остаток бюджета" if costs["remaining"] >= 0 else "Превышение бюджета", _money(abs(costs["remaining"]), currency)],
    ]
    budget_table = Table([[_p(label, small), _p(value, small)] for label, value in budget_rows],
                         colWidths=[11.5 * cm, 5.8 * cm])
    budget_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F6F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 3), (-1, 3), 0.5, colors.HexColor("#B9CFD8")),
    ]))
    story += [Spacer(1, 8), budget_table,
              _p(f"На программу после дороги и жилья: {_money(costs['program_budget'], currency)}. "
                 "Если дорога или жильё не указаны, общий итог неполный. Цены программы ориентировочные, на всех участников.", small)]
    if unpriced_note:
        story.append(_p(unpriced_note, small))

    for day in itinerary.days:
        story.append(PageBreak())
        story.append(_p(f"День {day.day} ({day.date}): {_clean(day.theme)}", heading))
        if day.day_mode == "free":
            story += [_p("Свободный режим", bold_style), _p(FREE_DAY_NOTE, base), Spacer(1, 6)]
        if day.base_area:
            story.append(_p(f"База дня: {day.base_area}", base))
        if day.transfer_note:
            story.append(_p(f"Переезд (ориентировочно): {day.transfer_note}", small))
        if day.base_area or day.transfer_note:
            story.append(Spacer(1, 6))
        rows = []
        for item in day.items:
            display_area = _esc(item.area)
            lines = [f"<b>{_esc(item.title)}</b>" + (f" ({display_area})" if display_area else "")]
            if item.description:
                lines.append(_esc(item.description))
            if item.kids_note and children_ages:
                lines.append(f"<i>Для детей: {_esc(item.kids_note)}</i>")
            if item.kind == "attraction" and not is_public_map_point(item):
                lines.append("<i>(место не подтверждено на карте, проверьте перед поездкой)</i>")
            cost = f"~{item.cost:g} {currency}" if item.cost else ""
            if is_unpriced_city_visit(itinerary, item):
                cost = UNPRICED_VISIT_LABEL
            rows.append([_p(item.time, base), Paragraph("<br/>".join(lines).replace("<b>", f'<font name="{bold}">').replace("</b>", "</font>"), base), _p(cost, small)])
        if rows:
            table = Table(rows, colWidths=[1.6 * cm, 12.8 * cm, 2.9 * cm])
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)
        if day.rainy_alternative:
            story += [Spacer(1, 3), _p(f"Если дождь: {day.rainy_alternative}", small)]
        story += [Spacer(1, 8), *_day_route(day, small, bold, doc.width)]

    closing_section = []
    if itinerary.tips:
        closing_section.append(_p("Советы", heading))
        closing_section += [_p(f"- {tip}", base) for tip in itinerary.tips]
    closing_section += [Spacer(1, 8), _p(f"Общие расходы поездки: около {_money(costs['total'], currency)}, "
                                       f"включая программу около {_money(costs['program'], currency)}.", bold_style)]
    if unpriced_note:
        closing_section.append(_p(unpriced_note, small))
    # Keep a normal short advice block with its heading and total instead of
    # orphaning the attribution/last line on an otherwise empty page.
    story.append(KeepTogether(closing_section))

    if checklist:
        story += [PageBreak(), _p("Чек-лист вещей в дорогу", title)]
        categories = list(dict.fromkeys(item["category"] for item in checklist))
        for category in categories:
            story.append(_p(_clean(category), heading))
            for item in checklist:
                if item["category"] == category:
                    mark = "[x]" if item["done"] else "[  ]"
                    story.append(_p(f"{mark}  {item['text']}", base))

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(1.8 * cm, 0.9 * cm, "Семейный планировщик путешествий")
        canvas.drawRightString(A4[0] - 1.8 * cm, 0.9 * cm, f"стр. {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
