"""Pure validation/disclosure of explicitly authorized source or map free days.

The caller owns the trust boundary: model responses must have these fields
cleared; source and map-candidate plans require distinct explicit contexts.
This report never replaces price, schedule, source, map or duplicate guards.
"""

from datetime import date, timedelta
from typing import Optional

from pydantic import ValidationError

from schemas import Itinerary, LimitedProgramReport, TripRequest


FREE_DAY_NOTE = (
    "Свободный день: посещения не запланированы. Резерв на обед учтён в бюджете; "
    "дополнительные занятия и остальные расходы не включены."
)


def _checked_report(itinerary: Itinerary) -> Optional[LimitedProgramReport]:
    report = itinerary.program_report
    if not isinstance(report, LimitedProgramReport):
        return None
    try:
        # Revalidate assignments / model_copy and older loaded objects too.
        return LimitedProgramReport.model_validate(report.model_dump(warnings=False))
    except (ValidationError, TypeError, ValueError):
        return None


def _base_key(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _report_problems(
    itinerary: Itinerary,
    request: Optional[TripRequest],
    *,
    allow_limited_program: bool,
    allow_city_limited_program: bool = False,
) -> list[str]:
    problems: list[str] = []
    source = getattr(itinerary, "source_schedule", False)
    city = getattr(itinerary, "city_schedule", False)
    if type(source) is not bool or type(city) is not bool or (source and city):
        problems.append("Признаки серверного плана должны быть логическими и не могут противоречить друг другу.")
    if city is True:
        if request is not None and request.destination_kind != "city":
            problems.append("План по городским кандидатам допустим только для городской поездки.")
        if any(day.base_area != "" for day in itinerary.days):
            problems.append("Городской план по карте не может объявлять непроверенную базу или гостиницу.")
    modes = [day.day_mode for day in itinerary.days]
    if any(mode not in ("program", "free") for mode in modes):
        problems.append("В маршруте указан некорректный режим дня.")
    has_free = "free" in modes
    if not has_free and itinerary.program_report is None:
        return problems
    basis = "map" if city is True and source is False else "source" if source is True and city is False else None
    authorized = (basis == "source" and allow_limited_program is True
                  or basis == "map" and allow_city_limited_program is True)
    if not authorized:
        problems.append("Свободные дни требуют явного серверного разрешения для соответствующего источника программы.")
    report = _checked_report(itinerary)
    if report is None:
        problems.append("Нет корректного серверного отчёта об ограниченной программе.")
    elif report.coverage_basis != basis:
        problems.append("Основание отчёта об ограниченной программе не совпадает с серверным планом.")
    if not has_free or "program" not in modes:
        problems.append("Ограниченная программа должна содержать и полные дни с посещениями, и свободные дни.")

    expected_days = request.days_count if request is not None else len(itinerary.days)
    if report is not None:
        selected = sum(item.kind == "attraction" for day in itinerary.days for item in day.items)
        if (
            report.requested_days != expected_days
            or len(itinerary.days) != expected_days
            or report.program_days != modes.count("program")
            or report.free_days != modes.count("free")
            or report.program_days + report.free_days != report.requested_days
            or report.selected_unique_visits != selected
            or report.eligible_unique_visits < selected
        ):
            problems.append("Числа в отчёте об ограниченной программе не совпадают с маршрутом.")

    for index, day in enumerate(itinerary.days):
        if type(day.day) is not int or day.day != index + 1:
            problems.append("Ограниченная программа должна сохранять порядок и номера всех дней поездки.")
        if request is not None:
            expected_date = (date.fromisoformat(request.start_date) + timedelta(days=index)).isoformat()
            if day.date != expected_date:
                problems.append(f"День {index + 1}: дата ограниченной программы не совпадает с анкетой.")
        if day.day_mode != "free":
            if not any(item.kind == "attraction" for item in day.items):
                problems.append(f"День {day.day}: в дне с программой отсутствуют достопримечательности.")
            continue
        kinds = [item.kind for item in day.items]
        if len(kinds) != 2 or kinds.count("meal") != 1 or kinds.count("rest") != 1:
            problems.append(f"День {day.day}: свободный день может содержать только один обед и один отдых.")
        if any(
            item.verified is not False
            or item.lat is not None or item.lon is not None
            or item.place_query or item.official_name or item.provider or item.source_url
            for item in day.items
        ):
            problems.append(f"День {day.day}: свободный день не может содержать места или данные карты.")
        if day.transfer_note:
            problems.append(f"День {day.day}: в свободный день нельзя скрывать переезд.")
        if day.rainy_alternative:
            problems.append(f"День {day.day}: свободный день не может содержать запасное посещение.")
        reference = itinerary.days[index - 1] if index else next(
            (following for following in itinerary.days[1:] if following.day_mode == "program"),
            None,
        )
        if basis != "map" and (not _base_key(day.base_area) or reference is None
                               or _base_key(day.base_area) != _base_key(reference.base_area)):
            problems.append(f"День {day.day}: свободный день должен сохранять подтверждённую базу соседнего дня.")
    return problems


def limited_program_problems(
    itinerary: Itinerary,
    request: TripRequest,
    *,
    allow_limited_program: bool = False,
    allow_city_limited_program: bool = False,
) -> list[str]:
    """Default-deny metadata guard; the explicit context is never model-owned."""
    return _report_problems(itinerary, request, allow_limited_program=allow_limited_program,
                            allow_city_limited_program=allow_city_limited_program)


def authorized_free_days(
    itinerary: Itinerary,
    request: TripRequest,
    *,
    allow_limited_program: bool = False,
    allow_city_limited_program: bool = False,
) -> frozenset[int]:
    if limited_program_problems(itinerary, request, allow_limited_program=allow_limited_program,
                                allow_city_limited_program=allow_city_limited_program):
        return frozenset()
    return frozenset(day.day for day in itinerary.days if day.day_mode == "free")


def rebuild_program_report(
    itinerary: Itinerary,
    request: TripRequest,
    *,
    eligible_unique_visits: Optional[int] = None,
    allow_city_limited_program: bool = False,
) -> None:
    """Rebuild from trusted output/edit only; never from AI metadata.

    Selected visits are counted here; the caller must still reject duplicate
    identities. Existing eligibility is preserved when a trusted edit adds a
    visit. The report is removed when all days have a full program again.
    """
    source = getattr(itinerary, "source_schedule", False)
    city = getattr(itinerary, "city_schedule", False)
    if (type(source) is not bool or type(city) is not bool or source == city
            or city and (allow_city_limited_program is not True or request.destination_kind != "city")):
        raise ValueError("Отчёт можно строить только для однозначно авторизованного серверного плана.")
    basis = "map" if city else "source"
    if city and any(day.base_area != "" for day in itinerary.days):
        raise ValueError("Городской план по карте не может объявлять непроверенную базу или гостиницу.")
    modes = [day.day_mode for day in itinerary.days]
    if any(mode not in ("program", "free") for mode in modes):
        raise ValueError("Некорректный режим дня.")
    if "free" not in modes:
        itinerary.program_report = None
        return
    selected = sum(item.kind == "attraction" for day in itinerary.days for item in day.items)
    old_report = _checked_report(itinerary)
    if old_report is not None and old_report.coverage_basis != basis:
        raise ValueError("Нельзя переносить число проверенных мест между разными основаниями программы.")
    eligible = eligible_unique_visits
    if eligible is None:
        eligible = old_report.eligible_unique_visits if old_report is not None else selected
    if type(eligible) is not int or not 1 <= eligible <= 100:
        raise ValueError("Число проверенных мест должно быть целым числом от 1 до 100.")
    report = LimitedProgramReport(
        coverage_basis=basis,
        requested_days=request.days_count,
        program_days=modes.count("program"),
        free_days=modes.count("free"),
        eligible_unique_visits=max(eligible, selected, old_report.eligible_unique_visits if old_report else 0),
        selected_unique_visits=selected,
    )
    candidate = itinerary.model_copy(update={"program_report": report})
    problems = limited_program_problems(candidate, request, allow_limited_program=source,
                                        allow_city_limited_program=city and allow_city_limited_program is True)
    if problems:
        raise ValueError(" ".join(problems))
    itinerary.program_report = report


def program_report_text(itinerary: Itinerary) -> str:
    """Stable bounded public text; no model-supplied reasons or claims."""
    if getattr(itinerary, "program_report", None) is None:
        return ""
    if _report_problems(itinerary, None, allow_limited_program=True, allow_city_limited_program=True):
        return "Сведения о свободных днях требуют перепроверки."
    report = _checked_report(itinerary)
    assert report is not None
    if report.coverage_basis == "map":
        return (
            f"Подтверждено мест на карте: {report.eligible_unique_visits}. "
            f"Включено: {report.selected_unique_visits}. "
            f"Дней с программой: {report.program_days}. "
            f"Дней свободного режима: {report.free_days}. Даты сохранены. "
            "На остальные даты не удалось составить программу в рамках условий из найденных кандидатов; "
            "это не утверждение, что других мест нет. Карта подтверждает названия и местоположение, "
            "но не условия посещения, часы работы, цены, длительность или наличие помещения. "
            "Резерв на обед учтён во все дни; личные расходы не запланированы."
        )
    return (
        f"Проверено подходящих мест: {report.eligible_unique_visits}. "
        f"Включено: {report.selected_unique_visits}. "
        f"Дней с программой: {report.program_days}. "
        f"Дней свободного режима: {report.free_days}. Даты сохранены. "
        "На оставшиеся даты не удалось подобрать уникальную проверенную программу "
        "в рамках условий; это не утверждение, что других мест нет. "
        "Резерв на обед учтён во все дни; личные расходы не запланированы."
    )
