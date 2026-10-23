"""Чек-лист вещей в дорогу: зависит от погоды и возраста детей. Без обращений к модели, поэтому бесплатно и мгновенно."""
from typing import Any, Dict, List, Optional, Tuple

RAIN_MM = 1.0

DOCS = "📄 Документы"
MONEY = "💶 Деньги и связь"
CLOTHES = "👕 Одежда и обувь"
KIDS = "👶 Для детей"
MEDS = "💊 Аптечка"
TECH = "🔌 Техника"


def build_checklist(adults: int, children_ages: List[int], weather: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """Возвращает список пар (категория, вещь)."""
    items: List[Tuple[str, str]] = []

    def add(category: str, *texts: str) -> None:
        items.extend((category, text) for text in texts)

    people = adults + len(children_ages)
    add(DOCS,
        f"Паспорта: {people} шт.",
        "Билеты и бронь жилья (скачать офлайн)",
        "Медицинская страховка на всех",
        "Копии документов в облаке")
    if children_ages:
        add(DOCS, "Документы детей (паспорт или свидетельство)",
            "Согласие на выезд, если ребёнок едет не с обоими родителями")

    add(MONEY,
        "Банковские карты (две разных)",
        "Немного наличных в местной валюте",
        "Роуминг или местная SIM/eSIM",
        "Офлайн-карты города")

    days = weather["days"] if weather and weather.get("days") else []
    if days:
        tmin = min(d["tmin"] for d in days)
        tmax = max(d["tmax"] for d in days)
        rainy = any(d["rain_mm"] >= RAIN_MM for d in days)
        if tmin < 5:
            add(CLOTHES, "Тёплые куртки", "Шапки, шарфы, перчатки", "Тёплая обувь")
        elif tmin < 14:
            add(CLOTHES, "Куртки или ветровки", "Тёплые кофты на слои", "Закрытая удобная обувь")
        else:
            add(CLOTHES, "Лёгкая одежда на каждый день", "Удобная обувь для долгих прогулок")
        if tmax >= 24:
            add(CLOTHES, "Панамы или кепки", "Солнцезащитные очки", "Солнцезащитный крем")
        if rainy:
            add(CLOTHES, "Зонт и дождевики", "Запасная обувь на случай дождя")
    else:
        add(CLOTHES, "Одежда по прогнозу (проверьте погоду перед выездом)", "Удобная обувь для долгих прогулок")
    add(CLOTHES, "Сменная одежда на каждый день", "Пижамы и бельё")

    if children_ages:
        youngest = min(children_ages)
        add(KIDS, "Любимые игрушки или книги в дорогу", "Перекусы и вода для прогулок", "Влажные салфетки")
        if youngest < 3:
            add(KIDS, "Подгузники с запасом", "Коляска или эрго-рюкзак", "Детское питание и бутылочки",
                "Пустышки и средство для купания")
        elif youngest < 7:
            add(KIDS, "Коляска-трость для долгих переходов", "Запасной комплект одежды в рюкзак",
                "Раскраски и фломастеры")
        else:
            add(KIDS, "Планшет или книга с наушниками", "Небольшой рюкзак для ребёнка")
        if any(age >= 13 for age in children_ages):
            add(KIDS, "Зарядки и наушники подростков")

    add(MEDS,
        "Жаропонижающее и обезболивающее",
        "Средство от аллергии",
        "Пластыри, антисептик, градусник",
        "Личные лекарства всех членов семьи")
    if children_ages:
        add(MEDS, "Детские лекарства в нужной дозировке")

    add(TECH,
        "Зарядки и пауэрбанк",
        "Переходник для розеток",
        "Наушники")
    return items