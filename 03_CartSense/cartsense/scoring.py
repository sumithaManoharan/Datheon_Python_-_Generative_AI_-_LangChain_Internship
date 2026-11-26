"""Pure scoring functions: total cost of ownership and value score."""

from statistics import mean
from typing import Any

WEIGHT_PRIORITIES = 0.50
WEIGHT_BUDGET = 0.30
WEIGHT_WARRANTY = 0.20
MAX_RATING = 10.0
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365


def _require_non_negative(**values: float) -> None:
    """Raise ValueError naming the first argument that is negative."""
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")


def tco(
    price: float,
    years_of_use: float,
    yearly_running_cost: float = 0.0,
    yearly_repair_cost: float = 0.0,
    warranty_years: float = 0.0,
    resale_value: float = 0.0,
    yearly_consumables_cost: float = 0.0,
) -> float:
    """Total cost of ownership over the years of use.

    yearly_running_cost is energy and water; yearly_consumables_cost is things like
    detergent, cleaning liquid, filters. Repairs are only counted for the years
    the warranty does not cover.
    """
    _require_non_negative(
        price=price,
        yearly_running_cost=yearly_running_cost,
        yearly_repair_cost=yearly_repair_cost,
        warranty_years=warranty_years,
        resale_value=resale_value,
        yearly_consumables_cost=yearly_consumables_cost,
    )
    if years_of_use <= 0:
        raise ValueError(f"years_of_use must be positive, got {years_of_use}")

    uncovered_years = max(0.0, years_of_use - warranty_years)
    total = (
        price
        + yearly_running_cost * years_of_use
        + yearly_consumables_cost * years_of_use
        + yearly_repair_cost * uncovered_years
        - resale_value
    )
    return round(max(total, 0.0), 2)


def running_cost(
    uses_per_week: float,
    energy_cost_per_use: float = 0.0,
    water_cost_per_use: float = 0.0,
    consumable_cost_per_use: float = 0.0,
    other_yearly_cost: float = 0.0,
    energy_kwh_per_use: float = 0.0,
    electricity_rate_per_kwh: float = 0.0,
    water_litres_per_use: float = 0.0,
    water_rate_per_litre: float = 0.0,
) -> dict[str, float]:
    """Running cost per use, day, week, month and year (purchase price excluded).

    Consumables are things like detergent or cleaning liquid: pack price divided by
    the number of uses per pack gives consumable_cost_per_use.
    """
    _require_non_negative(
        energy_cost_per_use=energy_cost_per_use,
        water_cost_per_use=water_cost_per_use,
        consumable_cost_per_use=consumable_cost_per_use,
        other_yearly_cost=other_yearly_cost,
        energy_kwh_per_use=energy_kwh_per_use,
        electricity_rate_per_kwh=electricity_rate_per_kwh,
        water_litres_per_use=water_litres_per_use,
        water_rate_per_litre=water_rate_per_litre,
    )
    if uses_per_week <= 0:
        raise ValueError(f"uses_per_week must be positive, got {uses_per_week}")

    energy = energy_cost_per_use + energy_kwh_per_use * electricity_rate_per_kwh
    water = water_cost_per_use + water_litres_per_use * water_rate_per_litre
    per_use = energy + water + consumable_cost_per_use
    per_year = per_use * uses_per_week * WEEKS_PER_YEAR + other_yearly_cost
    return {
        "per_use": round(per_use, 2),
        "per_day": round(per_year / DAYS_PER_YEAR, 2),
        "per_week": round(per_year / WEEKS_PER_YEAR, 2),
        "per_month": round(per_year / MONTHS_PER_YEAR, 2),
        "per_year": round(per_year, 2),
    }


def value_score(
    price: float,
    budget: float,
    warranty_offered: float,
    warranty_wanted: float,
    priority_ratings: list[float],
    dealbreaker_hit: bool = False,
) -> float:
    """Score an option from 0 to 100 against the buyer's wishlist.

    priority_ratings holds one 0-10 rating per wishlist priority.
    A dealbreaker always gives 0.
    """
    _require_non_negative(price=price, warranty_offered=warranty_offered)
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if not isinstance(priority_ratings, list | tuple) or not priority_ratings:
        raise ValueError("priority_ratings must be a non-empty list of numbers, e.g. [8, 7]")
    if any(
        isinstance(rating, bool)
        or not isinstance(rating, int | float)
        or not 0 <= rating <= MAX_RATING
        for rating in priority_ratings
    ):
        raise ValueError(f"every rating must be a number between 0 and {MAX_RATING:g}")

    if dealbreaker_hit:
        return 0.0

    # Full marks within budget; the score falls linearly to 0 at double the budget.
    budget_fit = 1.0 if price <= budget else max(0.0, 1.0 - (price - budget) / budget)
    priority_fit = mean(priority_ratings) / MAX_RATING
    warranty_fit = 1.0 if warranty_wanted <= 0 else min(1.0, warranty_offered / warranty_wanted)

    score = 100 * (
        WEIGHT_PRIORITIES * priority_fit
        + WEIGHT_BUDGET * budget_fit
        + WEIGHT_WARRANTY * warranty_fit
    )
    return round(score, 1)


MAX_ITEMS = 10
MAX_ITEM_NAME = 60


def _clean_items(raw: object, label: str, packs: bool) -> list[dict[str, Any]]:
    """Validate a list of accessories (name, price) or consumables (name, pack price, uses)."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_ITEMS:
        raise ValueError(f"{label} must be a list of at most {MAX_ITEMS} items")
    key = "pack_price" if packs else "price"
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"every {label} item needs a name")
        name = name.strip()[:MAX_ITEM_NAME]
        values = [item.get(key)] + ([item.get("uses_per_pack")] if packs else [])
        if any(isinstance(v, bool) or not isinstance(v, int | float) for v in values):
            raise ValueError(f"{name}: {key} and uses_per_pack must be numbers")
        price = float(item[key])
        if price < 0:
            raise ValueError(f"{name}: {key} must not be negative")
        entry: dict[str, Any] = {"name": name, key: price}
        if packs:
            uses = float(item["uses_per_pack"])
            if uses <= 0:
                raise ValueError(f"{name}: uses_per_pack must be positive")
            entry["uses_per_pack"] = uses
        cleaned.append(entry)
    return cleaned


def _line(name: str, per_use: float, uses_per_year: float) -> dict[str, Any]:
    """One running-cost line with its cost per use, month and year."""
    per_year = per_use * uses_per_year
    return {
        "name": name,
        "per_use": round(per_use, 2),
        "per_month": round(per_year / MONTHS_PER_YEAR, 2),
        "per_year": round(per_year, 2),
    }


def ownership_cost(
    product_price: float,
    uses_per_week: float,
    accessories: list[dict[str, Any]] | None = None,
    consumables: list[dict[str, Any]] | None = None,
    energy_kwh_per_use: float = 0.0,
    electricity_rate_per_kwh: float = 0.0,
    water_litres_per_use: float = 0.0,
    water_rate_per_litre: float = 0.0,
    other_yearly_cost: float = 0.0,
) -> dict[str, Any]:
    """Upfront cost (product plus must-buy accessories) and running cost of owning a product.

    accessories are one-time must-buys: [{"name": ..., "price": ...}].
    consumables are bought repeatedly: [{"name": ..., "pack_price": ..., "uses_per_pack": ...}].
    """
    _require_non_negative(
        product_price=product_price,
        energy_kwh_per_use=energy_kwh_per_use,
        electricity_rate_per_kwh=electricity_rate_per_kwh,
        water_litres_per_use=water_litres_per_use,
        water_rate_per_litre=water_rate_per_litre,
        other_yearly_cost=other_yearly_cost,
    )
    if uses_per_week <= 0:
        raise ValueError(f"uses_per_week must be positive, got {uses_per_week}")
    extras = _clean_items(accessories, "accessories", packs=False)
    packs = _clean_items(consumables, "consumables", packs=True)

    uses_per_year = uses_per_week * WEEKS_PER_YEAR
    upfront = product_price + sum(item["price"] for item in extras)
    parts = [(p["name"], p["pack_price"] / p["uses_per_pack"]) for p in packs]
    parts.append(("Electricity", energy_kwh_per_use * electricity_rate_per_kwh))
    parts.append(("Water", water_litres_per_use * water_rate_per_litre))
    parts.append(("Other yearly costs", other_yearly_cost / uses_per_year))
    running = [_line(name, cost, uses_per_year) for name, cost in parts if cost > 0]
    per_use = sum(cost for _, cost in parts)
    per_year = per_use * uses_per_year
    return {
        "upfront": {
            "product": round(product_price, 2),
            "accessories": [
                {"name": item["name"], "price": round(item["price"], 2)} for item in extras
            ],
            "total": round(upfront, 2),
        },
        "running": running,
        "total_running": {
            "per_use": round(per_use, 2),
            "per_week": round(per_year / WEEKS_PER_YEAR, 2),
            "per_month": round(per_year / MONTHS_PER_YEAR, 2),
            "per_year": round(per_year, 2),
        },
        "first_year_total": round(upfront + per_year, 2),
    }
