"""Pure scoring functions: total cost of ownership and value score."""

from statistics import mean

WEIGHT_PRIORITIES = 0.50
WEIGHT_BUDGET = 0.30
WEIGHT_WARRANTY = 0.20
MAX_RATING = 10.0


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
) -> float:
    """Total cost of ownership over the years of use.

    Repairs are only counted for the years the warranty does not cover.
    """
    _require_non_negative(
        price=price,
        yearly_running_cost=yearly_running_cost,
        yearly_repair_cost=yearly_repair_cost,
        warranty_years=warranty_years,
        resale_value=resale_value,
    )
    if years_of_use <= 0:
        raise ValueError(f"years_of_use must be positive, got {years_of_use}")

    uncovered_years = max(0.0, years_of_use - warranty_years)
    total = (
        price
        + yearly_running_cost * years_of_use
        + yearly_repair_cost * uncovered_years
        - resale_value
    )
    return round(max(total, 0.0), 2)


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
    if not priority_ratings:
        raise ValueError("priority_ratings must not be empty")
    if any(not 0 <= rating <= MAX_RATING for rating in priority_ratings):
        raise ValueError(f"every rating must be between 0 and {MAX_RATING:g}")

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
