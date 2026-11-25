"""Tests for the pure scoring functions."""

import pytest

from cartsense.scoring import tco, value_score


def test_tco_adds_running_and_repair_costs() -> None:
    # 3 years, 1 covered by warranty: 2 uncovered years of repairs.
    result = tco(
        price=1000,
        years_of_use=3,
        yearly_running_cost=100,
        yearly_repair_cost=50,
        warranty_years=1,
    )
    assert result == 1000 + 300 + 100


def test_tco_subtracts_resale_and_never_goes_negative() -> None:
    assert tco(price=100, years_of_use=1, resale_value=500) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price": -1, "years_of_use": 2},
        {"price": 100, "years_of_use": 0},
        {"price": 100, "years_of_use": 2, "yearly_running_cost": -5},
    ],
)
def test_tco_rejects_invalid_input(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        tco(**kwargs)


def test_value_score_perfect_option() -> None:
    score = value_score(
        price=900, budget=1000, warranty_offered=2, warranty_wanted=1, priority_ratings=[10, 10]
    )
    assert score == 100.0


def test_value_score_dealbreaker_forces_zero() -> None:
    score = value_score(
        price=900,
        budget=1000,
        warranty_offered=2,
        warranty_wanted=1,
        priority_ratings=[10, 10],
        dealbreaker_hit=True,
    )
    assert score == 0.0


def test_value_score_over_budget_scores_lower() -> None:
    kwargs = {"budget": 1000, "warranty_offered": 1, "warranty_wanted": 1}
    within = value_score(price=1000, priority_ratings=[8], **kwargs)
    over = value_score(price=1400, priority_ratings=[8], **kwargs)
    assert over < within


@pytest.mark.parametrize("ratings", [[], [11], [-1]])
def test_value_score_rejects_bad_ratings(ratings: list[float]) -> None:
    with pytest.raises(ValueError):
        value_score(
            price=1, budget=10, warranty_offered=1, warranty_wanted=1, priority_ratings=ratings
        )


def test_tco_includes_consumables_every_year() -> None:
    # Detergent and cleaning liquid are paid every year, warranty or not.
    result = tco(price=1000, years_of_use=3, warranty_years=3, yearly_consumables_cost=200)
    assert result == 1000 + 600


def test_tco_rejects_negative_consumables() -> None:
    with pytest.raises(ValueError):
        tco(price=1000, years_of_use=3, yearly_consumables_cost=-1)


@pytest.mark.parametrize("ratings", [{"battery": 8}, "78", ["8"], [True], None])
def test_value_score_rejects_non_list_or_non_numeric_ratings(ratings: object) -> None:
    with pytest.raises(ValueError):
        value_score(
            price=1, budget=10, warranty_offered=1, warranty_wanted=1, priority_ratings=ratings
        )


def test_running_cost_breaks_down_by_period() -> None:
    from cartsense.scoring import running_cost

    result = running_cost(
        uses_per_week=7, energy_cost_per_use=5, water_cost_per_use=3, consumable_cost_per_use=10
    )
    assert result["per_use"] == 18.0
    assert result["per_week"] == 126.0
    assert result["per_month"] == 546.0
    assert result["per_year"] == 6552.0
    assert result["per_day"] == 17.95


def test_running_cost_adds_other_yearly_cost() -> None:
    from cartsense.scoring import running_cost

    result = running_cost(uses_per_week=1, other_yearly_cost=1200)
    assert result["per_year"] == 1200.0
    assert result["per_month"] == 100.0


@pytest.mark.parametrize(
    "kwargs",
    [{"uses_per_week": 0}, {"uses_per_week": -1}, {"uses_per_week": 3, "water_cost_per_use": -1}],
)
def test_running_cost_rejects_invalid_input(kwargs: dict[str, float]) -> None:
    from cartsense.scoring import running_cost

    with pytest.raises(ValueError):
        running_cost(**kwargs)


def test_running_cost_from_physical_units() -> None:
    from cartsense.scoring import running_cost

    result = running_cost(
        uses_per_week=7,
        energy_kwh_per_use=1,
        electricity_rate_per_kwh=7,
        water_litres_per_use=10,
        water_rate_per_litre=0.05,
    )
    assert result["per_use"] == 7.5
    assert result["per_year"] == 2730.0


def test_running_cost_rejects_negative_physical_units() -> None:
    from cartsense.scoring import running_cost

    with pytest.raises(ValueError):
        running_cost(uses_per_week=1, water_litres_per_use=-1)
