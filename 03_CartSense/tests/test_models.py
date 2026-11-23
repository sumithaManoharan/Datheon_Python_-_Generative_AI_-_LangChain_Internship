"""Tests for the Wishlist model."""

import pytest
from pydantic import ValidationError

from cartsense.models import Wishlist

BASE = {
    "category": "phone",
    "budget": 25000,
    "years_of_use": 3,
    "warranty_years": 1,
    "priorities": ["battery life"],
}


def test_currency_defaults_to_inr() -> None:
    assert Wishlist.model_validate(BASE).currency == "INR"


def test_currency_accepts_a_three_letter_code() -> None:
    assert Wishlist.model_validate({**BASE, "currency": "USD"}).currency == "USD"


@pytest.mark.parametrize("code", ["rupee", "inr", "IN", "INRR", ""])
def test_bad_currency_is_rejected(code: str) -> None:
    with pytest.raises(ValidationError):
        Wishlist.model_validate({**BASE, "currency": code})
