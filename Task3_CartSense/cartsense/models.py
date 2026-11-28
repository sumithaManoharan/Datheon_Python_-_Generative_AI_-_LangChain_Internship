"""Pydantic models for the saved wishlist."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_LIST_ITEMS = 10
MAX_ITEM_LENGTH = 80
CATEGORY_PATTERN = r"^[a-z0-9][a-z0-9 _-]{0,39}$"


def clean_category(raw: str) -> str:
    """Normalize a category name: lowercase, trimmed, single spaces."""
    return re.sub(r"\s+", " ", raw.strip().lower())


class Wishlist(BaseModel):
    """What the buyer wants, validated before it is saved or scored."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(pattern=CATEGORY_PATTERN)
    budget: float = Field(gt=0, le=10_000_000, description="Maximum spend in local currency.")
    currency: str = Field(default="INR", pattern=r"^[A-Z]{3}$", description="Currency code.")
    years_of_use: float = Field(gt=0, le=50, description="Expected years of ownership.")
    warranty_years: float = Field(ge=0, le=20, description="Minimum warranty wanted.")
    priorities: list[str] = Field(min_length=1, max_length=2)
    dealbreakers: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    specs: dict[str, str] = Field(default_factory=dict, max_length=MAX_LIST_ITEMS)

    @field_validator("category", mode="before")
    @classmethod
    def _clean_category(cls, value: object) -> object:
        """Clean the category before the pattern check runs."""
        return clean_category(value) if isinstance(value, str) else value

    @field_validator("priorities", "dealbreakers")
    @classmethod
    def _clean_items(cls, items: list[str]) -> list[str]:
        """Strip, drop blanks, de-duplicate, and cap each item's length."""
        cleaned: list[str] = []
        for item in items:
            text = item.strip()
            if not text:
                continue
            if len(text) > MAX_ITEM_LENGTH:
                raise ValueError(f"list item longer than {MAX_ITEM_LENGTH} characters")
            if text.lower() not in (c.lower() for c in cleaned):
                cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _priorities_not_empty(self) -> Self:
        """Reject a priorities list that became empty after cleaning."""
        if not self.priorities:
            raise ValueError("priorities must contain at least one non-blank item")
        return self
