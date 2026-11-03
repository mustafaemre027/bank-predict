"""Tahmin servisinin girdi şeması: 15 model değişkeni için geçerli kategori kümeleri ve sayısal aralıklar.

Değerler src/config.py içindeki CATEGORY_VALUES ve NUMERIC_RANGES sabitlerinden gelir. `duration` ve hedef değişken şemada yoktur,
extra="forbid" nedeniyle gönderilirse reddedilir (FG-3, FG-11).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import CATEGORY_VALUES, NUMERIC_RANGES


def _choices(column: str) -> Any:
    # Geçerli değerler çalışma anında config'ten okunduğu için Literal dinamik kurulur; böylece iki ayrı liste tutulmaz.
    return Literal[tuple(CATEGORY_VALUES[column])]


def _numeric(column: str) -> Any:
    minimum, maximum = NUMERIC_RANGES[column]
    return Field(ge=minimum, le=maximum)


class CustomerInput(BaseModel):
    """Bir müşterinin model girdisi. Alan sırası veri dosyasındaki sırayla aynıdır."""

    model_config = ConfigDict(extra="forbid")

    age: int = _numeric("age")
    job: _choices("job")  # type: ignore[valid-type]  # Literal, config'ten dinamik kuruluyor
    marital: _choices("marital")  # type: ignore[valid-type]
    education: _choices("education")  # type: ignore[valid-type]
    default: _choices("default")  # type: ignore[valid-type]
    balance: int = _numeric("balance")
    housing: _choices("housing")  # type: ignore[valid-type]
    loan: _choices("loan")  # type: ignore[valid-type]
    contact: _choices("contact")  # type: ignore[valid-type]
    day: int = _numeric("day")
    month: _choices("month")  # type: ignore[valid-type]
    campaign: int = _numeric("campaign")
    pdays: int = _numeric("pdays")
    previous: int = _numeric("previous")
    poutcome: _choices("poutcome")  # type: ignore[valid-type]


def _field_name(error: dict[str, Any]) -> str:
    return ".".join(str(part) for part in error["loc"]) or "girdi"


def _describe_error(error: dict[str, Any]) -> str:
    """Tek bir doğrulama hatasını Türkçe ve teknik ayrıntı içermeyen bir cümleye çevirir."""
    name, kind = _field_name(error), error["type"]
    if kind == "missing":
        return f"`{name}` alanı eksik."
    if kind == "extra_forbidden":
        return f"`{name}` alanı kabul edilmiyor. Model yalnızca şemadaki 15 alanı kullanır."
    if kind == "literal_error":
        return f"`{name}` alanı geçersiz. Geçerli değerler: {', '.join(CATEGORY_VALUES[error['loc'][0]])}."
    if name in NUMERIC_RANGES and kind in {"greater_than_equal", "less_than_equal"}:
        minimum, maximum = NUMERIC_RANGES[name]
        return f"`{name}` alanı {minimum} ile {maximum} arasında olmalı."
    if name in NUMERIC_RANGES:
        return f"`{name}` alanı tam sayı olmalı."
    return f"`{name}` alanı geçersiz."


def format_validation_errors(error: ValidationError) -> list[str]:
    """Pydantic hatalarını alan bazında Türkçe mesajlara çevirir. Hata türü, değer ve iz (traceback) mesaja girmez."""
    return [_describe_error(item) for item in error.errors()]


__all__ = ["CustomerInput", "format_validation_errors"]
