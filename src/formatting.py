"""Rapor ve grafiklerde sayıları Türkçe biçimde (binlik ayırıcı nokta, ondalık ayırıcı virgül) yazar."""
from __future__ import annotations

P_VALUE_FLOOR = 0.001


def format_number(value: float, decimals: int = 0) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_percent(ratio: float, decimals: int = 2) -> str:
    return f"%{format_number(ratio * 100, decimals)}"


def join_names(names: list[str]) -> str:
    """Listeyi Türkçe sıralama biçiminde birleştirir: "a, b ve c"."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} ve {names[-1]}"


def format_p_value(p_value: float) -> str:
    return "< 0,001" if p_value < P_VALUE_FLOOR else format_number(p_value, 3)


__all__ = ["P_VALUE_FLOOR", "format_number", "format_p_value", "format_percent", "join_names"]
