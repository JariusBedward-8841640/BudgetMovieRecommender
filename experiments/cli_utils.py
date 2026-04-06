"""Small CLI helpers shared by experiment runners."""

from __future__ import annotations


def parse_profile_mix(raw: str | None) -> list[str] | None:
    """Parse comma-separated profile names into a list."""
    if not raw:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def ensure_profile_args_valid(profile: str | None, profile_mix: str | None) -> None:
    """Enforce mutually exclusive profile selection arguments."""
    if profile and profile_mix:
        raise ValueError("Use either --profile or --profile-mix, not both.")
