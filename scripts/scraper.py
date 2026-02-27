"""
Scraper for az.bilet.com event search.

Endpoint : GET https://az.bilet.com/fealiyyet/axtaris?query=&type=search
Response : {"items": [...130 events...], "slug": "..."}
Output   : data/data.csv  (one row per event)

Usage:
    pip install aiohttp
    python scripts/scraper.py
"""

import asyncio
import csv
import logging
import sys
from pathlib import Path

import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://az.bilet.com"
SEARCH_URL  = BASE_URL + "/fealiyyet/axtaris"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "data.csv"

HEADERS = {
    "accept":           "application/json, text/plain, */*",
    "accept-language":  "en-GB,en-US;q=0.9,en;q=0.8",
    "dnt":              "1",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-origin",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Fetch ─────────────────────────────────────────────────────────────────────
async def fetch_events(query: str = "", event_type: str = "search") -> list[dict]:
    """Return the raw list of event dicts from the API."""
    params = {"query": query, "type": event_type}
    connector = aiohttp.TCPConnector(ssl=False)
    timeout   = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(SEARCH_URL, params=params, headers=HEADERS) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    items = data.get("items", [])
    log.info("Fetched %d events", len(items))
    return items


# ── Transform ─────────────────────────────────────────────────────────────────
def _join(lst: list, key: str, sep: str = " | ") -> str:
    """Pull `key` from each dict in a list and join into a string."""
    return sep.join(str(d[key]) for d in lst if isinstance(d, dict) and key in d)


def _nested_join(lst: list, outer_key: str, inner_key: str, sep: str = " | ") -> str:
    """e.g. activity_categories → outer_key='category', inner_key='name'"""
    parts = []
    for d in lst:
        if isinstance(d, dict) and outer_key in d and isinstance(d[outer_key], dict):
            val = d[outer_key].get(inner_key)
            if val is not None:
                parts.append(str(val))
    return sep.join(parts)


def transform(item: dict) -> dict:
    """Flatten one event dict into a single-level row for CSV."""

    # ── firms (object) ───────────────────────────────────────────────────────
    firms = item.get("firms") or {}

    # ── activity_places (list of {place_id, place: {...}}) ───────────────────
    places = item.get("activity_places") or []
    place  = places[0]["place"] if places and isinstance(places[0], dict) else {}

    # ── activity_categories (list of {category: {...}}) ──────────────────────
    cats = item.get("activity_categories") or []

    # ── activity_option (list of option dicts) ───────────────────────────────
    options = item.get("activity_option") or []

    # ── translations (list) ──────────────────────────────────────────────────
    translations = item.get("translations") or []
    # prefer Turkish translation (first in list) for the richer description
    tr = translations[0] if translations else {}

    # ── country (object) ─────────────────────────────────────────────────────
    country = item.get("country") or {}

    return {
        # Identity
        "id":                       item.get("id"),
        "name":                     item.get("name"),
        "slug":                     item.get("slug"),
        "adapter":                  item.get("adapter"),
        "type":                     item.get("type"),
        "status":                   item.get("status"),
        "is_seated":                item.get("is_seated"),

        # Dates
        "start_date":               item.get("start_date"),
        "end_date":                 item.get("end_date"),
        "created_at":               item.get("created_at"),
        "updated_at":               item.get("updated_at"),

        # Pricing
        "min_price":                item.get("min_price"),
        "max_price":                item.get("max_price"),
        "price_before_discount":    item.get("price_before_discount"),
        "currency":                 item.get("currency"),
        "discount_percent":         item.get("discount_percent"),
        "discount_code_active":     item.get("discount_code_active"),
        "coupon_code_active":       item.get("coupon_code_active"),

        # Quality
        "rating":                   item.get("rating"),

        # Media
        "photo":                    item.get("photo"),
        "banner":                   item.get("banner"),

        # Description (use translated version if richer)
        "description":              tr.get("description") or item.get("description"),
        "important_info":           tr.get("important_info") or item.get("important_info"),

        # Categories (pipe-separated)
        "categories":               _nested_join(cats, "category", "name"),
        "category_slugs":           _nested_join(cats, "category", "slug"),

        # Organizer / Firm
        "firm_id":                  firms.get("id"),
        "firm_name":                firms.get("name"),
        "firm_slug":                firms.get("slug"),
        "firm_website":             firms.get("website"),
        "firm_email":               firms.get("email"),
        "firm_phone":               firms.get("phone"),

        # Venue / Place
        "place_name":               place.get("name"),
        "place_address":            place.get("address"),
        "place_district":           place.get("district"),
        "place_city":               place.get("city"),
        "place_country":            place.get("country"),
        "place_latitude":           place.get("latitude"),
        "place_longitude":          place.get("longitude"),
        "place_online":             place.get("online"),

        # Country
        "country_name":             country.get("name"),
        "country_code":             country.get("code"),

        # Ticket options (pipe-separated names + count)
        "option_count":             len(options),
        "option_names":             _join(options, "name"),

        # URL helper
        "url": f"{BASE_URL}/fealiyyet/{item.get('slug')}",
    }


# ── Save ──────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "id", "name", "slug", "adapter", "type", "status", "is_seated",
    "start_date", "end_date", "created_at", "updated_at",
    "min_price", "max_price", "price_before_discount", "currency",
    "discount_percent", "discount_code_active", "coupon_code_active",
    "rating", "photo", "banner",
    "description", "important_info",
    "categories", "category_slugs",
    "firm_id", "firm_name", "firm_slug", "firm_website", "firm_email", "firm_phone",
    "place_name", "place_address", "place_district", "place_city",
    "place_country", "place_latitude", "place_longitude", "place_online",
    "country_name", "country_code",
    "option_count", "option_names",
    "url",
]


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        log.warning("No data to save.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("Saved %d rows -> %s", len(rows), path)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    items = await fetch_events(query="", event_type="search")
    rows  = [transform(item) for item in items]
    save_csv(rows, OUTPUT_FILE)
    log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
