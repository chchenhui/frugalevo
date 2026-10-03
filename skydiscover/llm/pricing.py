"""Token accounting and cost estimation for LLM API calls.

Prices come from ``models.yaml`` next to this file (override the path with the
``SKYDISCOVER_MODELS_YAML`` environment variable).  Nothing here talks to a
provider: it turns the ``usage`` block an API already returned into token
counts and a dollar estimate.

Two things are deliberately kept apart:

* **tokens** are always counted, for every model;
* **cost** is only reported for models listed in ``models.yaml``.

An unlisted model is *unpriced*, not free -- its cost comes back as ``None``
and the summary reports those calls separately, so a missing entry can never
masquerade as a cheap run.

``models.yaml`` is grouped by provider, since the same model can cost
different amounts depending on who serves it (``gpt-5.6-luna`` direct from
OpenAI vs. through OpenRouter)::

    openai:
      - name: gpt-5.6-luna
        input_per_million: 0.20
        output_per_million: 1.20
        cache_read_per_million: 0.02
        cache_write_per_million: 0.25

The provider is inferred from the model's ``api_base`` (see
``provider_from_api_base``); when it cannot be inferred, an unambiguous name
match across all providers is used.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

logger = logging.getLogger("skydiscover.llm")

_DEFAULT_PRICING_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
_PRICING_PATH_ENV = "SKYDISCOVER_MODELS_YAML"
_PER_MILLION = 1_000_000

# The token buckets that actually cost money, in report order.  Each bills at
# its own rate, so a single "prompt tokens" figure would hide where the money
# went.  ``reasoning_tokens`` is deliberately absent: it is a subset of
# ``output_tokens`` and is already billed there.
_BILLABLE_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("input_tokens", "input"),
    ("cache_read_tokens", "cache read"),
    ("cache_write_tokens", "cache write"),
    ("cache_write_1h_tokens", "cache write 1h"),
    ("output_tokens", "output"),
)

# "gpt-5-2025-08-07" -> "gpt-5": dated snapshots bill as their base model.
_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

# api_base substring -> provider section in models.yaml
_API_BASE_PROVIDERS: Tuple[Tuple[str, str], ...] = (
    ("openrouter.ai", "openrouter"),
    ("api.anthropic.com", "anthropic"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("api.deepseek.com", "deepseek"),
    ("api.mistral.ai", "mistral"),
    ("api.cohere.com", "cohere"),
    (".openai.azure.com", "openai"),
    ("api.openai.com", "openai"),
)


def provider_from_api_base(api_base: Optional[str]) -> Optional[str]:
    """Infer the models.yaml provider section from an API base URL."""
    if not api_base:
        return None
    lowered = api_base.lower()
    for needle, provider in _API_BASE_PROVIDERS:
        if needle in lowered:
            return provider
    return None


# ----------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Token counts for one or more API calls, split by how they are billed.

    The four input buckets are disjoint -- ``input_tokens`` holds only the
    tokens charged at the full input rate, with cached reads and cache writes
    counted separately.  ``reasoning_tokens`` is the one exception: it is a
    subset of ``output_tokens``, tracked for visibility, never billed twice.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    @property
    def prompt_tokens(self) -> int:
        """All input-side tokens, cached and uncached."""
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.cache_write_1h_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_write_1h_tokens=self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
        )

    def __sub__(self, other: "Usage") -> "Usage":
        """Field-wise difference, clamped at zero."""
        return Usage(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            cache_read_tokens=max(0, self.cache_read_tokens - other.cache_read_tokens),
            cache_write_tokens=max(0, self.cache_write_tokens - other.cache_write_tokens),
            cache_write_1h_tokens=max(0, self.cache_write_1h_tokens - other.cache_write_1h_tokens),
            reasoning_tokens=max(0, self.reasoning_tokens - other.reasoning_tokens),
            calls=max(0, self.calls - other.calls),
        )

    def to_dict(self) -> Dict[str, int]:
        d = asdict(self)
        d["prompt_tokens"] = self.prompt_tokens
        d["total_tokens"] = self.total_tokens
        return d

    @classmethod
    def from_response(cls, response: Any) -> Optional["Usage"]:
        """Extract usage from an API response object.

        Handles the three shapes SkyDiscover sees: OpenAI Chat Completions
        (``prompt_tokens``), the OpenAI Responses API (``input_tokens``), and
        Anthropic Messages (``cache_creation_input_tokens``).  Returns None
        when the response carries no usage block, as some proxies and
        streaming responses omit it.
        """
        usage = _attr(response, "usage")
        if usage is None:
            return None

        output = _int(_attr(usage, "completion_tokens", "output_tokens"))

        reasoning = 0
        output_details = _attr(usage, "completion_tokens_details", "output_tokens_details")
        if output_details is not None:
            reasoning = _int(_attr(output_details, "reasoning_tokens"))

        cache_read = _int(_attr(usage, "cache_read_input_tokens"))
        cache_write_total = _int(_attr(usage, "cache_creation_input_tokens"))
        is_anthropic = (
            _attr(usage, "cache_read_input_tokens") is not None
            or _attr(usage, "cache_creation_input_tokens") is not None
        )

        if is_anthropic:
            # Anthropic reports input_tokens *excluding* cached tokens, and
            # splits cache writes by TTL under `cache_creation`.
            uncached = _int(_attr(usage, "input_tokens"))
            write_5m, write_1h = cache_write_total, 0
            creation = _attr(usage, "cache_creation")
            if creation is not None:
                write_5m = _int(_attr(creation, "ephemeral_5m_input_tokens"))
                write_1h = _int(_attr(creation, "ephemeral_1h_input_tokens"))
                if write_5m + write_1h == 0:
                    write_5m = cache_write_total
        else:
            # OpenAI-style: prompt_tokens *includes* the cached tokens.
            prompt = _int(_attr(usage, "prompt_tokens", "input_tokens"))
            input_details = _attr(usage, "prompt_tokens_details", "input_tokens_details")
            cache_read = (
                _int(_attr(input_details, "cached_tokens")) if input_details is not None else 0
            )
            cache_read = min(cache_read, prompt)
            uncached = prompt - cache_read
            write_5m = write_1h = 0

        if uncached == 0 and output == 0 and cache_read == 0 and cache_write_total == 0:
            return None

        return cls(
            input_tokens=uncached,
            output_tokens=output,
            cache_read_tokens=cache_read,
            cache_write_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
            reasoning_tokens=min(reasoning, output),
            calls=1,
        )


def _attr(obj: Any, *names: str) -> Any:
    """First present attribute/key among *names*, or None."""
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _price(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price >= 0 else None


# ----------------------------------------------------------------------
# Pricing table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for one model as served by one provider."""

    name: str
    provider: str
    input_per_million: Optional[float] = None
    output_per_million: Optional[float] = None
    cache_read_per_million: Optional[float] = None
    cache_write_per_million: Optional[float] = None
    cache_write_1h_per_million: Optional[float] = None

    @property
    def is_priced(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None

    def rate_for(self, field: str) -> Optional[float]:
        """Per-million rate billed for one ``Usage`` token field.

        Cache rates fall back to the plain input rate when unset, so a table
        entry that omits them still produces a sane (if slightly high) number
        rather than silently dropping those tokens.
        """
        if not self.is_priced:
            return None
        if field == "input_tokens":
            return self.input_per_million
        if field == "output_tokens":
            return self.output_per_million
        if field == "cache_read_tokens":
            return _first_not_none(self.cache_read_per_million, self.input_per_million)
        if field == "cache_write_tokens":
            return _first_not_none(self.cache_write_per_million, self.input_per_million)
        if field == "cache_write_1h_tokens":
            return _first_not_none(
                self.cache_write_1h_per_million,
                self.cache_write_per_million,
                self.input_per_million,
            )
        return None

    def cost(self, usage: Usage) -> Optional[float]:
        """Cost in USD for *usage*, or None if this model has no usable price."""
        if not self.is_priced:
            return None
        billed = sum(
            getattr(usage, field) * (self.rate_for(field) or 0.0)
            for field, _label in _BILLABLE_BUCKETS
        )
        return billed / _PER_MILLION


def _fmt_cost(cost: Optional[float]) -> str:
    """Format a cost, keeping enough precision that cheap calls aren't shown as $0."""
    if cost is None:
        return "unknown"
    if cost and cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def _fmt_rate(rate: Optional[float]) -> str:
    """Format a per-million rate: $0.20, but $0.018 keeps its precision."""
    if rate is None:
        return "?"
    return f"${rate:.2f}" if rate >= 0.01 else f"${rate:g}"


def _fmt_buckets(usage: "Usage") -> str:
    """Nonzero billable buckets as ``input 100, cache read 20, output 5``."""
    parts = [
        f"{label} {getattr(usage, field)}"
        for field, label in _BILLABLE_BUCKETS
        if getattr(usage, field)
    ]
    return ", ".join(parts) if parts else "no billable tokens"


def _fmt_call_bucket(
    usage: "Usage", pricing: Optional["ModelPricing"], field: str, label: str
) -> str:
    """One stable per-call token/rate/cost field, including zero-token buckets."""
    tokens = getattr(usage, field)
    rate = pricing.rate_for(field) if pricing is not None else None
    if rate is None:
        return f"{label}={tokens} (rate unknown)"
    return f"{label}={tokens} (@ {_fmt_rate(rate)}/1M = {_fmt_cost(tokens * rate / _PER_MILLION)})"


def _fmt_call(
    model: str,
    provider: Optional[str],
    usage: "Usage",
    pricing: Optional["ModelPricing"],
) -> str:
    """Format every billable token bucket and its charge for one API response."""
    resolved_provider = pricing.provider if pricing is not None else provider
    label = f"{resolved_provider}/{model}" if resolved_provider else model
    buckets = ", ".join(
        _fmt_call_bucket(usage, pricing, field, label.replace(" ", "_"))
        for field, label in _BILLABLE_BUCKETS
    )
    reasoning = (
        f", reasoning={usage.reasoning_tokens} (included in output)"
        if usage.reasoning_tokens
        else ""
    )
    cost = pricing.cost(usage) if pricing is not None else None
    return (
        f"LLM call {label}: {buckets}{reasoning}; "
        f"total_tokens={usage.total_tokens}, total_cost={_fmt_cost(cost)}"
    )


def _first_not_none(*values: Optional[float]) -> float:
    for value in values:
        if value is not None:
            return value
    return 0.0


class PricingTable:
    """Parsed ``models.yaml``, indexed by (provider, model name)."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, source: str = "<memory>"):
        self.source = source
        self._by_provider: Dict[str, Dict[str, ModelPricing]] = {}

        for provider, entries in (data or {}).items():
            if not isinstance(entries, list):
                continue  # skip scalar top-level keys (version, comments, ...)
            section: Dict[str, ModelPricing] = {}
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                name = str(entry["name"])
                section[name] = ModelPricing(
                    name=name,
                    provider=provider,
                    input_per_million=_price(entry.get("input_per_million")),
                    output_per_million=_price(entry.get("output_per_million")),
                    cache_read_per_million=_price(entry.get("cache_read_per_million")),
                    cache_write_per_million=_price(entry.get("cache_write_per_million")),
                    cache_write_1h_per_million=_price(entry.get("cache_write_1h_per_million")),
                )
            if section:
                self._by_provider[provider] = section

    @property
    def providers(self) -> List[str]:
        return list(self._by_provider)

    def _candidate_names(self, model: str) -> List[str]:
        """Names to try for *model*, most specific first."""
        names = [model]
        undated = _DATE_SUFFIX.sub("", model)
        if undated != model:
            names.append(undated)
        # "openai/gpt-5.6-luna" also billed as the bare "gpt-5.6-luna"
        for name in list(names):
            if "/" in name:
                names.append(name.rsplit("/", 1)[1])
        return names

    def resolve(self, model: str, provider: Optional[str] = None) -> Optional[ModelPricing]:
        """Pricing for *model*, preferring the *provider* section if given."""
        if not model:
            return None
        candidates = self._candidate_names(model)

        if provider and provider in self._by_provider:
            section = self._by_provider[provider]
            for name in candidates:
                if name in section:
                    return section[name]

        # No provider (or no hit there): accept an unambiguous name match.
        for name in candidates:
            matches = [section[name] for section in self._by_provider.values() if name in section]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                _warn_once(
                    f"'{model}' is priced under multiple providers "
                    f"({', '.join(m.provider for m in matches)}) and none matched the "
                    f"resolved provider '{provider}'; billing it as "
                    f"'{matches[0].provider}'"
                )
                return matches[0]
        return None

    def __len__(self) -> int:
        return sum(len(section) for section in self._by_provider.values())


_warned: set = set()
_warned_lock = threading.Lock()


def _warn_once(message: str) -> None:
    with _warned_lock:
        if message in _warned:
            return
        _warned.add(message)
    logger.warning(message)


_table_lock = threading.Lock()
_table: Optional[PricingTable] = None
_table_path: Optional[str] = None


def pricing_path() -> str:
    """Path of the pricing YAML actually in use."""
    return os.environ.get(_PRICING_PATH_ENV) or _DEFAULT_PRICING_PATH


def load_pricing(path: Optional[str] = None, refresh: bool = False) -> PricingTable:
    """Load (and cache) the pricing table.

    A missing or malformed file is not fatal: it yields an empty table, so
    every model is simply unpriced.
    """
    global _table, _table_path
    path = path or pricing_path()
    with _table_lock:
        if _table is not None and _table_path == path and not refresh:
            return _table
        data: Dict[str, Any] = {}
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.debug(f"No pricing file at {path}; costs will be unknown")
        except Exception as e:
            logger.warning(f"Could not parse pricing file {path}: {e}")
        if not isinstance(data, dict):
            logger.warning(f"Pricing file {path} is not a provider->models mapping; ignoring")
            data = {}
        _table = PricingTable(data, source=path)
        _table_path = path
        return _table


def resolve_pricing(
    model: str,
    provider: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[ModelPricing]:
    """Pricing for *model*, or None if it is not in the table."""
    return load_pricing().resolve(model, provider or provider_from_api_base(api_base))


def estimate_cost(
    model: str,
    usage: Usage,
    provider: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[float]:
    """USD cost of *usage* for *model*, or None if the model has no price."""
    pricing = resolve_pricing(model, provider, api_base)
    return pricing.cost(usage) if pricing is not None else None


# ----------------------------------------------------------------------
# Tracker
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCost:
    """What one model cost: the tokens it used, the rates applied, the total."""

    model: str
    provider: Optional[str]
    usage: Usage
    pricing: Optional[ModelPricing]
    cost_usd: Optional[float]

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model

    def rates(self) -> Optional[Dict[str, Optional[float]]]:
        """The per-million rates from models.yaml that produced ``cost_usd``."""
        if self.pricing is None:
            return None
        return {
            "input_per_million": self.pricing.input_per_million,
            "output_per_million": self.pricing.output_per_million,
            "cache_read_per_million": self.pricing.cache_read_per_million,
            "cache_write_per_million": self.pricing.cache_write_per_million,
            "cache_write_1h_per_million": self.pricing.cache_write_1h_per_million,
        }


class CostTracker:
    """Counts tokens per (provider, model); prices them only when asked.

    Nothing is multiplied at record time -- the tracker is a pure token
    counter.  Rates are looked up in ``models.yaml`` when a report is
    produced, so totals reflect the price table as it stands *then*: fix a
    wrong rate, or fill in a missing one, and re-reporting gives the right
    number without re-running anything.

    Keyed by (provider, model) so the same model served by two providers at
    two prices stays two separate line items.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: Dict[Tuple[Optional[str], str], Usage] = {}

    # -- recording (tokens only) ---------------------------------------

    def record(
        self,
        model: str,
        usage: Optional[Usage],
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Add *usage* for *model*. No pricing happens here."""
        if usage is None or usage.calls == 0:
            return
        model = model or "<unknown>"
        provider = provider or provider_from_api_base(api_base)
        key = (provider, model)
        with self._lock:
            self._tokens[key] = self._tokens.get(key, Usage()) + usage

        # Surface the full billable breakdown at INFO for every successful
        # response. SkyDiscover's normal CLI logging enables this logger at
        # INFO, so users see spend while a long run is still in progress.
        # Logging does not affect the accumulated totals.
        pricing = resolve_pricing(model, provider)
        logger.info(_fmt_call(model, provider, usage, pricing))
        if pricing is None:
            _warn_once(
                f"No price for model '{model}'"
                + (f" under provider '{provider}'" if provider else "")
                + f" in {pricing_path()} -- tokens are tracked but cost is unknown. "
                f"Add an entry to record it."
            )

    def record_response(
        self,
        model: str,
        response: Any,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Optional[Usage]:
        """Extract usage from a raw API *response* and record it."""
        try:
            usage = Usage.from_response(response)
        except Exception:  # accounting must never break a generation
            logger.debug("Failed to extract usage from response", exc_info=True)
            return None
        if usage is not None:
            self.record(model, usage, provider=provider, api_base=api_base)
        return usage

    def reset(self) -> None:
        with self._lock:
            self._tokens.clear()

    # -- scoping --------------------------------------------------------

    def snapshot(self) -> Dict[Tuple[Optional[str], str], Usage]:
        """Opaque mark of the current token counts, for use with ``since()``."""
        with self._lock:
            return dict(self._tokens)

    def since(self, snapshot: Dict[Tuple[Optional[str], str], Usage]) -> "CostTracker":
        """A detached tracker holding only what was recorded after *snapshot*.

        Lets one process run several searches and report each one separately,
        instead of every run inheriting the previous run's totals.
        """
        delta = CostTracker()
        with self._lock:
            for key, usage in self._tokens.items():
                remaining = usage - snapshot.get(key, Usage())
                if remaining.calls:
                    delta._tokens[key] = remaining
        return delta

    # -- pricing (on demand) --------------------------------------------

    def costs(self) -> List[ModelCost]:
        """Price the counted tokens against models.yaml, as of right now."""
        with self._lock:
            counted = sorted(self._tokens.items(), key=lambda kv: (kv[0][1], kv[0][0] or ""))
        rows = []
        for (provider, model), usage in counted:
            pricing = resolve_pricing(model, provider)
            rows.append(
                ModelCost(
                    model=model,
                    provider=pricing.provider if pricing else provider,
                    usage=usage,
                    pricing=pricing,
                    cost_usd=pricing.cost(usage) if pricing else None,
                )
            )
        return rows

    @property
    def total_usage(self) -> Usage:
        with self._lock:
            total = Usage()
            for usage in self._tokens.values():
                total = total + usage
            return total

    @property
    def total_cost_usd(self) -> Optional[float]:
        """Cost of everything priced, or None if nothing could be priced.

        None means "unknown", not zero: an unpriced model must never make a
        run look free.
        """
        priced = [row.cost_usd for row in self.costs() if row.cost_usd is not None]
        return sum(priced) if priced else None

    @property
    def unpriced_models(self) -> List[str]:
        return sorted(row.label for row in self.costs() if row.cost_usd is None)

    def to_dict(self) -> Dict[str, Any]:
        rows = self.costs()
        return {
            "currency": "USD",
            "pricing_file": pricing_path(),
            "total": {**self.total_usage.to_dict(), "cost_usd": self.total_cost_usd},
            "models": {
                row.label: {
                    **row.usage.to_dict(),
                    "provider": row.provider,
                    "rates_per_million": row.rates(),
                    "cost_usd": row.cost_usd,
                }
                for row in rows
            },
            "unpriced_models": [row.label for row in rows if row.cost_usd is None],
        }

    def format_summary(self) -> str:
        """Human-readable summary, with every billable bucket broken out.

        Uncached input, cache reads, cache writes and output each bill at a
        different rate, so each gets its own line showing the tokens, the rate
        applied and what it cost.  A single "prompt tokens" figure would hide
        where the money actually went -- cache writes especially, which cost
        *more* than plain input.
        """
        rows = self.costs()
        total = self.total_usage
        if total.calls == 0:
            return "LLM usage: no tracked API calls"

        lines = ["LLM usage:"]
        for row in rows:
            cost = _fmt_cost(row.cost_usd) if row.cost_usd is not None else "cost unknown"
            lines.append(
                f"  {row.label}: {row.usage.calls} calls, "
                f"{row.usage.total_tokens} tokens, {cost}"
            )
            for field, label in _BILLABLE_BUCKETS:
                tokens = getattr(row.usage, field)
                if not tokens:
                    continue
                rate = row.pricing.rate_for(field) if row.pricing is not None else None
                priced = (
                    f" @ {_fmt_rate(rate)} per 1M = {_fmt_cost(tokens * rate / _PER_MILLION)}"
                    if rate is not None
                    else " (rate unknown)"
                )
                lines.append(f"    {label}: {tokens} tokens{priced}")
            if row.usage.reasoning_tokens:
                # A subset of output, not a separate charge -- shown so a run
                # whose spend is mostly reasoning is visible, not surprising.
                lines.append(
                    f"    (reasoning: {row.usage.reasoning_tokens} tokens, "
                    f"already counted in output above)"
                )
        lines.append(
            f"  TOTAL: {total.calls} calls, {_fmt_buckets(total)} "
            f"= {total.total_tokens} tokens, cost {_fmt_cost(self.total_cost_usd)}"
        )
        unpriced = [row.label for row in rows if row.cost_usd is None]
        if unpriced:
            lines.append(f"  (unpriced, add to {pricing_path()}: {', '.join(unpriced)})")
        return "\n".join(lines)


_global_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """The process-wide tracker every LLM backend reports into."""
    return _global_tracker


def record_response(
    model: str,
    response: Any,
    api_base: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Usage]:
    """Record a raw API response against the global tracker.

    Also appends the usage to the enclosing ``collect_usage()`` sink, if any,
    so the calling ``generate()`` can attach it to its ``LLMResponse``.
    """
    usage = _global_tracker.record_response(model, response, provider=provider, api_base=api_base)
    if usage is not None:
        sink = _usage_sink.get()
        if sink is not None:
            sink.append(usage)
    return usage


# Per-call usage collection.
#
# The ContextVar holds a *mutable list* rather than the usage itself, and that
# matters: `asyncio.wait_for` wraps its coroutine in a child Task on Python
# < 3.12, and a child Task gets a *copy* of the context, so a plain
# `ContextVar.set()` inside the API call would never be visible to the
# generate() that started it. The copied context still points at the same list
# object, so appending to it is.
_usage_sink: ContextVar[Optional[List[Usage]]] = ContextVar("skydiscover_usage_sink", default=None)


@contextmanager
def collect_usage() -> Iterator[List[Usage]]:
    """Collect the usage of every API call recorded inside this block.

    Nests safely: an inner block collects only its own calls, and the outer
    sink resumes when it exits.
    """
    sink: List[Usage] = []
    token = _usage_sink.set(sink)
    try:
        yield sink
    finally:
        _usage_sink.reset(token)


def sum_usage(usages: Iterable[Usage]) -> Optional[Usage]:
    """Fold *usages* into one, or None if there were none.

    None means "the provider reported nothing", which is distinct from a
    zero-token call.
    """
    total = None
    for usage in usages:
        total = usage if total is None else total + usage
    return total
