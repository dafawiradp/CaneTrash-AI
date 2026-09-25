"""
slm_agent.py
============
Local Small Language Model (SLM) reasoning agent.

Responsibilities:
  1. Compute the auto-refaction price deduction from trash % (density-weighted
     mass-equivalent, from Lidar3DEngine) against configurable mill tariff
     rules.
  2. Decide Mill Station SOP adjustments (e.g. "route to manual re-inspection",
     "flag for washing station", "reject load") based on thresholds.
  3. Produce a natural-language audit summary suitable for the operator
     dashboard AI terminal and the printed slip.

Backend: attempts to call a local Ollama server (default model "qwen2.5:3b")
or an OpenAI-compatible vLLM endpoint for the natural-language summary only.
The numeric refaction/SOP decision is ALWAYS computed deterministically by
rule-based logic first (this is a financial calculation and must be
auditable/reproducible) -- the SLM is only used to phrase the human-readable
explanation. If no local LLM endpoint is reachable, a template-based summary
generator is used instead, so the agent works fully offline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("canetrash.ai_core.slm_agent")

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False


# --------------------------------------------------------------------------- #
# Mill tariff / SOP configuration (defaults; override via constructor)
# --------------------------------------------------------------------------- #
DEFAULT_TARIFF_TABLE: list[dict[str, float]] = [
    # trash_pct upper bound (exclusive) -> deduction rate applied to price/ton
    {"max_trash_pct": 5.0, "deduction_pct": 0.0},
    {"max_trash_pct": 10.0, "deduction_pct": 3.0},
    {"max_trash_pct": 15.0, "deduction_pct": 7.0},
    {"max_trash_pct": 20.0, "deduction_pct": 12.0},
    {"max_trash_pct": 30.0, "deduction_pct": 20.0},
    {"max_trash_pct": 100.0, "deduction_pct": 35.0},
]

SOP_REJECT_THRESHOLD_PCT = 40.0
SOP_MANUAL_REVIEW_THRESHOLD_PCT = 25.0
SOP_WASH_STATION_ROTTEN_PCT = 8.0  # rotten_stalk mass-equivalent share trigger


@dataclass
class RefactionDecision:
    """Final auto-refaction + SOP decision for a single inspected load."""

    transaction_id: str
    trash_mass_equivalent_pct: float
    gross_weight_kg: float
    base_price_per_ton: float
    deduction_pct: float
    deducted_weight_kg: float
    net_payable_weight_kg: float
    price_deduction_amount: float
    final_payable_amount: float
    sop_actions: list[str]
    audit_summary: str
    summary_backend: str  # "ollama", "vllm", or "template"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "trash_mass_equivalent_pct": round(self.trash_mass_equivalent_pct, 2),
            "gross_weight_kg": round(self.gross_weight_kg, 2),
            "base_price_per_ton": round(self.base_price_per_ton, 2),
            "deduction_pct": round(self.deduction_pct, 2),
            "deducted_weight_kg": round(self.deducted_weight_kg, 2),
            "net_payable_weight_kg": round(self.net_payable_weight_kg, 2),
            "price_deduction_amount": round(self.price_deduction_amount, 2),
            "final_payable_amount": round(self.final_payable_amount, 2),
            "sop_actions": self.sop_actions,
            "audit_summary": self.audit_summary,
            "summary_backend": self.summary_backend,
        }


class SLMAgent:
    """
    Reasoning agent orchestrating refaction pricing, SOP dispatch, and
    natural-language audit summaries.

    Usage:
        agent = SLMAgent()
        decision = agent.evaluate(
            transaction_id="TRX-2026-0001",
            trash_mass_equivalent_pct=12.4,
            gross_weight_kg=18500,
            base_price_per_ton=850000,
            class_breakdown={"rotten_stalk": 0.06, ...},
        )
    """

    def __init__(
        self,
        tariff_table: list[dict[str, float]] | None = None,
        ollama_host: str | None = None,
        ollama_model: str = "qwen2.5:3b",
        vllm_base_url: str | None = None,
        request_timeout_s: float = 4.0,
    ) -> None:
        self.tariff_table = sorted(
            tariff_table or DEFAULT_TARIFF_TABLE, key=lambda r: r["max_trash_pct"]
        )
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model
        self.vllm_base_url = vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.request_timeout_s = request_timeout_s

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #
    def evaluate(
        self,
        transaction_id: str,
        trash_mass_equivalent_pct: float,
        gross_weight_kg: float,
        base_price_per_ton: float,
        class_breakdown: dict[str, float] | None = None,
    ) -> RefactionDecision:
        deduction_pct = self._lookup_deduction_pct(trash_mass_equivalent_pct)
        sop_actions = self._determine_sop_actions(trash_mass_equivalent_pct, class_breakdown or {})

        deducted_weight_kg = gross_weight_kg * (deduction_pct / 100.0)
        net_payable_weight_kg = gross_weight_kg - deducted_weight_kg

        base_amount = (gross_weight_kg / 1000.0) * base_price_per_ton
        final_amount = (net_payable_weight_kg / 1000.0) * base_price_per_ton
        price_deduction_amount = base_amount - final_amount

        summary, backend = self._generate_summary(
            transaction_id=transaction_id,
            trash_pct=trash_mass_equivalent_pct,
            deduction_pct=deduction_pct,
            deducted_weight_kg=deducted_weight_kg,
            price_deduction_amount=price_deduction_amount,
            sop_actions=sop_actions,
            class_breakdown=class_breakdown or {},
        )

        return RefactionDecision(
            transaction_id=transaction_id,
            trash_mass_equivalent_pct=trash_mass_equivalent_pct,
            gross_weight_kg=gross_weight_kg,
            base_price_per_ton=base_price_per_ton,
            deduction_pct=deduction_pct,
            deducted_weight_kg=deducted_weight_kg,
            net_payable_weight_kg=net_payable_weight_kg,
            price_deduction_amount=price_deduction_amount,
            final_payable_amount=final_amount,
            sop_actions=sop_actions,
            audit_summary=summary,
            summary_backend=backend,
        )

    # --------------------------------------------------------------- #
    # Deterministic pricing / SOP logic (never delegated to the LLM)
    # --------------------------------------------------------------- #
    def _lookup_deduction_pct(self, trash_pct: float) -> float:
        for row in self.tariff_table:
            if trash_pct < row["max_trash_pct"]:
                return row["deduction_pct"]
        return self.tariff_table[-1]["deduction_pct"]

    def _determine_sop_actions(
        self, trash_pct: float, class_breakdown: dict[str, float]
    ) -> list[str]:
        actions: list[str] = []

        if trash_pct >= SOP_REJECT_THRESHOLD_PCT:
            actions.append("REJECT_LOAD")
            actions.append("NOTIFY_SUPPLIER")
        elif trash_pct >= SOP_MANUAL_REVIEW_THRESHOLD_PCT:
            actions.append("ROUTE_MANUAL_REINSPECTION")

        rotten_share = class_breakdown.get("rotten_stalk", 0.0) * 100
        if rotten_share >= SOP_WASH_STATION_ROTTEN_PCT:
            actions.append("FLAG_WASH_STATION")

        if not actions:
            actions.append("APPROVE_STANDARD_UNLOAD")

        return actions

    # --------------------------------------------------------------- #
    # Natural language summary (LLM-assisted, template fallback)
    # --------------------------------------------------------------- #
    def _generate_summary(
        self,
        transaction_id: str,
        trash_pct: float,
        deduction_pct: float,
        deducted_weight_kg: float,
        price_deduction_amount: float,
        sop_actions: list[str],
        class_breakdown: dict[str, float],
    ) -> tuple[str, str]:
        prompt = self._build_prompt(
            transaction_id,
            trash_pct,
            deduction_pct,
            deducted_weight_kg,
            price_deduction_amount,
            sop_actions,
            class_breakdown,
        )

        if _HAS_HTTPX:
            if self.vllm_base_url:
                text = self._try_vllm(prompt)
                if text:
                    return text, "vllm"
            text = self._try_ollama(prompt)
            if text:
                return text, "ollama"

        return self._template_summary(
            transaction_id, trash_pct, deduction_pct, deducted_weight_kg,
            price_deduction_amount, sop_actions,
        ), "template"

    def _build_prompt(
        self,
        transaction_id: str,
        trash_pct: float,
        deduction_pct: float,
        deducted_weight_kg: float,
        price_deduction_amount: float,
        sop_actions: list[str],
        class_breakdown: dict[str, float],
    ) -> str:
        return (
            "You are a sugar mill quality control auditor. Write a concise, "
            "2-3 sentence professional audit note (no markdown) for a truck "
            "inspection record. Do not invent numbers beyond what is given.\n\n"
            f"Transaction ID: {transaction_id}\n"
            f"Trash mass-equivalent percentage: {trash_pct:.2f}%\n"
            f"Class breakdown (area ratio): {json.dumps(class_breakdown)}\n"
            f"Applied deduction rate: {deduction_pct:.2f}%\n"
            f"Deducted weight: {deducted_weight_kg:.1f} kg\n"
            f"Price deduction amount: {price_deduction_amount:,.2f}\n"
            f"SOP actions triggered: {', '.join(sop_actions)}\n"
        )

    def _try_ollama(self, prompt: str) -> str | None:
        try:
            resp = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama backend unreachable/failed (%s); falling back.", exc)
            return None

    def _try_vllm(self, prompt: str) -> str | None:
        try:
            resp = httpx.post(
                f"{self.vllm_base_url}/v1/chat/completions",
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 220,
                },
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("vLLM backend unreachable/failed (%s); falling back.", exc)
            return None

    def _template_summary(
        self,
        transaction_id: str,
        trash_pct: float,
        deduction_pct: float,
        deducted_weight_kg: float,
        price_deduction_amount: float,
        sop_actions: list[str],
    ) -> str:
        action_text = ", ".join(a.replace("_", " ").title() for a in sop_actions)
        return (
            f"Inspection {transaction_id} recorded a mass-equivalent trash content of "
            f"{trash_pct:.2f}%, triggering a {deduction_pct:.2f}% refaction deduction "
            f"({deducted_weight_kg:.0f} kg, equivalent to {price_deduction_amount:,.2f} "
            f"in price deduction). Recommended action: {action_text}."
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    agent = SLMAgent()
    decision = agent.evaluate(
        transaction_id="TRX-DEMO-0001",
        trash_mass_equivalent_pct=17.8,
        gross_weight_kg=18250,
        base_price_per_ton=850000,
        class_breakdown={
            "clean_cane": 0.70,
            "dry_leaf": 0.12,
            "green_top": 0.08,
            "soil_dirt": 0.06,
            "rotten_stalk": 0.04,
        },
    )
    print(json.dumps(decision.to_dict(), indent=2))
