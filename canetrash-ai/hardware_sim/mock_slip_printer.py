"""
mock_slip_printer.py
=====================
Emulates a thermal ESC/POS slip printer for the inspection receipt.

Rather than sending raw ESC/POS byte sequences to a physical USB/network
printer, this module renders the slip as plain text (as it would appear on
a 42-column thermal roll) and writes it to a local `printed_slips/`
directory, logging the action. Swap `print_slip` internals for a real
`python-escpos` Network/Usb printer call in production.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("canetrash.hardware_sim.mock_slip_printer")

SLIP_WIDTH = 42
OUTPUT_DIR = Path("printed_slips")


class MockSlipPrinter:
    """
    Usage:
        printer = MockSlipPrinter(device="mock://thermal_printer_01")
        printer.print_slip({
            "transaction_code": "TRX-20260925-AB12CD",
            "plate_number": "B9821XYZ",
            "gross_weight_kg": 18500.0,
            "trash_mass_equivalent_pct": 12.4,
            "deduction_pct": 7.0,
            "net_payable_weight_kg": 17205.0,
            "final_payable_amount": 14624250.0,
            "sop_actions": ["APPROVE_STANDARD_UNLOAD"],
            "status": "completed",
        })
    """

    def __init__(self, device: str = "mock://thermal_printer_01", output_dir: Path | None = None) -> None:
        self.device = device
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_slip(self, data: dict[str, Any]) -> Path:
        """Renders and 'prints' (writes to disk + logs) an inspection slip."""
        text = self._render_slip(data)
        filename = f"{data.get('transaction_code', 'UNKNOWN')}.txt"
        out_path = self.output_dir / filename
        out_path.write_text(text, encoding="utf-8")

        logger.info("Slip printed to %s (device=%s)", out_path, self.device)
        return out_path

    def _render_slip(self, data: dict[str, Any]) -> str:
        def line(char: str = "-") -> str:
            return char * SLIP_WIDTH

        def kv(label: str, value: str) -> str:
            return f"{label:<20}{value:>{SLIP_WIDTH - 20}}"

        rows: list[str] = []
        rows.append("CANETRASH-AI MILL INSPECTION".center(SLIP_WIDTH))
        rows.append("Autonomous Quality Control Slip".center(SLIP_WIDTH))
        rows.append(line("="))
        rows.append(kv("Transaction:", str(data.get("transaction_code", "-"))))
        rows.append(kv("Plate No:", str(data.get("plate_number", "-"))))
        rows.append(
            kv("Printed At:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        )
        rows.append(line())
        rows.append(kv("Gross Weight (kg):", f"{data.get('gross_weight_kg', 0):,.1f}"))
        rows.append(
            kv("Trash % (mass-eq):", f"{data.get('trash_mass_equivalent_pct', 0):.2f}%")
        )
        rows.append(kv("Deduction Rate:", f"{data.get('deduction_pct', 0):.2f}%"))
        rows.append(
            kv("Net Payable (kg):", f"{data.get('net_payable_weight_kg', 0):,.1f}")
        )
        rows.append(line())
        rows.append(
            kv("FINAL AMOUNT:", f"{data.get('final_payable_amount', 0):,.2f}")
        )
        rows.append(line())
        sop_actions = data.get("sop_actions") or []
        rows.append("SOP ACTIONS:")
        for action in sop_actions:
            rows.append(f"  - {action}")
        rows.append(line())
        rows.append(kv("Status:", str(data.get("status", "-")).upper()))
        rows.append(line("="))
        rows.append("Thank you.".center(SLIP_WIDTH))
        return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    printer = MockSlipPrinter()
    path = printer.print_slip(
        {
            "transaction_code": "TRX-DEMO-0001",
            "plate_number": "B9821XYZ",
            "gross_weight_kg": 18500.0,
            "trash_mass_equivalent_pct": 12.4,
            "deduction_pct": 7.0,
            "net_payable_weight_kg": 17205.0,
            "final_payable_amount": 14624250.0,
            "sop_actions": ["APPROVE_STANDARD_UNLOAD"],
            "status": "completed",
        }
    )
    print(path.read_text())
