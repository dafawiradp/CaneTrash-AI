"""
mock_weighbridge.py
====================
Software simulator for an RS485 Modbus weighbridge scale indicator.

Generates deterministic-per-truck (seeded by plate/truck_hint) but slightly
noisy gross-weight readings representative of loaded sugarcane trucks
(typically 8,000 - 28,000 kg gross depending on vehicle class), with a small
chance of a "settling" transient on the first read to emulate real scale
stabilization behavior.
"""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger("canetrash.hardware_sim.mock_weighbridge")


class MockWeighbridge:
    """
    Usage:
        wb = MockWeighbridge()
        weight_kg = wb.read_weight_kg(truck_hint="B9821XYZ")
    """

    def __init__(
        self,
        min_weight_kg: float = 8_000.0,
        max_weight_kg: float = 28_000.0,
        noise_kg: float = 15.0,
        settling_delay_s: float = 0.0,
    ) -> None:
        self.min_weight_kg = min_weight_kg
        self.max_weight_kg = max_weight_kg
        self.noise_kg = noise_kg
        self.settling_delay_s = settling_delay_s
        self._cache: dict[str, float] = {}

    def read_weight_kg(self, truck_hint: str | None = None) -> float:
        """Returns a stable, plausible gross weight in kg for the given truck."""
        seed_key = truck_hint or "unknown_truck"

        if seed_key not in self._cache:
            rng = random.Random(seed_key)
            base_weight = rng.uniform(self.min_weight_kg, self.max_weight_kg)
            self._cache[seed_key] = base_weight
            if self.settling_delay_s > 0:
                time.sleep(self.settling_delay_s)

        base = self._cache[seed_key]
        noisy = base + random.uniform(-self.noise_kg, self.noise_kg)
        weight = round(max(0.0, noisy), 1)
        logger.debug("Mock weighbridge reading for %s: %.1f kg", seed_key, weight)
        return weight


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    wb = MockWeighbridge()
    for _ in range(3):
        print(wb.read_weight_kg("B9821XYZ"))
