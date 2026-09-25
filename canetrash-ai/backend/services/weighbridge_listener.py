"""
weighbridge_listener.py
========================
RS485 Modbus weighbridge reader.

Attempts to talk to a real Modbus RTU (serial) or Modbus TCP scale indicator
via `pymodbus`. If pymodbus is not installed, or the configured serial port /
TCP host is unreachable, transparently falls back to
`hardware_sim.mock_weighbridge.MockWeighbridge` so the pipeline always has a
gross-weight reading available in dev/CI environments.
"""

from __future__ import annotations

import logging

from backend.config import Settings

logger = logging.getLogger("canetrash.backend.weighbridge_listener")

try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient  # type: ignore

    _HAS_PYMODBUS = True
except ImportError:  # pragma: no cover
    _HAS_PYMODBUS = False


class WeighbridgeListener:
    """
    Reads gross weight from a physical Modbus weighbridge indicator, with an
    automatic fallback to the hardware simulator when no real device is
    reachable.

    Usage:
        listener = WeighbridgeListener(settings)
        weight_kg = listener.read_weight_kg()
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self.backend = "simulator"

        if _HAS_PYMODBUS and not settings.use_hardware_simulators:
            try:
                self._client = ModbusSerialClient(
                    port=settings.weighbridge_port,
                    baudrate=settings.weighbridge_baudrate,
                    timeout=1.0,
                )
                connected = self._client.connect()
                if connected:
                    self.backend = "modbus_rtu"
                    logger.info(
                        "Connected to real weighbridge on %s @ %d baud",
                        settings.weighbridge_port,
                        settings.weighbridge_baudrate,
                    )
                else:
                    logger.warning(
                        "Could not open serial port %s; falling back to simulator.",
                        settings.weighbridge_port,
                    )
                    self._client = None
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Weighbridge Modbus connection failed (%s); using simulator.", exc
                )
                self._client = None

        if self._client is None:
            from hardware_sim.mock_weighbridge import MockWeighbridge

            self._simulator = MockWeighbridge()
        else:
            self._simulator = None

    def read_weight_kg(self, truck_hint: str | None = None) -> float:
        """
        Returns the current stable gross weight reading in kilograms.

        `truck_hint` is only used to seed the simulator deterministically per
        truck when running in mock mode; it is ignored by the real backend.
        """
        if self._client is not None:
            try:
                result = self._client.read_holding_registers(
                    address=0, count=2, slave=self.settings.weighbridge_modbus_unit_id
                )
                if result.isError():
                    raise IOError(f"Modbus error response: {result}")
                raw = (result.registers[0] << 16) | result.registers[1]
                weight_kg = raw / 10.0  # indicator reports weight * 10 (0.1kg resolution)
                return round(weight_kg, 1)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to read real weighbridge (%s); falling back to simulator "
                    "for this reading.",
                    exc,
                )
                from hardware_sim.mock_weighbridge import MockWeighbridge

                if self._simulator is None:
                    self._simulator = MockWeighbridge()

        assert self._simulator is not None
        return self._simulator.read_weight_kg(truck_hint=truck_hint)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
