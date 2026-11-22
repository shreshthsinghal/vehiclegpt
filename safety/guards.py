"""Isolation guards. The AI side of the vehicle may LISTEN, never TALK.

ListenOnlyBus wraps any python-can bus: receiving works, sending raises and is counted.
COUNTERS is reported by every tool and benchmark: tx_attempts and egress_bytes must stay 0.
(On real hardware the CAN controller is also put in listen-only mode: `ip link set can0 type can listen-only on`.)
"""
from __future__ import annotations

COUNTERS = {"tx_attempts": 0, "egress_bytes": 0}


class TransmitBlocked(PermissionError):
    pass


class ListenOnlyBus:
    def __init__(self, bus):
        self._bus = bus

    def recv(self, timeout=None):
        return self._bus.recv(timeout=timeout)

    def send(self, *_a, **_k):
        COUNTERS["tx_attempts"] += 1
        raise TransmitBlocked("VehicleGPT is listen-only: CAN transmit is not permitted")

    def shutdown(self):
        self._bus.shutdown()

    def __iter__(self):
        while True:
            msg = self.recv(timeout=1.0)
            if msg is not None:
                yield msg
