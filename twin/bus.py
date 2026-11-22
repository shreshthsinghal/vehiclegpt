"""Which CAN 'cable' to use. Same frames, different transport (decision D-10).

- Linux with a vcan0 interface (CI, a Raspberry Pi, a real car): SocketCAN.
- Windows/macOS, or Linux without vcan: UDP multicast on this machine (python-can 'udp_multicast').
Override with env vars VGPT_CAN_INTERFACE / VGPT_CAN_CHANNEL, or --interface/--channel flags.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import can

UDP_GROUP = "239.74.163.2"      # python-can's default IPv4 multicast group


def default_config() -> dict:
    iface = os.environ.get("VGPT_CAN_INTERFACE")
    chan = os.environ.get("VGPT_CAN_CHANNEL")
    if iface:
        return {"interface": iface, "channel": chan or ("vcan0" if iface == "socketcan" else UDP_GROUP)}
    if platform.system() == "Linux" and Path("/sys/class/net/vcan0").exists():
        return {"interface": "socketcan", "channel": "vcan0"}
    return {"interface": "udp_multicast", "channel": UDP_GROUP}


def open_bus(interface: str | None = None, channel: str | None = None, receive_own: bool = False) -> can.BusABC:
    cfg = default_config()
    if interface:
        cfg = {"interface": interface, "channel": channel or cfg["channel"]}
    kw = {"fd": True}
    if cfg["interface"] == "udp_multicast":
        kw["receive_own_messages"] = receive_own
    return can.Bus(**cfg, **kw)
