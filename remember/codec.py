"""Compact time-series encoding for the WARM tier (inspired by Facebook's Gorilla, VLDB 2015).

Timestamps: delta-of-delta (a steady 1 Hz stream becomes a run of zeros).
Values:     quantised to the signal's own resolution (e.g. 0.01 km/h from the DBC), so it is LOSSLESS for
            vehicle data, then delta-encoded (slowly changing signals become small numbers / zeros).
Both are packed in the smallest integer type that fits and compressed with zlib.
Optional AES-256-GCM encryption of the whole block (data at rest).
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

HDR = struct.Struct("<qiidBB")       # t0, n, first_dt, quantum, ts_dtype, val_dtype
_DT = [np.int8, np.int16, np.int32, np.int64]


def _pack_small(x: np.ndarray) -> tuple[int, bytes]:
    m = int(np.abs(x).max()) if x.size else 0
    for code, dt in enumerate(_DT):
        if m <= np.iinfo(dt).max:
            return code, x.astype(dt).tobytes()
    raise OverflowError


def encode(ts: np.ndarray, vals: np.ndarray, quantum: float) -> bytes:
    ts = np.asarray(ts, dtype=np.int64)
    n = ts.size
    q = np.rint(np.asarray(vals, dtype=np.float64) / quantum).astype(np.int64)
    d = np.diff(ts)
    first_dt = int(d[0]) if n > 1 else 0
    dod = np.diff(d, prepend=first_dt) if n > 1 else np.zeros(0, np.int64)
    dv = np.diff(q, prepend=0)                          # first element stores the absolute value
    tc, tb = _pack_small(dod)
    vc, vb = _pack_small(dv)
    body = HDR.pack(int(ts[0]) if n else 0, n, first_dt, quantum, tc, vc) + tb + vb
    return zlib.compress(body, 6)


def decode(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    body = zlib.decompress(blob)
    t0, n, first_dt, quantum, tc, vc = HDR.unpack_from(body)
    off = HDR.size
    tdt, vdt = _DT[tc], _DT[vc]
    nd = max(n - 1, 0)
    dod = np.frombuffer(body, tdt, nd, off).astype(np.int64)
    off += nd * np.dtype(tdt).itemsize
    dv = np.frombuffer(body, vdt, n, off).astype(np.int64)
    d = first_dt + np.cumsum(dod) if nd else np.zeros(0, np.int64)
    ts = np.concatenate([[t0], t0 + np.cumsum(d)]) if n else np.zeros(0, np.int64)
    return ts, np.cumsum(dv) * quantum


def encode_rollup(minutes: np.ndarray, mn, mx, mean, cnt) -> bytes:
    """COLD tier: one-minute aggregates (min/max/mean/count) as float32 + compressed."""
    arr = np.stack([minutes.astype(np.float64), mn, mx, mean, cnt]).astype(np.float64)
    body = struct.pack("<i", minutes.size) + arr[0].astype(np.int64).tobytes() + arr[1:].astype(np.float32).tobytes()
    return zlib.compress(body, 6)


def decode_rollup(blob: bytes):
    body = zlib.decompress(blob)
    n = struct.unpack_from("<i", body)[0]
    minutes = np.frombuffer(body, np.int64, n, 4)
    rest = np.frombuffer(body, np.float32, 4 * n, 4 + 8 * n).reshape(4, n).astype(np.float64)
    return minutes, rest[0], rest[1], rest[2], rest[3]


# --------------------------------------------------------------------------- encryption at rest
class Cipher:
    """AES-256-GCM. In a vehicle the key lives in the HSM / TPM; here a key file stands in for it."""

    def __init__(self, key: bytes | None):
        self.aes = None
        if key:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self.aes = AESGCM(key)

    def seal(self, data: bytes, aad: bytes = b"") -> bytes:
        if not self.aes:
            return data
        nonce = os.urandom(12)
        return nonce + self.aes.encrypt(nonce, data, aad)

    def open(self, data: bytes, aad: bytes = b"") -> bytes:
        if not self.aes:
            return data
        return self.aes.decrypt(data[:12], data[12:], aad)
