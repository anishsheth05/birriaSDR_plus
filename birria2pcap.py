#!/usr/bin/env python3
"""
birria2pcap.py - decode 802.11 frames from a BirriaSDR I/Q recording into a pcap.

Input
    The file written by `birria_rx`: little-endian interleaved signed 16-bit I,Q
    pairs (cs16), 14-bit values in -8192..8191. Raw complex float32 (cf32) is
    also accepted. Record at 80 MSps with e.g.

        ./build/birria_rx --rate 80000000 --frequency 2437000000 cap.cs16 80000000 manual

What it decodes
    The capture is split into 20 MHz Wi-Fi channels in the frequency domain, and
    each channel is run through two receivers:

    * OFDM (20 MHz): 802.11a/g legacy frames (6-54 Mbps), and 802.11n HT-mixed
      frames with one spatial stream, BCC, no STBC, MCS 0-7, long or short GI.
      A-MPDUs are split into their MPDUs.
    * DSSS/CCK: 802.11b, long and short preamble, 1, 2, 5.5 and 11 Mbps.

    Not decoded (detected, counted, skipped): 40/80 MHz PPDUs, VHT, HE, MCS > 7,
    LDPC, STBC, greenfield, PBCC. There is no multipath equalizer for DSSS.

Output
    pcap (nanosecond timestamps) with a radiotap header per frame: flags (FCS
    present), rate or HT MCS, channel frequency, and dB_antsignal/dB_antnoise.
    The two dB values are relative to digital full scale plus 100 (so 60 means
    -40 dBFS); BirriaSDR gain is uncalibrated, so they only compare frames
    within one capture at one gain setting. Only frames that pass the FCS are
    written unless --keep-bad-fcs is given.

Dependencies: numpy, scipy.
"""

import argparse
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.fft as sfft

# ---------------------------------------------------------------------------
# Bit helpers, CRCs, scramblers
# ---------------------------------------------------------------------------


def bits_to_int(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= int(b) << i
    return v


def int_to_bits(v, n):
    return np.array([(v >> i) & 1 for i in range(n)], np.uint8)


def bits_to_bytes(bits):
    return np.packbits(np.asarray(bits, np.uint8), bitorder="little").tobytes()


def crc_bits_msb(bits, width, poly, init):
    """Shift-register CRC fed one bit at a time, MSB-first register, ones-complement output."""
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    reg = init
    for b in bits:
        fb = (1 if reg & top else 0) ^ int(b)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg ^ mask


def crc8(bits):  # HT-SIG and A-MPDU delimiter: x^8 + x^2 + x + 1
    return crc_bits_msb(bits, 8, 0x07, 0xFF)


def crc16(bits):  # 802.11b PLCP header: x^16 + x^12 + x^5 + 1
    return crc_bits_msb(bits, 16, 0x1021, 0xFFFF)


def msb_bits(v, n):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], np.uint8)


def lfsr_continue(prev7, n):
    """Continue the x^7 + x^4 + 1 sequence: x[k] = x[k-7] ^ x[k-4]."""
    b = [int(v) for v in prev7]
    for _ in range(n):
        b.append(b[-7] ^ b[-4])
    return np.array(b[7:], np.uint8)


def ofdm_descramble(bits):
    """The first 7 SERVICE bits are zero before scrambling, so they are the scrambler sequence."""
    head = bits[:7]
    period = np.concatenate([head, lfsr_continue(head, 120)])  # period 127
    reps = -(-len(bits) // 127)
    return bits ^ np.tile(period, reps)[: len(bits)]


def dsss_descramble(z):
    """802.11b self-synchronizing descrambler: b[n] = z[n] ^ z[n-4] ^ z[n-7]."""
    z = np.asarray(z, np.uint8)
    out = z.copy()
    out[4:] ^= z[:-4]
    out[7:] ^= z[:-7]
    return out


def fcs_ok(mpdu):
    return len(mpdu) >= 14 and zlib.crc32(mpdu[:-4]) == struct.unpack("<I", mpdu[-4:])[0]


# ---------------------------------------------------------------------------
# OFDM tables
# ---------------------------------------------------------------------------

LTF_L = np.array(
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
     1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1],
    float)
assert len(LTF_L) == 53
LTF64 = np.zeros(64)
for _k in range(-26, 27):
    LTF64[_k % 64] = LTF_L[_k + 26]
HTLTF64 = LTF64.copy()
HTLTF64[-28 % 64], HTLTF64[-27 % 64], HTLTF64[27], HTLTF64[28] = 1, 1, -1, -1

PILOT_K = np.array([-21, -7, 7, 21])
PILOT_IDX = PILOT_K % 64
PILOT_PSI = np.array([1, 1, 1, -1], float)
PILOT_POL = 1 - 2 * lfsr_continue([1] * 7, 127).astype(float)

DATA_K_L = np.array([k for k in range(-26, 27) if k not in (0, -21, -7, 7, 21)])
DATA_K_HT = np.array([k for k in range(-28, 29) if k not in (0, -21, -7, 7, 21)])

# rate code (R1 is bit 0) -> (Mbps, bits per subcarrier, coding rate)
LEGACY_RATES = {
    0b1011: (6, 1, Fraction(1, 2)), 0b1111: (9, 1, Fraction(3, 4)),
    0b1010: (12, 2, Fraction(1, 2)), 0b1110: (18, 2, Fraction(3, 4)),
    0b1001: (24, 4, Fraction(1, 2)), 0b1101: (36, 4, Fraction(3, 4)),
    0b1000: (48, 6, Fraction(2, 3)), 0b1100: (54, 6, Fraction(3, 4)),
}
HT_MCS = [(1, Fraction(1, 2)), (2, Fraction(1, 2)), (2, Fraction(3, 4)), (4, Fraction(1, 2)),
          (4, Fraction(3, 4)), (6, Fraction(2, 3)), (6, Fraction(3, 4)), (6, Fraction(5, 6))]

# Puncturing keep-masks over the encoder output A0 B0 A1 B1 ...
PUNCTURE = {
    Fraction(1, 2): np.array([1, 1], bool),
    Fraction(2, 3): np.array([1, 1, 1, 0], bool),
    Fraction(3, 4): np.array([1, 1, 1, 0, 0, 1], bool),
    Fraction(5, 6): np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1], bool),
}


def interleave_perm(ncbps, nbpsc, ht):
    """perm[k] = position of coded bit k after interleaving (one symbol, one stream)."""
    k = np.arange(ncbps)
    s = max(nbpsc // 2, 1)
    if ht:
        ncol, nrow = 13, 4 * nbpsc
        i = nrow * (k % ncol) + k // ncol
        j = s * (i // s) + (i + ncbps - (ncol * i) // ncbps) % s
    else:
        i = (ncbps // 16) * (k % 16) + k // 16
        j = s * (i // s) + (i + ncbps - (16 * i) // ncbps) % s
    return j


def demap(z, nbpsc):
    """Soft bits (positive -> 1) for 802.11 Gray-mapped BPSK/QPSK/16/64-QAM, per subcarrier."""
    if nbpsc == 1:
        return z.real[:, None]
    if nbpsc == 2:
        s = math.sqrt(2)
        return np.stack([z.real * s, z.imag * s], 1)
    if nbpsc == 4:
        s = math.sqrt(10)
        i, q = z.real * s, z.imag * s
        return np.stack([i, 2 - np.abs(i), q, 2 - np.abs(q)], 1)
    s = math.sqrt(42)
    i, q = z.real * s, z.imag * s
    return np.stack([i, 4 - np.abs(i), 2 - np.abs(np.abs(i) - 4),
                     q, 4 - np.abs(q), 2 - np.abs(np.abs(q) - 4)], 1)


class Viterbi:
    """K=7 rate-1/2 decoder for g0=133, g1=171 (octal), soft input, positive llr -> bit 1."""

    def __init__(self):
        ns = np.arange(64)
        inp = ns >> 5
        self.p0 = (ns << 1) & 63
        self.p1 = self.p0 | 1

        def outs(prev):
            reg = (inp << 6) | prev
            a = np.array([bin(r & 0o133).count("1") & 1 for r in reg])
            b = np.array([bin(r & 0o171).count("1") & 1 for r in reg])
            return (2 * a - 1).astype(np.float32), (2 * b - 1).astype(np.float32)

        self.a0, self.b0 = outs(self.p0)
        self.a1, self.b1 = outs(self.p1)

    def decode(self, llr):
        llr = np.asarray(llr, np.float32).reshape(-1, 2)
        n = len(llr)
        bm0 = llr[:, :1] * self.a0 + llr[:, 1:] * self.b0
        bm1 = llr[:, :1] * self.a1 + llr[:, 1:] * self.b1
        pm = np.full(64, -1e30, np.float32)
        pm[0] = 0
        dec = np.empty((n, 64), bool)
        p0, p1 = self.p0, self.p1
        for t in range(n):
            m0 = pm[p0] + bm0[t]
            m1 = pm[p1] + bm1[t]
            d = m1 > m0
            dec[t] = d
            pm = np.where(d, m1, m0)
            if t & 1023 == 1023:
                pm -= pm.max()
        s = int(np.argmax(pm))
        out = np.empty(n, np.uint8)
        for t in range(n - 1, -1, -1):
            out[t] = s >> 5
            s = ((s << 1) & 63) | int(dec[t, s])
        return out


VITERBI = Viterbi()


def depuncture(llr, rate, n_info_bits):
    mask = PUNCTURE[rate]
    full_len = 2 * n_info_bits
    m = np.tile(mask, -(-full_len // len(mask)))[:full_len]
    out = np.zeros(full_len, np.float32)
    need = int(m.sum())
    if len(llr) < need:
        return None
    out[m] = llr[:need]
    return out


def split_ampdu(psdu):
    """Return the MPDUs inside an A-MPDU (delimiter signature 0x4E, CRC-8 checked)."""
    out, i = [], 0
    while i + 4 <= len(psdu):
        d0, d1, c, sig = psdu[i:i + 4]
        if sig == 0x4E:
            v = d0 | (d1 << 8)
            crc = crc8(int_to_bits(v, 16))
            rev = int("{:08b}".format(crc)[::-1], 2)
            if c in (crc, rev):
                ln = ((v >> 4) & 0xFFF) | (((v >> 2) & 3) << 12)
                if ln == 0:
                    i += 4
                    continue
                if i + 4 + ln <= len(psdu):
                    out.append(psdu[i + 4:i + 4 + ln])
                i += 4 + ((ln + 3) & ~3)
                continue
        i += 4
    return out


# ---------------------------------------------------------------------------
# Frame record
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    index: float            # position in the 20/22 MSps channel stream
    data: bytes
    phy: str                # 'ofdm', 'ht', 'dsss'
    rate: float = 0.0       # Mbps for ofdm/dsss
    mcs: int = -1
    sgi: bool = False
    short_preamble: bool = False
    fcs_ok: bool = True
    snr_db: float = 0.0
    sig_dbfs: float = -100.0
    noise_dbfs: float = -100.0
    # filled by the driver
    raw_index: float = 0.0
    freq_mhz: int = 0
    channel: int = 0


def dbfs(p):
    return 10 * math.log10(max(p, 1e-15))


# ---------------------------------------------------------------------------
# OFDM receiver (20 MSps)
# ---------------------------------------------------------------------------


class OfdmRx:
    BK = 3  # FFT window starts this many samples inside the cyclic prefix

    def __init__(self, keep_bad, stats):
        self.keep_bad = keep_bad
        self.stats = stats
        self.ltf_t = np.fft.ifft(LTF64)
        self.ltf_norm = np.linalg.norm(self.ltf_t)
        self.perm_sig = interleave_perm(48, 1, False)

    @staticmethod
    def _ma(v, n):
        c = np.cumsum(np.concatenate([[0], v]))
        return c[n:] - c[:-n]

    def process(self, x, lo, hi):
        """Decode packets whose preamble starts in x[lo:hi]. Returns a list of Frames."""
        frames = []
        if len(x) < 1000:
            return frames
        x = x.astype(np.complex128)
        prod = x[:-16] * np.conj(x[16:])
        c = self._ma(prod, 48)
        e = np.abs(x) ** 2
        p1 = self._ma(e[:-16], 48)
        p2 = self._ma(e[16:], 48)
        m = np.abs(c) / np.sqrt(p1 * p2 + 1e-30)
        floor = max(np.percentile(p2, 10), 1e-20)
        mask = (m > 0.6) & (p2 > 4 * floor)
        self.noise = floor / 48
        if not mask.any():
            return frames
        mi = mask.astype(np.int32)
        cm = np.concatenate([[0], np.cumsum(mi)])
        edges = np.flatnonzero(mask & ~np.concatenate([[False], mask[:-1]]))
        next_free = 0
        for s in edges:
            if s < lo or s >= hi:
                continue
            if s < next_free:
                continue
            if cm[min(s + 64, len(mask))] - cm[s] < 48:
                continue
            got, end = self._decode(x, int(s))
            frames.extend(got)
            next_free = end
        return frames

    def _eq(self, y, start, cp, H, pilots, data_idx, data_k):
        bk = self.BK
        a = start + cp - bk
        Y = np.fft.fft(y[a:a + 64])
        zp = Y[PILOT_IDX] / H[PILOT_IDX]
        cpl = zp * np.conj(pilots)
        ph0 = np.angle(cpl.sum())
        res = np.angle(cpl * np.exp(-1j * ph0))
        w = np.abs(H[PILOT_IDX]) ** 2
        slope = float(np.sum(w * res * PILOT_K) / np.sum(w * PILOT_K ** 2))
        z = Y[data_idx] / H[data_idx]
        return z * np.exp(-1j * (ph0 + slope * data_k))

    def _decode(self, x, s):
        n = len(x)
        if s + 500 > n:
            return [], s + 16
        a = x[s + 16:s + 112]
        b = x[s + 32:s + 128]
        w = -np.angle(np.vdot(b, a)) / 16
        span = min(n - s, 760)
        seg = x[s:s + span] * np.exp(-1j * w * np.arange(span))
        win = seg[80:420]
        xc = np.abs(np.correlate(win, self.ltf_t, "valid"))
        mm = xc[:-64] + xc[64:]
        t = int(np.argmax(mm))
        t1 = 80 + t
        # LTF match quality gate: rejects adjacent-channel leakage and noise
        nrm = np.linalg.norm(win[t:t + 64]) * self.ltf_norm + 1e-30
        if xc[t] / nrm < 0.5:
            return [], s + 80
        w += -np.angle(np.vdot(seg[t1 + 64:t1 + 128], seg[t1:t1 + 64])) / 64

        def rot(length):
            length = min(length, n - s)
            return x[s:s + length] * np.exp(-1j * w * np.arange(length))

        y = rot(t1 + 128 + 80 * 6 + 64)
        bk = self.BK
        if len(y) < t1 + 128 + 80 * 6:
            return [], s + 80
        Y1 = np.fft.fft(y[t1 - bk:t1 - bk + 64])
        Y2 = np.fft.fft(y[t1 + 64 - bk:t1 + 128 - bk])
        used = LTF64 != 0
        H = np.ones(64, complex)
        H[used] = (Y1[used] + Y2[used]) / 2 / LTF64[used]
        nv = max(float(np.mean(np.abs(Y1[used] - Y2[used]) ** 2) / 2), 1e-30)
        snr_db = 10 * math.log10(max(float(np.mean(np.abs(H[used]) ** 2)) / nv, 1e-9))
        sig_dbfs = dbfs(float(np.mean(np.abs(y[t1:t1 + 128]) ** 2)))
        noise_dbfs = dbfs(self.noise)

        didx = DATA_K_L % 64
        wts = np.abs(H[didx]) ** 2 / nv
        t_sig = t1 + 128
        z = self._eq(y, t_sig, 16, H, PILOT_PSI * PILOT_POL[0], didx, DATA_K_L)
        llr = (z.real * wts)[self.perm_sig]
        sb = VITERBI.decode(llr)
        rate_code = bits_to_int(sb[0:4])
        length = bits_to_int(sb[5:17])
        if sb[:18].sum() % 2 or sb[18:24].any() or rate_code not in LEGACY_RATES or length == 0:
            return [], s + 160
        mbps, nbpsc, cr = LEGACY_RATES[rate_code]
        ndbps = int(48 * nbpsc * cr)
        nsym = -(-(16 + 8 * length + 6) // ndbps)
        end = s + t_sig + 80 * (nsym + 1)
        meta = dict(snr_db=snr_db, sig_dbfs=sig_dbfs, noise_dbfs=noise_dbfs)

        if mbps == 6:
            z2 = self._eq(y, t_sig + 80, 16, H, PILOT_PSI * PILOT_POL[1], didx, DATA_K_L)
            if np.sum(z2.imag ** 2) > 2 * np.sum(z2.real ** 2):
                return self._decode_ht(x, s, w, y, H, nv, t_sig, end, meta)

        y = rot(t_sig + 80 * (nsym + 1) + 8)
        if len(y) < t_sig + 80 * (nsym + 1):
            return [], end
        perm = interleave_perm(48 * nbpsc, nbpsc, False)
        llrs = []
        for k in range(nsym):
            st = t_sig + 80 * (k + 1)
            z = self._eq(y, st, 16, H, PILOT_PSI * PILOT_POL[(k + 1) % 127], didx, DATA_K_L)
            llrs.append((demap(z, nbpsc) * wts[:, None]).reshape(-1)[perm])
        full = depuncture(np.concatenate(llrs), cr, nsym * ndbps)
        bits = ofdm_descramble(VITERBI.decode(full))
        psdu = bits_to_bytes(bits[16:16 + 8 * length])
        self.stats["ofdm_ppdu"] = self.stats.get("ofdm_ppdu", 0) + 1
        return self._emit([psdu], s, "ofdm", meta, rate=mbps), end

    def _smooth(self, H):
        """3-tap [1 2 1] smoothing across subcarriers -28..-1 and 1..28, done with the
        FFT-window-offset phase ramp removed so neighbours average coherently."""
        k = np.arange(64)
        k = np.where(k < 32, k, k - 64)
        ramp = np.exp(-2j * np.pi * k * self.BK / 64)
        G = H / ramp
        out = H.copy()
        for side in (np.arange(-28, 0), np.arange(1, 29)):
            g = G[side % 64]
            pad = np.concatenate([[g[0]], g, [g[-1]]])
            out[side % 64] = (pad[:-2] + 2 * pad[1:-1] + pad[2:]) / 4 * ramp[side % 64]
        return out

    def _decode_ht(self, x, s, w, y, H, nv, t_sig, l_end, meta):
        didx = DATA_K_L % 64
        wts = np.abs(H[didx]) ** 2 / nv
        llrs = []
        for k in (1, 2):
            z = self._eq(y, t_sig + 80 * k, 16, H, PILOT_PSI * PILOT_POL[k], didx, DATA_K_L)
            llrs.append((z.imag * wts)[self.perm_sig])
        hb = VITERBI.decode(np.concatenate(llrs))
        if hb[42:48].any():
            return [], l_end
        crc = crc8(hb[:34])
        if not (np.array_equal(hb[34:42], msb_bits(crc, 8)) or
                np.array_equal(hb[34:42], int_to_bits(crc, 8))):
            self.stats["ht_sig_crc_fail"] = self.stats.get("ht_sig_crc_fail", 0) + 1
            return [], l_end
        mcs = bits_to_int(hb[0:7])
        cbw40 = hb[7]
        length = bits_to_int(hb[8:24])
        aggregation = hb[27]
        stbc = bits_to_int(hb[28:30])
        ldpc = hb[30]
        sgi = bool(hb[31])
        ness = bits_to_int(hb[32:34])
        if mcs > 7 or cbw40 or stbc or ldpc or ness or length == 0:
            self.stats["ht_unsupported"] = self.stats.get("ht_unsupported", 0) + 1
            return [], l_end
        nbpsc, cr = HT_MCS[mcs]
        ndbps = int(52 * nbpsc * cr)
        nsym = -(-(16 + 8 * length + 6) // ndbps)
        symlen, cp = (72, 8) if sgi else (80, 16)
        d0 = t_sig + 400
        need = d0 + nsym * symlen + 8
        n = len(x)
        if s + need > n:
            return [], l_end
        y = x[s:s + need] * np.exp(-1j * w * np.arange(need))
        bk = self.BK
        Yl = np.fft.fft(y[t_sig + 320 + 16 - bk:t_sig + 320 + 16 - bk + 64])
        used = HTLTF64 != 0
        Hh = np.ones(64, complex)
        Hh[used] = Yl[used] / HTLTF64[used]
        if hb[24]:
            Hh = self._smooth(Hh)  # transmitter allows smoothing; recovers most of the single-LTF loss
        hidx = DATA_K_HT % 64
        hw = np.abs(Hh[hidx]) ** 2 / nv
        perm = interleave_perm(52 * nbpsc, nbpsc, True)
        llrs = []
        for k in range(nsym):
            pil = np.roll(PILOT_PSI, -k) * PILOT_POL[(k + 3) % 127]
            z = self._eq(y, d0 + k * symlen, cp, Hh, pil, hidx, DATA_K_HT)
            llrs.append((demap(z, nbpsc) * hw[:, None]).reshape(-1)[perm])
        full = depuncture(np.concatenate(llrs), cr, nsym * ndbps)
        bits = ofdm_descramble(VITERBI.decode(full))
        psdu = bits_to_bytes(bits[16:16 + 8 * length])
        self.stats["ht_ppdu"] = self.stats.get("ht_ppdu", 0) + 1
        mpdus = split_ampdu(psdu) if aggregation else [psdu]
        if aggregation and not mpdus:
            self.stats["ampdu_no_delimiters"] = self.stats.get("ampdu_no_delimiters", 0) + 1
        return self._emit(mpdus, s, "ht", meta, mcs=mcs, sgi=sgi), s + d0 + nsym * symlen

    def _emit(self, mpdus, s, phy, meta, **kw):
        out = []
        for m in mpdus:
            ok = fcs_ok(m)
            key = "fcs_ok" if ok else "fcs_bad"
            self.stats[key] = self.stats.get(key, 0) + 1
            if ok or self.keep_bad:
                out.append(Frame(index=float(s), data=bytes(m), phy=phy, fcs_ok=ok, **meta, **kw))
        return out


# ---------------------------------------------------------------------------
# DSSS / CCK receiver (22 MSps = 2 samples per chip)
# ---------------------------------------------------------------------------

BARKER = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], float)
SFD_LONG = int_to_bits(0xF3A0, 16)
SFD_SHORT = int_to_bits(0x05CF, 16)
DSSS_RATES = {0x0A: 1.0, 0x14: 2.0, 0x37: 5.5, 0x6E: 11.0}
DQPSK_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}


def _cck_codebooks():
    q = np.exp(1j * np.pi / 2 * np.arange(4))

    def word(p2, p3, p4):
        return np.array([p2 * p3 * p4, p3 * p4, p2 * p4, -p4, p2 * p3, p3, -p2, 1])

    cb11, bits11 = [], []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                cb11.append(word(q[a], q[b], q[c]))
                bits11.append([a >> 1, a & 1, b >> 1, b & 1, c >> 1, c & 1])
    cb55, bits55 = [], []
    for d2 in range(2):
        for d3 in range(2):
            cb55.append(word(np.exp(1j * (d2 * np.pi + np.pi / 2)), 1.0, np.exp(1j * d3 * np.pi)))
            bits55.append([d2, d3])
    return (np.array(cb11), np.array(bits11, np.uint8), np.array(cb55), np.array(bits55, np.uint8))


CCK11, CCK11_BITS, CCK55, CCK55_BITS = _cck_codebooks()


class _Track:
    """Follows Barker symbol peaks, absorbing sample-clock drift."""

    def __init__(self, abscorr, p0):
        self.c = abscorr
        self.pos = float(p0)
        self.idx = []

    def run(self, count):
        c, n = self.c, len(self.c)
        for _ in range(count):
            i = int(round(self.pos))
            if i < 3 or i + 4 >= n:
                return False
            if len(self.idx) < 8:  # pull in onto the true correlation peak
                i = i - 2 + int(np.argmax(c[i - 2:i + 3]))
                self.pos = float(i)
            a, b, d = c[i - 1], c[i], c[i + 1]
            den = a - 2 * b + d
            off = 0.5 * (a - d) / den if den < 0 else 0.0
            off = min(max(off, -1.0), 1.0)
            g = 0.3 if len(self.idx) < 24 else 0.08
            self.idx.append(i)
            self.pos = self.pos + 22 + g * (i + off - self.pos)
        return True


class DsssRx:
    def __init__(self, keep_bad, stats):
        self.keep_bad = keep_bad
        self.stats = stats

    def process(self, x, lo, hi):
        frames = []
        n = len(x) - 21
        if n < 22 * 64:
            return frames
        x = x.astype(np.complex128)
        corr = np.zeros(n, complex)
        for i in range(11):
            corr += BARKER[i] * x[2 * i:2 * i + n]
        self._abs = np.abs(corr)
        e = self._abs ** 2
        nsym = n // 22
        M = e[:nsym * 22].reshape(nsym, 22)
        cs = np.concatenate([np.zeros((1, 22)), np.cumsum(M, 0)])
        W = cs[16:] - cs[:-16]
        mx = W.max(1)
        mean = W.mean(1)
        floor = max(np.percentile(mean, 20), 1e-20)
        det = (mx > 3.0 * mean) & (mx > 6 * floor)
        self.noise = float(np.percentile(np.abs(x) ** 2, 20))
        cand = np.flatnonzero(det)
        next_free = 0
        k = 0
        while k < len(cand):
            # first detection window at or after next_free (re-acquires inside a
            # long detection run after a failed or finished packet)
            k += int(np.searchsorted(cand[k:], max(next_free // 22, 0)))
            if k >= len(cand):
                break
            w0 = int(cand[k])
            # take the chip phase from a window that lies fully inside the preamble
            wp = min(w0 + 16, len(W) - 1)
            p0 = int(wp * 22 + np.argmax(W[wp]))
            if p0 >= hi:
                break
            if p0 < lo:
                next_free = (w0 + 1) * 22
                continue
            self._first, self._retry_pending = True, None
            got, end = self._decode(x, corr, p0)
            if not got and self._retry_pending is not None:
                # Acquisition started well before the SFD (on noise or another
                # frame's tail); re-acquire on the preamble just before the SFD.
                r = self._retry_pending
                ac = np.abs(corr[max(r - 11, 0):r + 11])
                self._first = False
                if len(ac):
                    got2, end2 = self._decode(x, corr, max(r - 11, 0) + int(np.argmax(ac)))
                    if got2:
                        got, end = got2, end2
            frames.extend(got)
            next_free = max(end, (w0 + 8) * 22)
        return frames

    def _decode(self, x, corr, p0):
        tr = _Track(self._abs, p0)
        tr.run(220)
        if len(tr.idx) < 60:
            return [], p0 + 22 * 16
        y = corr[tr.idx]
        d = y[1:] * np.conj(y[:-1])
        theta = np.angle(np.sum(d[:32] ** 2)) / 2
        rot = np.exp(-1j * theta)
        z = (np.real(d * rot) < 0).astype(np.uint8)  # z[k] belongs to symbol k+1
        desc = dsss_descramble(z)
        win = np.lib.stride_tricks.sliding_window_view(desc, 16)
        hits = []
        for pat, short, inv in ((SFD_LONG, False, False), (SFD_SHORT, True, False),
                                (1 - SFD_LONG, False, True), (1 - SFD_SHORT, True, True)):
            h = np.flatnonzero((win[7:] == pat).all(1))
            if len(h):
                hits.append((int(h[0]) + 7, short, inv))
        if not hits:
            return [], p0 + 22 * 16
        j, short, inv = min(hits)
        # Re-estimate the carrier offset on the symbols right before and inside the
        # SFD: acquisition may have started on noise or the tail of another frame.
        seg = d[max(0, j - 40):j + 16]
        theta = np.angle(np.sum(seg ** 2)) / 2
        rot = np.exp(-1j * theta)
        z = (np.real(d * rot) < 0).astype(np.uint8)
        if not np.array_equal(dsss_descramble(z)[j:j + 16], SFD_SHORT if short else SFD_LONG):
            theta += np.pi
            rot = -rot
            z ^= 1
        self._sfd_sym = j
        if j > 24 and self._first:
            self._retry_pending = tr.idx[j - 20]
        nhdr_sym = 24 if short else 48
        m_hdr = j + 17  # first header symbol
        if not tr.run(max(0, m_hdr + nhdr_sym + 1 - len(tr.idx))):
            return [], tr.idx[-1]
        y = corr[tr.idx]
        raw = list(z[:j + 16])
        if short:
            hdr_raw = self._dqpsk(y, m_hdr, nhdr_sym, rot)
        else:
            hdr_raw = self._dbpsk(y, m_hdr, nhdr_sym, rot)
        hdr = dsss_descramble(np.array(raw + hdr_raw, np.uint8))[j + 16:]
        crc = crc16(hdr[:32])
        if not (np.array_equal(hdr[32:48], msb_bits(crc, 16)) or
                np.array_equal(hdr[32:48], int_to_bits(crc, 16))):
            self.stats["dsss_hdr_crc_fail"] = self.stats.get("dsss_hdr_crc_fail", 0) + 1
            return [], p0 + 22 * 16
        signal = bits_to_int(hdr[0:8])
        service = hdr[8:16]
        lus = bits_to_int(hdr[16:32])
        if signal not in DSSS_RATES or lus == 0:
            return [], p0 + 22 * 16
        rate = DSSS_RATES[signal]
        if rate == 1:
            nbytes = lus // 8
        elif rate == 2:
            nbytes = lus // 4
        elif rate == 5.5:
            nbytes = (lus * 11) // 16
        else:
            nbytes = (lus * 11) // 8 - int(service[7])
        if nbytes < 14:
            return [], p0 + 22 * 16
        m_psdu = m_hdr + nhdr_sym
        raw += hdr_raw
        if rate in (1.0, 2.0):
            nsym = nbytes * 8 if rate == 1 else nbytes * 4
            if not tr.run(max(0, m_psdu + nsym - len(tr.idx))):
                return [], tr.idx[-1]
            y = corr[tr.idx]
            raw += self._dbpsk(y, m_psdu, nsym, rot) if rate == 1 else self._dqpsk(y, m_psdu, nsym, rot)
            end = tr.idx[m_psdu + nsym - 1] + 22
        else:
            nsym = nbytes * 8 // (4 if rate == 5.5 else 8)
            got = self._cck(x, tr, m_psdu, nsym, theta, rate)
            if got is None:
                return [], tr.idx[-1]
            bits, end = got
            raw += bits
        allb = dsss_descramble(np.array(raw, np.uint8))
        base = j + 16 + 48
        psdu = bits_to_bytes(allb[base:base + 8 * nbytes])
        ok = fcs_ok(psdu)
        self.stats["dsss_ppdu"] = self.stats.get("dsss_ppdu", 0) + 1
        self.stats["fcs_ok" if ok else "fcs_bad"] = self.stats.get("fcs_ok" if ok else "fcs_bad", 0) + 1
        if not ok and not self.keep_bad:
            return [], end
        i_sfd = tr.idx[j + 16]
        start = i_sfd - (71 if short else 143) * 22
        ys = np.abs(y[m_hdr:m_hdr + nhdr_sym]) ** 2
        mid = np.abs(corr[np.array(tr.idx[m_hdr:m_hdr + nhdr_sym]) + 11]) ** 2
        snr = 10 * math.log10(max(ys.mean() / max(mid.mean(), 1e-30), 1e-9))
        sig = dbfs(float(np.mean(np.abs(x[i_sfd:i_sfd + 22 * 48]) ** 2)))
        return [Frame(index=float(start), data=psdu, phy="dsss", rate=rate, short_preamble=short,
                      fcs_ok=ok, snr_db=snr, sig_dbfs=sig, noise_dbfs=dbfs(self.noise))], end

    @staticmethod
    def _dbpsk(y, m0, count, rot):
        d = y[m0:m0 + count] * np.conj(y[m0 - 1:m0 - 1 + count]) * rot
        return [int(v) for v in (d.real < 0)]

    @staticmethod
    def _dqpsk(y, m0, count, rot):
        d = y[m0:m0 + count] * np.conj(y[m0 - 1:m0 - 1 + count]) * rot
        q = np.round(np.angle(d) / (np.pi / 2)).astype(int) % 4
        out = []
        for v in q:
            out.extend(DQPSK_BITS[int(v)])
        return out

    def _cck(self, x, tr, m_psdu, nsym, theta, rate):
        # Fit the symbol clock on preamble/header symbols only; the tracker may
        # have run past the header into CCK chips, where Barker peaks are meaningless.
        hist = tr.idx[max(0, self._sfd_sym - 40):m_psdu]
        ids = np.array(hist[-min(len(hist), 200):], float)
        if len(ids) > 10:
            slope, icpt = np.polyfit(np.arange(len(ids)), ids, 1)
            base = icpt + slope * (len(ids) - 1)  # smoothed start of the last header symbol
        else:
            slope, base = 22.0, float(tr.idx[m_psdu - 1])
        omega = theta / slope
        cb, cbits = (CCK11, CCK11_BITS) if rate == 11 else (CCK55, CCK55_BITS)
        q = np.arange(nsym * 8)
        best_score, pick = -1.0, None
        # The Barker peak is about one sample wide at 2 samples/chip, so try the
        # neighbouring sample phases and keep the one CCK correlates best with.
        for delta in (-1.0, -0.5, 0.0, 0.5, 1.0):
            i_last = int(round(base + delta))
            pos = np.round(base + delta + slope * (11 + q) / 11).astype(int)
            if i_last < 0 or pos[-1] >= len(x):
                continue
            ns = min(nsym, 32)
            Rt = x[pos[:ns * 8]].reshape(ns, 8)
            score = float(np.abs(Rt @ cb.conj().T).max(1).sum())
            if score > best_score:
                best_score, pick = score, (i_last, pos)
        if pick is None:
            return None
        i_last, pos = pick
        t0 = float(pos[0]) - slope  # chip 0 of the last header symbol, on the chosen phase
        cs = slope / 11
        cbh = cb.conj().T
        yl = np.sum(BARKER * x[i_last + 2 * np.arange(11)])
        phi_prev = np.angle(yl) - omega * 10
        offs = np.array([-1, 0, 1])[:, None]
        c8 = np.arange(8) * cs
        tau = 0.0
        best = np.empty(nsym, int)
        phi = np.empty(nsym)
        n = len(x)
        last = 0
        # Decision-directed early/late tracking keeps the chip phase locked
        # over long CCK bursts, where a small symbol-clock error would slip chips.
        for s in range(nsym):
            ts = t0 + cs * (11 + 8 * s) + tau
            idx = np.round(ts + offs + c8).astype(int)
            if idx.max() >= n or idx.min() < 0:
                return None
            r = x[idx] * np.exp(-1j * omega * (idx - i_last))
            C = r @ cbh
            mag = np.abs(C)
            b = int(np.argmax(mag[1]))
            best[s] = b
            phi[s] = np.angle(C[1, b])
            e_m, e_0, e_p = mag[0].max(), mag[1, b], mag[2].max()
            den = e_m - 2 * e_0 + e_p
            if den < 0:
                tau += 0.1 * min(max(0.5 * (e_m - e_p) / den, -1.0), 1.0)
            last = int(idx[1, -1])
        dphi = np.diff(np.concatenate([[phi_prev], phi])) - np.pi * (np.arange(nsym) % 2)
        k = np.round(dphi / (np.pi / 2)).astype(int) % 4
        out = []
        for s in range(nsym):
            out.extend(DQPSK_BITS[int(k[s])])
            out.extend(int(b) for b in cbits[best[s]])
        return out, last + 2


# ---------------------------------------------------------------------------
# Channel plan and frequency-domain channelizer
# ---------------------------------------------------------------------------


def all_channels():
    ch = [(c, 2407 + 5 * c) for c in range(1, 14)] + [(14, 2484)]
    ch += [(c, 5000 + 5 * c) for c in list(range(32, 145, 4)) + list(range(149, 178, 4))]
    return ch


def pick_channels(center_hz, fs, edge, wanted):
    half = edge * fs / 2
    out = []
    for c, f in all_channels():
        off = f * 1e6 - center_hz
        if wanted is not None and c not in wanted:
            continue
        if abs(off) + 10e6 <= half or (wanted is not None and abs(off) + 10e6 <= fs / 2):
            out.append((c, f))
    return out


def extract(X, n, fs, f_off, fout, m, f_pass, f_stop):
    """Shift f_off to DC, low-pass with a cosine taper, return m samples at fout."""
    rel = (np.fft.fftfreq(m) * m).round().astype(np.int64)
    k0 = int(round(f_off * n / fs))
    f_rel = np.abs(rel * fs / n)
    taper = np.where(f_rel <= f_pass, 1.0,
                     np.where(f_rel >= f_stop, 0.0,
                              0.5 * (1 + np.cos(np.pi * (f_rel - f_pass) / (f_stop - f_pass)))))
    f_abs = (k0 + rel) * fs / n
    taper *= np.abs(f_abs) < fs / 2
    Y = X[(k0 + rel) % n] * taper.astype(np.float32)
    return (sfft.ifft(Y, workers=-1) * (m / n)).astype(np.complex64)


# ---------------------------------------------------------------------------
# pcap / radiotap
# ---------------------------------------------------------------------------


def radiotap(fr):
    present = (1 << 1) | (1 << 3) | (1 << 12) | (1 << 13)
    body = bytearray()
    flags = 0x10 | (0x40 if not fr.fcs_ok else 0) | (0x02 if fr.short_preamble else 0)
    body.append(flags)
    if fr.phy != "ht":
        present |= 1 << 2
        body.append(int(round(fr.rate * 2)))
    if (8 + len(body)) % 2:
        body.append(0)
    chflags = (0x0080 if fr.freq_mhz < 3000 else 0x0100) | (0x0020 if fr.phy == "dsss" else 0x0040)
    body += struct.pack("<HH", fr.freq_mhz, chflags)
    body.append(int(min(255, max(0, round(fr.sig_dbfs + 100)))))
    body.append(int(min(255, max(0, round(fr.noise_dbfs + 100)))))
    if fr.phy == "ht":
        present |= 1 << 19
        body += bytes([0x1F, 0x04 if fr.sgi else 0x00, fr.mcs])
    return struct.pack("<BBHI", 0, 0, 8 + len(body), present) + bytes(body)


def write_pcap(path, frames, t0, fs):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 262144, 127))
        for fr in frames:
            pkt = radiotap(fr) + fr.data
            t = t0 + fr.raw_index / fs
            sec = int(math.floor(t))
            ns = int(round((t - sec) * 1e9))
            if ns >= 1000000000:
                sec, ns = sec + 1, ns - 1000000000
            f.write(struct.pack("<IIII", sec, ns, len(pkt), len(pkt)))
            f.write(pkt)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_freq(s):
    v = float(s)
    return v * 1e6 if v < 1e5 else v  # accept MHz or Hz


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode 802.11 frames from a BirriaSDR capture into a pcap.")
    ap.add_argument("input", help="capture file (cs16 from birria_rx, or cf32)")
    ap.add_argument("-o", "--output", help="output pcap (default: input name with .pcap)")
    ap.add_argument("-f", "--center", type=parse_freq, default=2437e6,
                    help="center frequency the capture was tuned to, Hz or MHz (default 2437 MHz)")
    ap.add_argument("-r", "--rate", type=float, default=80e6, help="sample rate in S/s (default 80e6)")
    ap.add_argument("--format", choices=["cs16", "cf32"], default="cs16")
    ap.add_argument("-c", "--channels", help="comma-separated Wi-Fi channel numbers (default: all that fit)")
    ap.add_argument("--edge", type=float, default=0.9,
                    help="fraction of the sampled span treated as usable when picking channels (default 0.9)")
    ap.add_argument("--phy", default="ofdm,dsss", help="receivers to run: ofdm, dsss (default both)")
    ap.add_argument("--ppm", type=float, default=0.0, help="dongle crystal error to correct, in ppm")
    ap.add_argument("--swap-iq", action="store_true", help="swap I and Q (use if the spectrum is mirrored)")
    ap.add_argument("--keep-bad-fcs", action="store_true", help="also write frames that fail the FCS")
    ap.add_argument("--start-time", type=float,
                    help="UNIX time of the first sample (default: file mtime minus capture length)")
    ap.add_argument("--skip", type=float, default=0.0, help="seconds to skip at the start")
    ap.add_argument("--duration", type=float, help="seconds to process")
    ap.add_argument("--block", type=float, default=0.1, help="processing block length in seconds")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    fs_nom = args.rate
    corr = 1 + args.ppm * 1e-6
    fs = fs_nom * corr
    fc = args.center * corr
    out_path = args.output or os.path.splitext(args.input)[0] + ".pcap"
    phys = {p.strip() for p in args.phy.split(",")}
    wanted = None if not args.channels else {int(c) for c in args.channels.split(",")}

    if args.format == "cs16":
        raw = np.memmap(args.input, dtype="<i2", mode="r")
        total = len(raw) // 2
        scale = 1 / 8192.0
    else:
        raw = np.memmap(args.input, dtype="<f4", mode="r")
        total = len(raw) // 2
        scale = 1.0
    if total == 0:
        sys.exit("input is empty")

    chans = pick_channels(args.center, fs_nom, args.edge, wanted)
    if not chans:
        sys.exit("no Wi-Fi channel fits inside the captured span; check --center/--rate/--channels")

    rates = []
    if "ofdm" in phys:
        rates.append(20e6)
    if "dsss" in phys:
        rates.append(22e6)
    ratios = [Fraction(int(r), int(fs_nom)) for r in rates]
    if any(r > 1 for r in ratios):
        sys.exit("sample rate must be at least 22 MS/s for DSSS (or 20 MS/s with --phy ofdm)")
    lcm = 1
    for r in ratios:
        lcm = lcm * r.denominator // math.gcd(lcm, r.denominator)

    t0 = args.start_time
    if t0 is None:
        t0 = os.path.getmtime(args.input) - total / fs
    first = int(args.skip * fs_nom)
    last = total if args.duration is None else min(total, first + int(args.duration * fs_nom))
    step = int(args.block * fs_nom)
    pre = 4096
    post = int(0.025 * fs_nom)

    if not args.quiet:
        names = ", ".join("{}({:+.0f})".format(c, (f * 1e6 - args.center) / 1e6) for c, f in chans)
        print("channels (MHz offset): " + names, file=sys.stderr)

    stats = {}
    orx = OfdmRx(args.keep_bad_fcs, stats)
    drx = DsssRx(args.keep_bad_fcs, stats)
    frames = []
    tstart = time.time()
    for a0 in range(first, last, step):
        a = max(first, a0 - pre)
        b = min(last, a0 + step + post)
        seg = np.asarray(raw[2 * a:2 * b], np.float32).reshape(-1, 2)
        x = (seg[:, 0] + 1j * seg[:, 1]).astype(np.complex64) * np.float32(scale)
        if args.swap_iq:
            x = (seg[:, 1] + 1j * seg[:, 0]).astype(np.complex64) * np.float32(scale)
        x -= x.mean()
        L = len(x)
        n = lcm * sfft.next_fast_len(-(-L // lcm))
        X = sfft.fft(x, n, workers=-1)
        for ch, fmhz in chans:
            off = fmhz * 1e6 - fc
            for rate_out in rates:
                m = int(Fraction(n) * Fraction(int(rate_out), int(fs_nom)))
                ratio = rate_out / fs_nom
                y = extract(X, n, fs_nom, off * fs_nom / fs, rate_out, m,
                            9.0e6 if rate_out == 20e6 else 9.5e6,
                            10.0e6 if rate_out == 20e6 else 11.0e6)
                y = y[: int(L * ratio) + 1]
                lo = int((a0 - a) * ratio)
                hi = int((min(a0 + step, last) - a) * ratio)
                rx = orx if rate_out == 20e6 else drx
                for fr in rx.process(y, lo, hi):
                    fr.raw_index = a + fr.index / ratio
                    fr.freq_mhz = fmhz
                    fr.channel = ch
                    frames.append(fr)
        if not args.quiet:
            done = (min(a0 + step, last) - first) / max(last - first, 1)
            print("\r{:5.1f}%  {} frames  {:.0f}s".format(100 * done, len(frames), time.time() - tstart),
                  end="", file=sys.stderr)
    if not args.quiet:
        print(file=sys.stderr)

    # Drop duplicates: the same bytes within 100 us (block overlap, adjacent-channel leakage).
    frames.sort(key=lambda fr: fr.raw_index)
    kept, last_seen = [], {}
    win = 100e-6 * fs
    for fr in frames:
        prev = last_seen.get(fr.data)
        if prev is not None and fr.raw_index - kept[prev].raw_index < win:
            if fr.snr_db > kept[prev].snr_db:
                kept[prev] = fr
            continue
        last_seen[fr.data] = len(kept)
        kept.append(fr)
    write_pcap(out_path, kept, t0, fs)

    if not args.quiet:
        by = {}
        for fr in kept:
            k = (fr.channel, fr.phy)
            by[k] = by.get(k, 0) + 1
        print("wrote {} frames to {}".format(len(kept), out_path), file=sys.stderr)
        for (ch, phy), cnt in sorted(by.items()):
            print("  ch {:>3} {:<5} {}".format(ch, phy, cnt), file=sys.stderr)
        if stats:
            print("  " + ", ".join("{}={}".format(k, v) for k, v in sorted(stats.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
