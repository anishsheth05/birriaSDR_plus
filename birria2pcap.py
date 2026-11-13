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

    * OFDM. Detection and the legacy preamble run on each 20 MHz channel; when
      HT-SIG or VHT-SIG-A announces a 40 or 80 MHz PPDU, the wider channel is cut
      from the same block FFT and decoded there.
        - 802.11a/g legacy, 6-54 Mbps.
        - 802.11n HT-mixed: MCS 0-7 (one spatial stream), 20 and 40 MHz, long
          or short GI, BCC or LDPC, 2x1 STBC (Alamouti), A-MPDUs split.
        - 802.11ac VHT single-user: MCS 0-9 (one spatial stream, 256-QAM),
          20, 40 and 80 MHz, long or short GI, BCC or LDPC (with the LDPC tone
          mapper), 2x1 STBC, A-MPDUs split.
    * DSSS/CCK: 802.11b, long and short preamble, 1, 2, 5.5 and 11 Mbps.

    Not decoded (detected and counted in the summary): two or more spatial
    streams (BirriaSDR records one receive chain, and one antenna cannot
    separate them), VHT MU-MIMO, 160 MHz, HE (802.11ax), HT greenfield, PBCC.
    Frames that a multi-antenna AP sends as one stream (cyclic shift diversity,
    spatial expansion, beamforming, STBC) do decode. There is no multipath
    equalizer for DSSS.

    40 and 80 MHz: the whole wide channel must lie inside the captured span.
    An 80 MHz PPDU needs an 80 MSps capture centred on the 80 MHz channel
    (5210, 5290, 5530, 5610, 5690, 5775 MHz); its outer tones then sit at the
    edge of BirriaSDR's passband, which is not guaranteed flat there. In
    2.4 GHz a 40 MHz PPDU's secondary channel can be above or below; both are
    tried. The radiotap channel of a 40/80 MHz PPDU is the 20 MHz sub-channel
    where it was first detected, which is not necessarily the primary.

Output
    pcap (nanosecond timestamps) with a radiotap header per frame: flags (FCS
    present), legacy rate, HT MCS field or VHT field (bandwidth, MCS, GI, STBC,
    LDPC), channel frequency, and dB_antsignal/dB_antnoise.
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
# LDPC (802.11n/ac): prototype matrices, min-sum decoder, codeword layout
# ---------------------------------------------------------------------------

# Matrix prototypes from IEEE 802.11 (Annex F / 19.3.11.7): entry = cyclic shift
# of a Z x Z identity (columns shifted right), '-' = all-zero block.
LDPC_PROTOTYPES = """
648 1/2
0 - - - 0 0 - - 0 - - 0 1 0 - - - - - - - - - -
22 0 - - 17 - 0 0 12 - - - - 0 0 - - - - - - - - -
6 - 0 - 10 - - - 24 - 0 - - - 0 0 - - - - - - - -
2 - - 0 20 - - - 25 0 - - - - - 0 0 - - - - - - -
23 - - - 3 - - - 0 - 9 11 - - - - 0 0 - - - - - -
24 - 23 1 17 - 3 - 10 - - - - - - - - 0 0 - - - - -
25 - - - 8 - - - 7 18 - - 0 - - - - - 0 0 - - - -
13 24 - - 0 - 8 - 6 - - - - - - - - - - 0 0 - - -
7 20 - 16 22 10 - - 23 - - - - - - - - - - - 0 0 - -
11 - - - 19 - - - 13 - 3 17 - - - - - - - - - 0 0 -
25 - 8 - 23 18 - 14 9 - - - - - - - - - - - - - 0 0
3 - - - 16 - - 2 25 5 - - 1 - - - - - - - - - - 0
648 2/3
25 26 14 - 20 - 2 - 4 - - 8 - 16 - 18 1 0 - - - - - -
10 9 15 11 - 0 - 1 - - 18 - 8 - 10 - - 0 0 - - - - -
16 2 20 26 21 - 6 - 1 26 - 7 - - - - - - 0 0 - - - -
10 13 5 0 - 3 - 7 - - 26 - - 13 - 16 - - - 0 0 - - -
23 14 24 - 12 - 19 - 17 - - - 20 - 21 - 0 - - - 0 0 - -
6 22 9 20 - 25 - 17 - 8 - 14 - 18 - - - - - - - 0 0 -
14 23 21 11 20 - 24 - 18 - 19 - - - - 22 - - - - - - 0 0
17 11 11 20 - 21 - 26 - 3 - - 18 - 26 - 1 - - - - - - 0
648 3/4
16 17 22 24 9 3 14 - 4 2 7 - 26 - 2 - 21 - 1 0 - - - -
25 12 12 3 3 26 6 21 - 15 22 - 15 - 4 - - 16 - 0 0 - - -
25 18 26 16 22 23 9 - 0 - 4 - 4 - 8 23 11 - - - 0 0 - -
9 7 0 1 17 - - 7 3 - 3 23 - 16 - - 21 - 0 - - 0 0 -
24 5 26 7 1 - - 15 24 15 - 8 - 13 - 13 - 11 - - - - 0 0
2 2 19 14 24 1 15 19 - 21 - 2 - 24 - 3 - 2 1 - - - - 0
648 5/6
17 13 8 21 9 3 18 12 10 0 4 15 19 2 5 10 26 19 13 13 1 0 - -
3 12 11 14 11 25 5 18 0 9 2 26 26 10 24 7 14 20 4 2 - 0 0 -
22 16 4 3 10 21 12 5 21 14 19 5 - 8 5 18 11 5 5 15 0 - 0 0
7 7 14 14 4 16 16 24 24 10 1 7 15 6 10 26 8 18 21 14 1 - - 0
1296 1/2
40 - - - 22 - 49 23 43 - - - 1 0 - - - - - - - - - -
50 1 - - 48 35 - - 13 - 30 - - 0 0 - - - - - - - - -
39 50 - - 4 - 2 - - - - 49 - - 0 0 - - - - - - - -
33 - - 38 37 - - 4 1 - - - - - - 0 0 - - - - - - -
45 - - - 0 22 - - 20 42 - - - - - - 0 0 - - - - - -
51 - - 48 35 - - - 44 - 18 - - - - - - 0 0 - - - - -
47 11 - - - 17 - - 51 - - - 0 - - - - - 0 0 - - - -
5 - 25 - 6 - 45 - 13 40 - - - - - - - - - 0 0 - - -
33 - - 34 24 - - - 23 - - 46 - - - - - - - - 0 0 - -
1 - 27 - 1 - - - 38 - 44 - - - - - - - - - - 0 0 -
- 18 - - 23 - - 8 0 35 - - - - - - - - - - - - 0 0
49 - 17 - 30 - - - 34 - - 19 1 - - - - - - - - - - 0
1296 2/3
39 31 22 43 - 40 4 - 11 - - 50 - - - 6 1 0 - - - - - -
25 52 41 2 6 - 14 - 34 - - - 24 - 37 - - 0 0 - - - - -
43 31 29 0 21 - 28 - - 2 - - 7 - 17 - - - 0 0 - - - -
20 33 48 - 4 13 - 26 - - 22 - - 46 42 - - - - 0 0 - - -
45 7 18 51 12 25 - - - 50 - - 5 - - - 0 - - - 0 0 - -
35 40 32 16 5 - - 18 - - 43 51 - 32 - - - - - - - 0 0 -
9 24 13 22 28 - - 37 - - 25 - - 52 - 13 - - - - - - 0 0
32 22 4 21 16 - - - 27 28 - 38 - - - 8 1 - - - - - - 0
1296 3/4
39 40 51 41 3 29 8 36 - 14 - 6 - 33 - 11 - 4 1 0 - - - -
48 21 47 9 48 35 51 - 38 - 28 - 34 - 50 - 50 - - 0 0 - - -
30 39 28 42 50 39 5 17 - 6 - 18 - 20 - 15 - 40 - - 0 0 - -
29 0 1 43 36 30 47 - 49 - 47 - 3 - 35 - 34 - 0 - - 0 0 -
1 32 11 23 10 44 12 7 - 48 - 4 - 9 - 17 - 16 - - - - 0 0
13 7 15 47 23 16 47 - 43 - 29 - 52 - 2 - 53 - 1 - - - - 0
1296 5/6
48 29 37 52 2 16 6 14 53 31 34 5 18 42 53 31 45 - 46 52 1 0 - -
17 4 30 7 43 11 24 6 14 21 6 39 17 40 47 7 15 41 19 - - 0 0 -
7 2 51 31 46 23 16 11 53 40 10 7 46 53 33 35 - 25 35 38 0 - 0 0
19 48 41 1 10 7 36 47 5 29 52 52 31 10 26 6 3 2 - 51 1 - - 0
1944 1/2
57 - - - 50 - 11 - 50 - 79 - 1 0 - - - - - - - - - -
3 - 28 - 0 - - - 55 7 - - - 0 0 - - - - - - - - -
30 - - - 24 37 - - 56 14 - - - - 0 0 - - - - - - - -
62 53 - - 53 - - 3 35 - - - - - - 0 0 - - - - - - -
40 - - 20 66 - - 22 28 - - - - - - - 0 0 - - - - - -
0 - - - 8 - 42 - 50 - - 8 - - - - - 0 0 - - - - -
69 79 79 - - - 56 - 52 - - - 0 - - - - - 0 0 - - - -
65 - - - 38 57 - - 72 - 27 - - - - - - - - 0 0 - - -
64 - - - 14 52 - - 30 - - 32 - - - - - - - - 0 0 - -
- 45 - 70 0 - - - 77 9 - - - - - - - - - - - 0 0 -
2 56 - 57 35 - - - - - 12 - - - - - - - - - - - 0 0
24 - 61 - 60 - - 27 51 - - 16 1 - - - - - - - - - - 0
1944 2/3
61 75 4 63 56 - - - - - - 8 - 2 17 25 1 0 - - - - - -
56 74 77 20 - - - 64 24 4 67 - 7 - - - - 0 0 - - - - -
28 21 68 10 7 14 65 - - - 23 - - - 75 - - - 0 0 - - - -
48 38 43 78 76 - - - - 5 36 - 15 72 - - - - - 0 0 - - -
40 2 53 25 - 52 62 - 20 - - 44 - - - - 0 - - - 0 0 - -
69 23 64 10 22 - 21 - - - - - 68 23 29 - - - - - - 0 0 -
12 0 68 20 55 61 - 40 - - - 52 - - - 44 - - - - - - 0 0
58 8 34 64 78 - - 11 78 24 - - - - - 58 1 - - - - - - 0
1944 3/4
48 29 28 39 9 61 - - - 63 45 80 - - - 37 32 22 1 0 - - - -
4 49 42 48 11 30 - - - 49 17 41 37 15 - 54 - - - 0 0 - - -
35 76 78 51 37 35 21 - 17 64 - - - 59 7 - - 32 - - 0 0 - -
9 65 44 9 54 56 73 34 42 - - - 35 - - - 46 39 0 - - 0 0 -
3 62 7 80 68 26 - 80 55 - 36 - 26 - 9 - 72 - - - - - 0 0
26 75 33 21 69 59 3 38 - - - 35 - 62 36 26 - - 1 - - - - 0
1944 5/6
13 48 80 66 4 74 7 30 76 52 37 60 - 49 73 31 74 73 23 - 1 0 - -
69 63 74 56 64 77 57 65 6 16 51 - 64 - 68 9 48 62 54 27 - 0 0 -
51 15 0 80 24 25 42 54 44 71 71 9 67 35 - 58 - 29 - 53 0 - 0 0
16 29 36 41 44 56 59 37 50 24 - 65 4 65 52 - 4 - 73 52 1 - - 0
"""


def _parse_prototypes(text):
    protos, key = {}, None
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) == 2:
            a, b = parts[1].split("/")
            key = (int(parts[0]), Fraction(int(a), int(b)))
            protos[key] = []
        else:
            protos[key].append([-1 if v == "-" else int(v) for v in parts])
    return {k: np.array(v) for k, v in protos.items()}


LDPC_PROTO = _parse_prototypes(LDPC_PROTOTYPES)


class LdpcCode:
    """Normalized min-sum (flooding) decoder for one (length, rate) code."""

    def __init__(self, n, rate):
        proto = LDPC_PROTO[(n, rate)]
        z = n // 24
        mb = proto.shape[0]
        self.n, self.m, self.k = n, mb * z, n - mb * z
        rows = [[] for _ in range(self.m)]
        for r in range(mb):
            for c in range(24):
                s = proto[r, c]
                if s < 0:
                    continue
                for i in range(z):
                    rows[r * z + i].append(c * z + (i + s) % z)
        dmax = max(len(r) for r in rows)
        cv = np.full((self.m, dmax), n, np.int64)  # n = dummy variable (padding)
        for i, r in enumerate(rows):
            cv[i, :len(r)] = r
        self.cv = cv
        self.valid = cv < n

    def decode(self, llr, iters=50, alpha=0.75):
        """llr: (B, n), positive means bit 0. Returns (bits (B, n) uint8, ok (B,) bool)."""
        llr = np.atleast_2d(np.asarray(llr, np.float64))
        b, n = llr.shape
        cv, valid = self.cv, self.valid
        big = 1e9
        R = np.zeros((b,) + cv.shape)
        total = llr.copy()
        offs = (np.arange(b) * (n + 1))[:, None, None]
        flat_idx = (cv[None] + offs).ravel()
        ok = np.zeros(b, bool)
        hard = (total < 0).astype(np.uint8)
        for _ in range(iters):
            ext = np.concatenate([total, np.full((b, 1), big)], 1)
            Q = ext[:, cv] - R
            Q = np.where(valid, Q, big)
            sgn = np.where(Q < 0, -1.0, 1.0)
            psign = np.prod(sgn, axis=2, keepdims=True)
            mag = np.abs(Q)
            i1 = np.argmin(mag, axis=2)
            m1 = np.take_along_axis(mag, i1[..., None], 2)
            np.put_along_axis(mag, i1[..., None], big, 2)
            m2 = mag.min(axis=2, keepdims=True)
            first = np.arange(cv.shape[1])[None, None, :] == i1[..., None]
            R = alpha * psign * sgn * np.where(first, m2, m1)
            R = np.where(valid, R, 0.0)
            acc = np.bincount(flat_idx, weights=R.ravel(), minlength=b * (n + 1)).reshape(b, n + 1)
            total = llr + acc[:, :n]
            hard = (total < 0).astype(np.uint8)
            hx = np.concatenate([hard, np.zeros((b, 1), np.uint8)], 1)
            ok = ~(hx[:, cv].sum(axis=2) & 1).any(axis=1)
            if ok.all():
                break
        return hard, ok


_LDPC_CACHE = {}


def ldpc_code(n, rate):
    key = (n, rate)
    if key not in _LDPC_CACHE:
        _LDPC_CACHE[key] = LdpcCode(n, rate)
    return _LDPC_CACHE[key]


def ldpc_layout(n_pld, n_cbps, rate, m_stbc, n_avbits=None, extra=None):
    """Codeword parameters of 802.11-2020 19.3.11.7.5. For HT pass n_avbits=None
    (the receiver re-derives the extra-symbol decision); for VHT pass the
    N_avbits of N_SYM_init and the LDPC-extra-symbol bit from VHT-SIG-A."""
    vht = n_avbits is not None
    if not vht:
        n_avbits = n_cbps * m_stbc * math.ceil(n_pld / (n_cbps * rate * m_stbc))
    if n_avbits <= 648:
        ncw, L = 1, (1296 if n_avbits >= n_pld + 912 * (1 - rate) else 648)
    elif n_avbits <= 1296:
        ncw, L = 1, (1944 if n_avbits >= n_pld + 1464 * (1 - rate) else 1296)
    elif n_avbits <= 1944:
        ncw, L = 1, 1944
    elif n_avbits <= 2592:
        ncw, L = 2, (1944 if n_avbits >= n_pld + 2916 * (1 - rate) else 1296)
    else:
        ncw, L = math.ceil(n_pld / (1944 * rate)), 1944
    k = int(L * rate)
    n_shrt = max(0, ncw * k - n_pld)
    n_punc = max(0, ncw * L - n_avbits - n_shrt)
    need_extra = ((n_punc > 0.1 * ncw * L * (1 - rate) and n_shrt < 1.2 * n_punc * rate / (1 - rate))
                  or n_punc > 0.3 * ncw * L * (1 - rate))
    if (extra if vht else need_extra):
        n_avbits += n_cbps * m_stbc
        n_punc = max(0, ncw * L - n_avbits - n_shrt)
    n_rep = max(0, int(round(float(n_avbits - ncw * L * (1 - rate) - n_pld))))

    def spread(total):
        return [total // ncw + (1 if i < total % ncw else 0) for i in range(ncw)]

    return dict(ncw=ncw, L=L, k=k, shrt=spread(n_shrt), punc=spread(n_punc), rep=spread(n_rep),
                n_avbits=int(n_avbits), n_pld=int(n_pld))


def ldpc_decode_stream(llr, lay):
    """llr: soft bits of the whole Data field in transmit order, positive -> 1.
    Returns (n_pld info bits, number of codewords that failed their parity checks)."""
    code = ldpc_code(lay["L"], Fraction(lay["k"], lay["L"]))
    L, k = lay["L"], lay["k"]
    cws = np.zeros((lay["ncw"], L))
    pos = 0
    for i in range(lay["ncw"]):
        s, p, r = lay["shrt"][i], lay["punc"][i], lay["rep"][i]
        ni, npar = k - s, L - k - p
        seg = llr[pos:pos + ni + npar + r]
        if len(seg) < ni + npar + r:
            seg = np.concatenate([seg, np.zeros(ni + npar + r - len(seg))])
        pos += ni + npar + r
        cw = np.zeros(L)
        cw[:ni] = seg[:ni]
        cw[ni:k] = -1e3                         # shortened bits are known zeros
        cw[k:k + npar] = seg[ni:ni + npar]      # punctured parity stays at 0
        if r:
            src = np.concatenate([np.arange(ni), np.arange(k, k + npar)])
            np.add.at(cw, src[np.arange(r) % len(src)], seg[ni + npar:])
        cws[i] = cw
    bits, ok = code.decode(-cws)
    info = np.concatenate([bits[i, :k - lay["shrt"][i]] for i in range(lay["ncw"])])
    return info[:lay["n_pld"]], int((~ok).sum())


# ---------------------------------------------------------------------------
# Frame record
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    index: float            # position in the 20/22 MSps channel stream
    data: bytes
    phy: str                # 'ofdm', 'ht', 'vht', 'dsss'
    rate: float = 0.0       # Mbps for ofdm/dsss
    mcs: int = -1
    sgi: bool = False
    short_preamble: bool = False
    bw: int = 20            # PPDU bandwidth in MHz
    stbc: bool = False
    ldpc: bool = False
    center_mhz: float = None  # centre of a 40/80 MHz PPDU
    group_id: int = 0
    partial_aid: int = 0
    beamformed: bool = False
    disamb: bool = False
    ldpc_extra: bool = False
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
# HT/VHT numerology: 20, 40 and 80 MHz tone plans, training fields, pilots
# ---------------------------------------------------------------------------

_LNEG = list(LTF_L[:26])        # legacy L-LTF on k = -26..-1
_LPOS = list(LTF_L[27:])        # legacy L-LTF on k = 1..26
_LBLK = _LNEG + [1] + _LPOS     # 53 values, used as a building block for wide LTFs


def _ltf_values(bw):
    if bw == 20:
        return -28, [1, 1] + _LNEG + [0] + _LPOS + [-1, -1]
    if bw == 40:
        return -58, _LBLK + [-1, -1, -1, 1, 0, 0, 0, -1, 1, 1, -1] + _LBLK
    a = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1]
    return -122, _LBLK + a + _LBLK + [1, -1, 1, -1, 0, 0, 0, 1, -1, -1, 1] + _LBLK + a + _LBLK


class TonePlan:
    PILOTS = {20: [-21, -7, 7, 21], 40: [-53, -25, -11, 11, 25, 53],
              80: [-103, -75, -39, -11, 11, 39, 75, 103]}
    INTERLEAVER = {20: (13, 4), 40: (18, 6), 80: (26, 9)}   # (N_COL, N_ROW / N_BPSCS)
    D_TM = {20: 4, 40: 6, 80: 9}                             # VHT LDPC tone mapping distance

    def __init__(self, bw):
        self.bw, self.u = bw, bw // 20
        self.N = 64 * self.u
        k0, vals = _ltf_values(bw)
        self.ltf = np.zeros(self.N)
        for i, v in enumerate(vals):
            self.ltf[(k0 + i) % self.N] = v
        used = [k0 + i for i, v in enumerate(vals) if v != 0]
        self.pil_k = np.array(self.PILOTS[bw])
        self.data_k = np.array([k for k in used if k not in self.PILOTS[bw]])
        self.used_k = np.array(used)
        self.pil_i, self.data_i, self.used_i = self.pil_k % self.N, self.data_k % self.N, self.used_k % self.N
        self.nsd = len(self.data_k)
        # contiguous runs of used tones, for channel smoothing; an 80 MHz PPDU also
        # flips the sign of tones k >= -64 (tone rotation), so never smooth across that
        self.runs, run = [], [used[0]]
        for k in used[1:]:
            if k == run[-1] + 1 and not (bw == 80 and k == -64):
                run.append(k)
            else:
                self.runs.append(np.array(run))
                run = [k]
        self.runs.append(np.array(run))

    def perm(self, nbpsc):
        ncol, rowf = self.INTERLEAVER[self.bw]
        ncbps = self.nsd * nbpsc
        k = np.arange(ncbps)
        s = max(nbpsc // 2, 1)
        i = rowf * nbpsc * (k % ncol) + k // ncol
        return s * (i // s) + (i + ncbps - (ncol * i) // ncbps) % s

    def tone_map(self):
        d = self.D_TM[self.bw]
        k = np.arange(self.nsd)
        return d * (k % (self.nsd // d)) + (k * d) // self.nsd


PLANS = {bw: TonePlan(bw) for bw in (20, 40, 80)}

# HT pilot patterns per N_STS (rows = space-time stream); VHT uses one pattern for every stream.
HT_PILOTS = {
    (20, 1): [[1, 1, 1, -1]], (20, 2): [[1, 1, -1, -1], [1, -1, -1, 1]],
    (40, 1): [[1, 1, 1, -1, -1, 1]], (40, 2): [[1, 1, -1, -1, -1, -1], [1, 1, 1, -1, 1, 1]],
}
VHT_PILOTS = {20: [1, 1, 1, -1], 40: [1, 1, 1, -1, -1, 1], 80: [1, 1, 1, -1, -1, 1, 1, 1]}
P_LTF = np.array([[1, -1], [1, 1]])      # P_HTLTF / P_VHTLTF for 2 streams: [sts][ltf]
R_LTF = np.array([1, -1])                # VHT pilot tones use the first row of P
VHT_MCS = HT_MCS + [(8, Fraction(3, 4)), (8, Fraction(5, 6))]
ELTF = {0: 0, 1: 1, 2: 2, 3: 4}
VHT_SIGB_LEN_BITS = {20: 17, 40: 19, 80: 21}
VHT_SIGB_BITS = {20: 26, 40: 27, 80: 29}


def demap_any(z, nbpsc):
    if nbpsc <= 6:
        return demap(z, nbpsc)
    s = math.sqrt(170)
    i, q = z.real * s, z.imag * s

    def f(x):
        a = np.abs(x)
        return [x, 8 - a, 4 - np.abs(a - 8), 2 - np.abs(np.abs(a - 8) - 4)]

    return np.stack(f(i) + f(q), 1)


def wide_centers(ch, fmhz, bw):
    """Candidate centre frequencies (MHz) of the bw-MHz channel containing 20 MHz channel ch."""
    if fmhz > 4000:
        base = 36 if ch < 149 else 149
        if bw == 40:
            c = base + 8 * ((ch - base) // 8) + 2
        else:
            c = base + 16 * ((ch - base) // 16) + 6
        return [5000 + 5 * c]
    return [fmhz + 10, fmhz - 10] if bw == 40 else []


# ---------------------------------------------------------------------------
# OFDM receiver (detection and legacy part at 20 MSps; HT/VHT at 20/40/80 MSps)
# ---------------------------------------------------------------------------


class OfdmRx:
    BK = 3  # FFT window starts this many (20 MSps) samples inside the cyclic prefix

    def __init__(self, keep_bad, stats):
        self.keep_bad = keep_bad
        self.stats = stats
        self.ltf_t = np.fft.ifft(LTF64)
        self.ltf_norm = np.linalg.norm(self.ltf_t)
        self.perm_sig = interleave_perm(48, 1, False)

    def _count(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n

    @staticmethod
    def _ma(v, n):
        c = np.cumsum(np.concatenate([[0], v]))
        return c[n:] - c[:-n]

    def process(self, x, lo, hi, ctx=None):
        """Decode packets whose preamble starts in x[lo:hi]. Returns a list of Frames."""
        self.ctx = ctx
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
            if s < lo or s >= hi or s < next_free:
                continue
            if cm[min(s + 64, len(mask))] - cm[s] < 48:
                continue
            if ctx is not None:
                claimed = ctx.claimed(int(s))
                if claimed is not None:
                    next_free = claimed
                    continue
            got, end = self._decode(x, int(s))
            frames.extend(got)
            next_free = end
        return frames

    # -- legacy helpers ---------------------------------------------------

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

    def _sig48(self, y, start, H, wts, pol, quad=False):
        didx = DATA_K_L % 64
        z = self._eq(y, start, 16, H, PILOT_PSI * PILOT_POL[pol], didx, DATA_K_L)
        return z, ((z.imag if quad else z.real) * wts)[self.perm_sig]

    @staticmethod
    def _crc_ok(bits):
        crc = crc8(bits[:34])
        return (np.array_equal(bits[34:42], msb_bits(crc, 8)) or
                np.array_equal(bits[34:42], int_to_bits(crc, 8))) and not bits[42:48].any()

    # -- packet decode ----------------------------------------------------

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
        z0, llr = self._sig48(y, t_sig, H, wts, 0)
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
        pk = dict(x=x, s=s, w=w, t_sig=t_sig, l_len=length, l_end=end, meta=meta)

        if mbps == 6:
            # QBPSK tests are weighted by |H|^2: with multi-antenna cyclic shift
            # diversity the legacy channel has deep nulls that would dominate otherwise
            def quadrature(z):
                return np.sum(wts * z.imag ** 2) > 2 * np.sum(wts * z.real ** 2)

            z2, llr2 = self._sig48(y, t_sig + 80, H, wts, 1, quad=True)
            if quadrature(z2):
                _, llr3 = self._sig48(y, t_sig + 160, H, wts, 2, quad=True)
                return self._ht(pk, VITERBI.decode(np.concatenate([llr2, llr3])))
            z3, llr3 = self._sig48(y, t_sig + 160, H, wts, 2, quad=True)
            if quadrature(z3):
                _, llr2r = self._sig48(y, t_sig + 80, H, wts, 1)
                va = VITERBI.decode(np.concatenate([llr2r, llr3]))
                if self._crc_ok(va):
                    return self._vht(pk, va)
                self._count("vht_siga_crc_fail")
                return [], end
            corr = abs(np.vdot(z0 * wts, z2)) / (np.linalg.norm(z0 * wts) * np.linalg.norm(z2) + 1e-30)
            if corr > 0.8:  # RL-SIG repeats L-SIG: HE PPDU
                self._count("he_unsupported")
                return [], end

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
        self._count("ofdm_ppdu")
        return self._emit([psdu], s, "ofdm", meta, rate=mbps), end

    # -- HT ---------------------------------------------------------------

    def _ht(self, pk, hb):
        if not self._crc_ok(hb):
            self._count("ht_sig_crc_fail")
            return [], pk["l_end"]
        mcs = bits_to_int(hb[0:7])
        bw = 40 if hb[7] else 20
        length = bits_to_int(hb[8:24])
        smoothing = bool(hb[24])
        aggregation = bool(hb[27])
        stbc = bits_to_int(hb[28:30])
        ldpc = bool(hb[30])
        sgi = bool(hb[31])
        ness = bits_to_int(hb[32:34])
        if mcs > 7 or stbc > 1 or length == 0:
            # MCS 8+ means 2+ spatial streams: not separable with one receive chain
            self._count("ht_unsupported_mimo" if mcs > 7 else "ht_unsupported")
            return [], pk["l_end"]
        nsts = 1 + stbc
        nbpsc, cr = HT_MCS[mcs]
        plan = PLANS[bw]
        ncbps = plan.nsd * nbpsc
        ndbps = int(ncbps * cr)
        m_stbc = 2 if stbc else 1
        if ldpc:
            lay = ldpc_layout(8 * length + 16, ncbps, cr, m_stbc)
            nsym = lay["n_avbits"] // ncbps
        else:
            lay = None
            nsym = m_stbc * -(-(16 + 8 * length + 6) // (m_stbc * ndbps))
        nltf = 2 if nsts == 2 else 1
        p = dict(fmt="ht", bw=bw, nsts=nsts, stbc=bool(stbc), ldpc=ldpc, sgi=sgi, nbpsc=nbpsc, cr=cr,
                 ndbps=ndbps, nsym=nsym, nltf=nltf, ltf0=320, data0=320 + 80 * (nltf + ELTF[ness]),
                 pol0=3, smooth=smoothing and nsts == 1, lay=lay, mcs=mcs, length=length,
                 aggregation=aggregation)
        return self._nonlegacy(pk, p)

    # -- VHT --------------------------------------------------------------

    def _vht(self, pk, va):
        bw = {0: 20, 1: 40, 2: 80}.get(bits_to_int(va[0:2]))
        stbc = bool(va[3])
        group = bits_to_int(va[4:10])
        nsts = bits_to_int(va[10:13]) + 1
        paid = bits_to_int(va[13:22])
        sgi, disamb = bool(va[24]), bool(va[25])
        ldpc, extra = bool(va[26]), bool(va[27])
        mcs = bits_to_int(va[28:32])
        beamformed = bool(va[32])
        if bw is None:
            self._count("vht160_unsupported")
            return [], pk["l_end"]
        if group not in (0, 63):
            self._count("vht_mu_unsupported")
            return [], pk["l_end"]
        nss = nsts // 2 if stbc else nsts
        if nss != 1 or nsts > 2 or mcs > 9:
            self._count("vht_unsupported_mimo")
            return [], pk["l_end"]
        nbpsc, cr = VHT_MCS[mcs]
        plan = PLANS[bw]
        ncbps = plan.nsd * nbpsc
        ndbps = int(ncbps * cr)
        m_stbc = 2 if stbc else 1
        nltf = 2 if nsts == 2 else 1
        txtime = (pk["l_len"] + 3) // 3 * 4 + 20
        t_pre = 20 + 8 + 4 + 4 * nltf + 4
        nsym = int((txtime - t_pre) / 3.6) - int(disamb) if sgi else (txtime - t_pre) // 4
        if nsym <= 0:
            return [], pk["l_end"]
        lay = None
        if ldpc:
            nsym_init = nsym - m_stbc * int(extra)
            lay = ldpc_layout(nsym_init * ndbps, ncbps, cr, m_stbc,
                              n_avbits=nsym_init * ncbps, extra=extra)
        p = dict(fmt="vht", bw=bw, nsts=nsts, stbc=stbc, ldpc=ldpc, sgi=sgi, nbpsc=nbpsc, cr=cr,
                 ndbps=ndbps, nsym=nsym, nltf=nltf, ltf0=320, sigb=320 + 80 * nltf,
                 data0=320 + 80 * nltf + 80, pol0=4, smooth=not beamformed and nsts == 1, lay=lay,
                 mcs=mcs, aggregation=True, group=group, paid=paid, beamformed=beamformed,
                 disamb=disamb, extra=extra)
        return self._nonlegacy(pk, p)

    # -- shared HT/VHT data path -----------------------------------------

    def _stream(self, pk, bw):
        """Return (stream, start index, rad/sample CFO, centre MHz) for the PPDU's bandwidth."""
        base = pk["s"] + pk["t_sig"]   # HT/VHT field offsets are relative to the L-SIG start
        if bw == 20:
            return [(pk["x"], base, pk["w"], None)]
        ctx = self.ctx
        if ctx is None:
            return []
        out = []
        for fc in wide_centers(ctx.ch, ctx.fmhz, bw):
            st = ctx.wide(fc, bw)
            if st is not None:
                u = bw // 20
                out.append((st, base * u, pk["w"] / u, fc))
        return out

    def _chan(self, y, plan, p, bk):
        """Per-stream channel estimates from the HT/VHT-LTFs. Returns (Hs list, Heff for pilots)."""
        u = plan.u
        Ys = [np.fft.fft(y[(p["ltf0"] + 80 * i) * u + 16 * u - bk:][:plan.N]) for i in range(p["nltf"])]
        L = plan.ltf
        used = L != 0
        if p["nsts"] == 1:
            H = np.zeros(plan.N, complex)
            H[used] = Ys[0][used] / L[used]
            if p["smooth"]:
                H = self._smooth(H, plan, bk)
            return [H], H
        H0 = np.zeros(plan.N, complex)
        H1 = np.zeros(plan.N, complex)
        H0[used] = (Ys[0][used] - Ys[1][used]) / (2 * L[used])
        H1[used] = (Ys[0][used] + Ys[1][used]) / (2 * L[used])
        if p["fmt"] == "vht":
            # pilot tones carry R (same polarity on all streams): they see H0 + H1
            heff = H0 + H1
            pi = plan.pil_i
            heff[pi] = (Ys[0][pi] - Ys[1][pi]) / (2 * L[pi])
            H0[pi], H1[pi] = heff[pi] / 2, heff[pi] / 2
            return [H0, H1], heff
        return [H0, H1], None

    @staticmethod
    def _smooth(H, plan, bk):
        k = np.arange(plan.N)
        k = np.where(k < plan.N // 2, k, k - plan.N)
        ramp = np.exp(-2j * np.pi * k * bk / plan.N)
        G = H / ramp
        out = H.copy()
        for run in plan.runs:
            g = G[run % plan.N]
            pad = np.concatenate([[g[0]], g, [g[-1]]])
            out[run % plan.N] = (pad[:-2] + 2 * pad[1:-1] + pad[2:]) / 4 * ramp[run % plan.N]
        return out

    def _pilot_expect(self, plan, p, Hs, heff, n_idx, pol_idx):
        """Expected received pilots for symbols n_idx: shape (len(n_idx), n_pilots)."""
        npil = len(plan.pil_k)
        m = np.arange(npil)
        rot = (np.asarray(n_idx)[:, None] + m[None, :]) % npil
        pol = PILOT_POL[np.asarray(pol_idx) % 127][:, None]
        if p["fmt"] == "vht":
            psi = np.array(VHT_PILOTS[plan.bw])
            return heff[plan.pil_i][None, :] * psi[rot] * pol
        tab = np.array(HT_PILOTS[(plan.bw, p["nsts"])])
        ex = 0
        for i in range(p["nsts"]):
            ex = ex + Hs[i][plan.pil_i][None, :] * tab[i][rot] * pol
        return ex

    def _symbols(self, y, plan, starts, cp, bk):
        idx = np.asarray(starts)[:, None] + (cp - bk) + np.arange(plan.N)[None, :]
        return np.fft.fft(y[idx], axis=1)

    def _track(self, Y, plan, ex):
        """Common phase + timing slope from pilots; returns correction (nsym, N)."""
        c = Y[:, plan.pil_i] * np.conj(ex)
        ph0 = np.angle(c.sum(1))
        res = np.angle(c * np.exp(-1j * ph0)[:, None])
        wt = np.abs(ex) ** 2
        kp = plan.pil_k[None, :]
        slope = (wt * res * kp).sum(1) / np.maximum((wt * kp ** 2).sum(1), 1e-30)
        k = np.arange(plan.N)
        k = np.where(k < plan.N // 2, k, k - plan.N)
        return np.exp(-1j * (ph0[:, None] + slope[:, None] * k[None, :]))

    def _vht_sigb(self, y, plan, p, Hs, heff, bk):
        u = plan.u
        st = np.array([p["sigb"] * u])
        Y = self._symbols(y, plan, st, 16 * u, bk)
        ex = self._pilot_expect(plan, p, Hs, heff, [0], [3])
        Y = Y * self._track(Y, plan, ex)
        hd = Hs[0] if p["nsts"] == 1 else Hs[0] + Hs[1]
        di = plan.data_i
        z = Y[0, di] / hd[di]
        llr = (z.real * np.abs(hd[di]) ** 2)[plan.perm(1)]
        bits = VITERBI.decode(llr)
        nb = VHT_SIGB_BITS[plan.bw]
        sb = bits[:nb]
        return sb, bits_to_int(sb[:VHT_SIGB_LEN_BITS[plan.bw]]) * 4

    def _nonlegacy(self, pk, p):
        cands = self._stream(pk, p["bw"])
        if not cands:
            self._count("wide_out_of_span" if p["bw"] > 20 else "nonlegacy_no_stream")
            return [], pk["l_end"]
        plan = PLANS[p["bw"]]
        u = plan.u
        bk = self.BK * u
        symlen, cp = (72 * u, 8 * u) if p["sgi"] else (80 * u, 16 * u)
        end20 = pk["t_sig"] + p["data0"] + (p["nsym"] * symlen) // u
        if self.ctx is not None and p["bw"] > 20:
            self.ctx.claim(pk["s"], pk["s"] + end20 + 80, cands, p["bw"])
        best = None
        for st, s_w, w_w, fc in cands:
            need = (p["data0"] * u + p["nsym"] * symlen + 16 * u)
            if s_w + need > len(st):
                continue
            y = st[s_w:s_w + need] * np.exp(-1j * w_w * np.arange(need))
            Hs, heff = self._chan(y, plan, p, bk)
            # for a 2.4 GHz HT40 the secondary side is unknown: keep the candidate whose
            # two halves both carry signal
            half = plan.N // 2
            pw = np.abs(Hs[0]) ** 2
            lo_p = pw[plan.used_i[plan.used_k < 0]].mean()
            hi_p = pw[plan.used_i[plan.used_k > 0]].mean()
            score = abs(math.log10(max(lo_p, 1e-30) / max(hi_p, 1e-30)))
            if best is None or score < best[0]:
                best = (score, y, Hs, heff, fc)
        if best is None:
            return [], pk["l_end"]
        _, y, Hs, heff, fc = best
        apep = None
        sigb = None
        if p["fmt"] == "vht":
            sigb, apep = self._vht_sigb(y, plan, p, Hs, heff, bk)
        nsym = p["nsym"]
        starts = p["data0"] * u + np.arange(nsym) * symlen
        Y = self._symbols(y, plan, starts, cp, bk)
        n_idx = np.arange(nsym)
        ex = self._pilot_expect(plan, p, Hs, heff, n_idx, n_idx + p["pol0"])
        Y = Y * self._track(Y, plan, ex)
        di = plan.data_i
        if p["stbc"]:
            H0, H1 = Hs[0][di], Hs[1][di]
            g = np.abs(H0) ** 2 + np.abs(H1) ** 2
            Y0, Y1 = Y[0::2, di], Y[1::2, di]
            s0 = (np.conj(H0) * Y0 + H1 * np.conj(Y1)) / g
            s1 = (np.conj(H0) * Y1 - H1 * np.conj(Y0)) / g
            z = np.empty((nsym, plan.nsd), complex)
            z[0::2], z[1::2] = s0, s1
            wts = g
        else:
            hd = Hs[0][di]
            z = Y[:, di] / hd
            wts = np.abs(hd) ** 2
        wts = wts / max(float(np.median(wts)), 1e-30)
        nb = p["nbpsc"]
        soft = demap_any(z.reshape(-1), nb).reshape(nsym, plan.nsd, nb) * wts[None, :, None]
        if p["ldpc"]:
            maps = [None]
            if p["fmt"] == "vht":
                tm = plan.tone_map()
                inv = np.empty_like(tm)
                inv[tm] = np.arange(len(tm))
                maps = [tm, inv]
            info, bad = None, None
            for mp in maps:
                sm = soft if mp is None else soft[:, mp, :]
                i2, b2 = ldpc_decode_stream(sm.reshape(-1), p["lay"])
                if bad is None or b2 < bad:
                    info, bad = i2, b2
                if b2 == 0:
                    break
            if bad:
                self._count("ldpc_codeword_fail", bad)
            bits = ofdm_descramble(info)
            psdu_bits = len(bits) - 16
        else:
            perm = plan.perm(nb)
            llr = soft.reshape(nsym, -1)[:, perm].reshape(-1)
            full = depuncture(llr, p["cr"], nsym * p["ndbps"])
            bits = ofdm_descramble(VITERBI.decode(full))
            psdu_bits = nsym * p["ndbps"] - 16 - 6
        if p["fmt"] == "ht":
            psdu = bits_to_bytes(bits[16:16 + 8 * p["length"]])
        else:
            psdu = bits_to_bytes(bits[16:16 + 8 * (psdu_bits // 8)])
            if apep:
                psdu = psdu[:apep]
            if not np.array_equal(bits[8:16], msb_bits(crc8(sigb[:VHT_SIGB_BITS[plan.bw] - 6]), 8)) and \
                    not np.array_equal(bits[8:16], int_to_bits(crc8(sigb[:VHT_SIGB_BITS[plan.bw] - 6]), 8)):
                self._count("vht_sigb_crc_mismatch")
        self._count(p["fmt"] + "_ppdu")
        mpdus = split_ampdu(psdu) if p["aggregation"] else [psdu]
        if p["aggregation"] and not mpdus:
            self._count("ampdu_no_delimiters")
        kw = dict(mcs=p["mcs"], sgi=p["sgi"], bw=p["bw"], stbc=p["stbc"], ldpc=p["ldpc"],
                  center_mhz=fc)
        if p["fmt"] == "vht":
            kw.update(group_id=p["group"], partial_aid=p["paid"], beamformed=p["beamformed"],
                      disamb=p["disamb"], ldpc_extra=p["extra"])
        frames = self._emit(mpdus, pk["s"], p["fmt"], pk["meta"], **kw)
        return frames, pk["s"] + end20

    def _emit(self, mpdus, s, phy, meta, **kw):
        out = []
        for m in mpdus:
            ok = fcs_ok(m)
            self._count("fcs_ok" if ok else "fcs_bad")
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
    """Channels whose OFDM occupied band (+-8.4 MHz) lies inside edge * fs / 2."""
    half = edge * fs / 2
    out = []
    for c, f in all_channels():
        off = f * 1e6 - center_hz
        if wanted is not None and c not in wanted:
            continue
        if abs(off) + 8.4e6 <= half or (wanted is not None and abs(off) + 8.4e6 <= fs / 2):
            out.append((c, f))
    return out


class BlockCtx:
    """Per-block access to wider channels (40/80 MHz) cut from the same block FFT,
    plus a registry of wide PPDUs so overlapping 20 MHz sub-channels skip them."""

    OCCUPIED = {40: 18.2e6, 80: 38.2e6}   # half-width of the used tones

    def __init__(self, X, n, fs_nom, fs, fc, a, length, claims):
        self.X, self.n, self.fs_nom, self.fs, self.fc = X, n, fs_nom, fs, fc
        self.a, self.length, self.claims = a, length, claims
        self.cache = {}
        self.ch, self.fmhz = 0, 0

    def set_channel(self, ch, fmhz):
        self.ch, self.fmhz = ch, fmhz

    def _raw(self, s20):
        return self.a + s20 * self.fs_nom / 20e6

    def wide(self, fc_mhz, bw):
        key = (fc_mhz, bw)
        if key in self.cache:
            return self.cache[key]
        off = fc_mhz * 1e6 - self.fc
        ratio = Fraction(int(bw * 1e6), int(self.fs_nom))
        y = None
        if ratio <= 1 and abs(off) + self.OCCUPIED[bw] <= self.fs_nom / 2 and (self.n * ratio).denominator == 1:
            m = int(self.n * ratio)
            y = extract(self.X, self.n, self.fs_nom, off * self.fs_nom / self.fs, bw * 1e6, m,
                        bw / 2 * 1e6 - 1.0e6, bw / 2 * 1e6)
            y = y[: int(self.length * float(ratio)) + 1].astype(np.complex128)
        self.cache[key] = y
        return y

    def claim(self, s0, s1, cands, bw):
        r0, r1 = self._raw(s0), self._raw(s1)
        for _, _, _, fc in cands:
            self.claims.append((r0, r1, fc - bw / 2, fc + bw / 2))

    def claimed(self, s20):
        r = self._raw(s20)
        tol = 4e-6 * self.fs_nom
        for r0, r1, flo, fhi in self.claims:
            if abs(r - r0) < tol and flo <= self.fmhz <= fhi:
                return int((r1 - self.a) * 20e6 / self.fs_nom)
        return None


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
    if fr.phy in ("ofdm", "dsss"):
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
        mflags = (1 if fr.bw == 40 else 0) | (0x04 if fr.sgi else 0) | (0x10 if fr.ldpc else 0) | \
            (0x20 if fr.stbc else 0)
        body += bytes([0x3F, mflags, fr.mcs])
    if fr.phy == "vht":
        present |= 1 << 21
        if (8 + len(body)) % 2:
            body.append(0)
        vflags = ((0x01 if fr.stbc else 0) | (0x04 if fr.sgi else 0) | (0x08 if fr.disamb else 0) |
                  (0x10 if fr.ldpc_extra else 0) | (0x20 if fr.beamformed else 0))
        body += struct.pack("<HBB4BBBH", 0x01FD, vflags, {20: 0, 40: 1, 80: 4}[fr.bw],
                            (fr.mcs << 4) | 1, 0, 0, 0, 1 if fr.ldpc else 0, fr.group_id, fr.partial_aid)
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
    ap.add_argument("--edge", type=float, default=0.98,
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
    ratios += [Fraction(int(r), int(fs_nom)) for r in (40e6, 80e6) if r <= fs_nom]
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
    claims = []
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
        claims[:] = [c for c in claims if c[1] > a]
        ctx = BlockCtx(X, n, fs_nom, fs, fc, a, L, claims)
        for ch, fmhz in chans:
            ctx.set_channel(ch, fmhz)
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
                got = orx.process(y, lo, hi, ctx) if rate_out == 20e6 else drx.process(y, lo, hi)
                for fr in got:
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
