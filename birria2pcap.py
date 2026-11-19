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
      HT-SIG, VHT-SIG-A or HE-SIG-A announces a 40 or 80 MHz PPDU, the wider
      channel is cut from the same block FFT and decoded there.
        - 802.11a/g legacy, 6-54 Mbps.
        - 802.11n HT-mixed: MCS 0-7 (one spatial stream), 20 and 40 MHz, long
          or short GI, BCC or LDPC, 2x1 STBC (Alamouti), A-MPDUs split.
        - 802.11ac VHT single-user: MCS 0-9 (one spatial stream, 256-QAM),
          20, 40 and 80 MHz, long or short GI, BCC or LDPC (with the LDPC tone
          mapper), 2x1 STBC, A-MPDUs split.
        - 802.11ax HE SU and HE ER SU: MCS 0-11 (one spatial stream, up to
          1024-QAM), 20, 40 and 80 MHz (ER SU: 242-tone or upper 106-tone RU),
          1x/2x/4x HE-LTF with 0.8/1.6/3.2 us GI, BCC or LDPC, 2x1 STBC, DCM,
          pre-FEC padding factor and packet extension, A-MPDUs split.
        - 802.11ax HE MU (downlink OFDMA): HE-SIG-B (MCS 0-5, both content
          channels, compressed full-band mode), RU allocation, then every
          single-stream user on its own 26/52/106/242/484/996-tone RU with
          its own MCS, DCM and coding; pilots of all RUs track the AP's phase.
        - 802.11ax HE TB (uplink OFDMA): an HE TB PPDU carries no per-user
          parameters, so they are taken from the Trigger frame that solicited
          it (matched by UL Length and a SIFS gap; the Trigger has to be
          decoded first, usually as a non-HT duplicate). Each single-stream
          user's RU is decoded with its own channel estimate and phase tracking.
    * DSSS/CCK: 802.11b, long and short preamble, 1, 2, 5.5 and 11 Mbps.

    Not decoded (detected and counted in the summary): two or more spatial
    streams (BirriaSDR records one receive chain, and one antenna cannot
    separate them), VHT MU-MIMO, users of an MU-MIMO RU (HE MU or TB), HE TB
    PPDUs whose Trigger was not decoded (or was solicited by a TRS Control
    field inside a data frame), STBC in HE MU / TB, HE-SIG-B with DCM,
    preamble puncturing, HE with Doppler midambles, 160 MHz, HT greenfield,
    PBCC. Frames that a
    multi-antenna AP sends as one stream (cyclic shift diversity, spatial
    expansion, beamforming, STBC) do decode. There is no multipath equalizer
    for DSSS.

    HE notes. Pilot phase is tracked without pilot values (squared BPSK pilots,
    continuity from the HE-LTF). The 4x and 2x 20 MHz HE-LTF sequences agree
    with two independent implementations (one derived from live frames); the
    1x and 40/80 MHz ones come from the same source but are unconfirmed. DCM:
    only the BPSK upper-half rule is confirmed; for QPSK/16-QAM the receiver
    picks the map per PPDU from candidates that fit the received symbols, and
    for MCS 0 with DCM on 242/106-tone RUs (odd N_CBPS, BCC) two readings of
    the standard are tried and the FCS decides.

    HE MU / TB notes. The SIG-B User fields of the centre 26-tone RU of an
    80 MHz PPDU are assumed to come last in content channel 1 (unconfirmed).
    One reference implementation alternates the sign of the upper half of
    HE-SIG-B; both readings are tried and the CRCs decide (counted as
    he_sigb_upper_alternating). With 1x HE-LTF (estimate on every 4th tone)
    small RUs at high MCS can fail under multipath; tested up to MCS 5 there.
    MCS 0 with DCM on a 26-tone RU (N_DBPS,short = 1.5) is not handled.

    40 and 80 MHz: the whole wide channel must lie inside the captured span.
    An 80 MHz PPDU needs an 80 MSps capture centred on the 80 MHz channel
    (5210, 5290, 5530, 5610, 5690, 5775 MHz); its outer tones then sit at the
    edge of BirriaSDR's passband, which is not guaranteed flat there. In
    2.4 GHz a 40 MHz PPDU's secondary channel can be above or below; both are
    tried. The radiotap channel of a 40/80 MHz PPDU is the 20 MHz sub-channel
    where it was first detected, which is not necessarily the primary.

Output
    pcap (nanosecond timestamps) with a radiotap header per frame: flags (FCS
    present), legacy rate, HT MCS field, VHT field or HE field (format, MCS,
    DCM, coding, STBC, bandwidth/RU, RU offset for MU / TB, GI, LTF size, BSS
    colour, pre-FEC padding, PE disambiguity, TXOP), channel frequency, and dB_antsignal/dB_antnoise.
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

    def decode(self, llr, terminated=False):
        """terminated: the encoder ended in state 0 (zero tail bits), so trace back from it."""
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
        s = 0 if terminated else int(np.argmax(pm))
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


def ldpc_layout_explicit(n_pld, av_init, av_final, rate):
    """HE (27.3.12): the codeword count and length come from the initial available
    bits, puncturing and repetition from the final ones (after any extra symbol segment)."""
    if av_init <= 648:
        ncw, L = 1, (1296 if av_init >= n_pld + 912 * (1 - rate) else 648)
    elif av_init <= 1296:
        ncw, L = 1, (1944 if av_init >= n_pld + 1464 * (1 - rate) else 1296)
    elif av_init <= 1944:
        ncw, L = 1, 1944
    elif av_init <= 2592:
        ncw, L = 2, (1944 if av_init >= n_pld + 2916 * (1 - rate) else 1296)
    else:
        ncw, L = math.ceil(n_pld / (1944 * rate)), 1944
    k = int(L * rate)
    n_shrt = max(0, ncw * k - n_pld)
    n_punc = max(0, ncw * L - av_final - n_shrt)
    n_rep = max(0, int(round(float(av_final - ncw * L * (1 - rate) - n_pld))))

    def spread(total):
        return [total // ncw + (1 if i < total % ncw else 0) for i in range(ncw)]

    return dict(ncw=ncw, L=L, k=k, shrt=spread(n_shrt), punc=spread(n_punc), rep=spread(n_rep),
                n_avbits=int(av_final), n_pld=int(n_pld))


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
    dcm: bool = False         # HE fields
    he_er: bool = False
    he_ru: str = ""
    he_fmt: str = ""          # su, er, mu, tb
    he_ru_off26: int = -1     # RU offset in 26-tone units (MU / TB)
    he_ru_code: int = -1      # radiotap RU size code (MU / TB)
    he_sta_id: int = -1
    bss_color: int = 0
    ul_dl: int = 0
    beam_change: int = 0
    spatial_reuse: int = 0
    txop: int = 127
    gi_samples: int = 16
    ltf_mult: int = 2
    n_ltf: int = 1
    pre_fec_a: int = 4
    pe_disamb: bool = False
    nsts: int = 1
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
# HE (802.11ax) numerology: HE-LTF sequences, resource units, MCS, demapping
# ---------------------------------------------------------------------------

# HE-LTF sequences (IEEE 802.11ax 27.3.11.10) for 1x, 2x and 4x HE-LTF at 20, 40
# and 80 MHz, one character per subcarrier from -K to +K ('+' = +1, '-' = -1, '0' = 0).
HE_LTF_TEXT = {
    (20, 1): (
        "00-000+000+000-000+000-000+000+000+000+000-000-000+000+000+000-000-000-000+000-000-000+000+000-0"
        "00-000+000-000-000+000-0000000-000+000+000+000+000+000+000-000-000-000-000-000+000-000-000-000+0"
        "00-000-000+000-000-000+000-000+000-000-000-000-000-00"
    ),
    (20, 2): (
        "-0-0-0+0+0-0+0-0-0-0-0+0-0+0-0-0+0+0-0+0+0+0+0+0-0+0-0+0-0-0+0+0-0+0-0-0-0-0+0-0+0+0+0-0-0+0-0-0"
        "-0-0-0+0-0-0-0+0+0+0-0-0+000+0-0+0+0-0+0+0-0+0+0-0-0+0-0+0+0+0+0-0+0-0+0+0-0-0+0-0-0-0-0-0+0-0+0"
        "+0-0-0+0+0-0+0-0-0-0-0+0-0+0+0+0-0-0+0-0-0-0-0-0+0-0+"
    ),
    (20, 4): (
        "--+-+-+++-+++--+-----++----++-+-++++-+--++-++++--+---++++-++----+--++-+----+-+------++-----+--++"
        "+-+++-+-+-----+++---+-+++000-+-+-++-+++--+--+-+-+++-+++--+-----++------+-+----+-++--+----++-++++"
        "+++-++----+--++-+----+-+--++++--+++++-++---+---+-+-++"
    ),
    (40, 1): (
        "+000+000+000+000-000+000+000-000-000+000-000+000-000+000-000-000+000-000+000+000+000-000-000+000"
        "+000+000+000+000+000+000+000-000+000+000-000-000+000-000+000-000+000-000-000+000-000+000+000+000"
        "-000-000+000+000+000+000-000+000-000+000-000-000-0000000+000+000-000-000+000-000+000+000+000+000"
        "+000+000-000+000+000-000-000+000-000+000+000+000+000+000-000+000-000-000-000+000+000-000-000-000"
        "+000-000-000-000-000+000-000-000+000+000-000+000-000-000-000-000-000+000-000+000+000+000-000-000"
        "+000+000+"
    ),
    (40, 2): (
        "+0-0-0-0-0-0+0+0-0-0+0-0+0-0-0-0-0-0-0+0+0+0+0-0+0-0+0+0-0-0-0-0-0+0+0-0-0+0-0+0-0+0+0+0+0+0-0-0"
        "-0-0+0-0+0-0+0+0+0+0+0-0+0-0-0+0-0+0-0-0+0+0+0+0+0-0-0+0+0-0+0-0+0+0+0+0+0+0-0-0-0-0+0-0+0-0-0-0"
        "-0-0-0-0+0+0-0-0+0-0+0-0+0+0+0+0+0-0-0-0-0+0-0+0-0000000-0-0-0-0-0+0+0-0-0+0-0+0-0-0-0-0-0-0+0-0"
        "+0+0-0+0-0+0+0+0+0+0+0+0-0-0+0+0-0+0-0+0-0-0-0-0-0+0-0+0+0-0+0-0+0+0+0+0+0+0+0-0-0+0+0-0+0-0+0-0"
        "-0-0-0-0+0+0-0-0+0-0+0-0-0-0-0-0-0+0-0+0+0-0+0-0+0+0-0-0-0-0-0+0+0-0-0+0-0+0-0+0+0+0+0+0-0+0-0-0"
        "+0-0+0-0+"
    ),
    (40, 4): (
        "+----+--++-+-+-++-+---++-------+--++-+-----+-+++--+++--++++-++--+-+-+--+-+++--+++----+--++-+----"
        "-+-+++--++++++++-++--+-+++++-+---++--------+--++-+-+-++-+---++-------+--++-+-----+-+++--+++-----"
        "-+--++-+-+-++-+---++---++++-++--+-+++++-+---++---+00000-++++-++--+-+-+--+-+++--+++++++-++--+-+++"
        "++-+---++---++----+--++-+-+-++-+---++---++++-++--+-+++++-+---++---+----+--++-+-+-++-+---++---+--"
        "--+--++-+-+-++-+---++-------+--++-+-----+-+++--+++-+----+--++-+-+-++-+---++---++++-++--+-+++++-+"
        "---++----"
    ),
    (80, 1): (
        "-000-000+000+000+000+000+000-000-000-000+000+000-000-000+000-000+000-000-000-000-000-000-000+000"
        "+000-000-000+000-000+000-000-000-000-000-000+000+000-000-000+000-000+000-000+000+000+000+000+000"
        "-000-000+000+000-000+000-000+000-000-000-000-000+000-000-000-000-000-000-000+000+000+000-000-000"
        "+000+000-000+000-000+000+000+000+000+000+000-000-000+000+000-000+000-000+000+000+000+000-000+000"
        "+000-000-000+000-000+000-000+000+000+000+000+000-000-000+000+000-000+000-000+000-000-000-000-000"
        "+000-000+000-000-0000000-000+000+000-000-000+000+000-000-000+000+000-000+000-000+000+000+000+000"
        "+000+000-000-000+000+000-000+000-000+000+000+000+000+000-000-000+000+000-000+000-000+000-000-000"
        "-000-000-000+000+000-000-000+000-000+000-000+000+000+000+000-000+000-000-000+000-000-000+000+000"
        "+000-000-000+000+000-000+000-000+000+000+000+000+000+000-000-000+000+000-000+000-000+000+000+000"
        "+000-000+000+000-000-000+000-000+000-000+000+000+000+000+000-000-000+000+000-000+000-000+000-000"
        "-000-000-000+000-000+000-000-000-000+000+"
    ),
    (80, 2): (
        "+0+0-0+0+0+0-0+0+0+0+0-0-0-0+0-0-0+0+0-0+0+0+0-0-0-0-0+0+0+0+0-0+0+0+0-0+0-0-0+0-0-0-0+0+0-0-0-0"
        "+0-0-0+0-0+0+0-0+0+0-0+0-0-0-0+0-0-0-0-0+0-0-0+0+0-0+0+0+0-0-0-0-0+0-0+0+0+0-0+0+0+0+0-0-0-0+0+0"
        "+0-0-0+0+0+0-0+0+0-0+0-0-0-0-0+0-0-0-0+0-0+0+0-0+0+0-0-0-0+0-0-0-0+0-0+0+0-0+0+0+0-0-0+0+0+0-0+0"
        "+0-0+0-0-0+0-0+0+0+0-0+0+0+0+0-0-0-0+0-0-0+0+0-0+0+0+0-0-0-0-0+0+0+0-0-0-0-0+0-0-0-0+0-0+0+0+0-0"
        "-0+0+0+0-0+0+0-0+0-0-0-0-0+0-0-0-0+0-0+0+0-0+0+0-0+0+0-0-0+0-0-0-0+0+0+0+0-0+0-0-0-0+0-0-0-0-0+0"
        "+0+0+0-0-0-0+0+0+0000000+0-0-0+0+0-0+0-0-0-0+0+0+0+0-0+0+0+0-0+0-0-0-0-0+0+0+0-0+0+0-0-0+0-0-0+0"
        "-0-0+0-0+0+0+0-0+0+0+0+0-0+0-0-0+0-0-0-0+0+0-0-0-0+0-0+0+0+0-0+0+0+0+0-0-0-0+0+0+0+0-0-0-0+0-0-0"
        "+0+0-0+0+0+0-0-0-0-0+0-0-0-0+0-0+0+0-0+0-0-0+0-0-0-0+0+0-0-0-0+0-0-0+0-0+0+0+0-0+0+0+0-0+0-0+0+0"
        "-0+0-0-0-0+0-0-0-0-0+0-0+0+0-0+0+0+0-0-0+0+0+0-0-0-0+0+0+0+0-0+0+0+0-0+0-0-0-0-0+0+0+0-0+0+0-0-0"
        "+0-0-0-0-0+0-0-0-0+0-0+0+0-0+0+0-0+0-0-0+0-0-0-0+0+0-0-0-0+0-0-0+0-0+0+0+0-0+0+0+0+0-0-0-0-0+0+0"
        "+0-0+0+0-0-0+0-0-0-0+0+0+0+0-0+0+0+0-0+0+"
    ),
    (80, 4): (
        "++-+-+---+---++-+++++--++++-+-+--++-+++--+----+++------++++++-+++-++---+-+--++-+-+++++--+++-++--"
        "-+-++-++-+--++++--+++++-++---+---+-+-++-+-+-+++-+++--+-----++----+-+-++--+---++-++++---++++-++++"
        "+++-+++-++---+-+--++-+-+++++--+++-++---+-++-++-+--+-+-+-+++-+++--+-----++----+-+-++--+---++-++++"
        "---++++-+------+---+--+++-+-++--+-+-----++---+--+++-+--+--+-++++++--+++++-++---+---+-+-++-+-+-++"
        "+-+++--+-----++----+-+-++--+---++-++++---++++--++++++-+++-++---+-+--++-+-+++++--+++-++---+-++-++"
        "-+---+-+----+++--+00000+------+-++--++-+-++--+-+---++-+++-+++++++-+--+--+-+++--+---++-----+-----"
        "++-----+--+++-+++-+-+-----+++----+--+++-++--+-+-------++---+--+++-+--+--+-+++-+--++-+-+++--+---+"
        "-------+-++-++-+---++-+++--+++++-+----++-----+--+++-+++-+-+-----+++----+--+++-++--+-+-+++-+--++-"
        "+-+++--+---+-------+-++-++-+---++-+++--+++++-+++++--+++++-++---+---+-+-+++++---++++-++---+--++-+"
        "-+------++---+--+++-+--+--+-+-+-+--++-+-+++--+---+-------+-++-++-+---++-+++--+++++-+----++-----+"
        "--+++-+++-+-+-----+++----+--+++-++--+-+-+"
    ),
}


def _he_ltf(bw, mult):
    text = "".join(HE_LTF_TEXT[(bw, mult)])
    kmax = (len(text) - 1) // 2
    n = 256 * bw // 20
    out = np.zeros(n)
    for i, ch in enumerate(text):
        if ch != "0":
            out[(i - kmax) % n] = 1.0 if ch == "+" else -1.0
    return out


HE_LTF = {key: _he_ltf(*key) for key in HE_LTF_TEXT}
HE_SIGA_K = np.array([k for k in range(-28, 29) if k != 0 and k not in (-21, -7, 7, 21)])
PERM52 = interleave_perm(52, 1, True)
HE_MCS = VHT_MCS + [(10, Fraction(3, 4)), (10, Fraction(5, 6))]
HE_GI_LTF = {0: (1, 16), 1: (2, 16), 2: (2, 32), 3: (4, 64)}   # GI+LTF field -> (LTF size, GI samples @ 20 MSps)


class HeRu:
    """Tone plan of one HE resource unit as used by HE SU / ER SU PPDUs."""

    def __init__(self, bw, ranges, pilots, sd_short, bcc, bcc_dcm, dtm, dtm_dcm, ru_code):
        self.bw, self.u, self.N = bw, bw // 20, 256 * bw // 20
        tones = np.concatenate([np.arange(a, b + 1) for a, b in ranges])
        self.tones = tones
        self.pil_k = np.array(pilots)
        self.data_k = np.array([k for k in tones if k not in set(pilots)])
        self.nsd = len(self.data_k)
        self.sd_short, self.bcc, self.bcc_dcm = sd_short, bcc, bcc_dcm
        self.dtm, self.dtm_dcm, self.ru_code = dtm, dtm_dcm, ru_code
        self.pil_i, self.data_i, self.tone_i = self.pil_k % self.N, self.data_k % self.N, tones % self.N
        # Contiguous runs of tones for interpolation and smoothing. 80 MHz is also split
        # where the pre-HE tone rotation changes sign, in case a transmitter rotates HE fields too.
        self.runs, run = [], [tones[0]]
        for k in tones[1:]:
            if k == run[-1] + 1 and not (bw == 80 and k == -256):
                run.append(k)
            else:
                self.runs.append(np.array(run))
                run = [k]
        self.runs.append(np.array(run))

    def bcc_perm(self, nbpsc, dcm):
        ncol, rowf = self.bcc_dcm if dcm else self.bcc
        ncbps = ncol * rowf * nbpsc
        k = np.arange(ncbps)
        s = max(nbpsc // 2, 1)
        i = rowf * nbpsc * (k % ncol) + k // ncol
        return s * (i // s) + (i + ncbps - (ncol * i) // ncbps) % s

    def tone_map(self, dcm):
        n = self.nsd // 2 if dcm else self.nsd
        d = self.dtm_dcm if dcm else self.dtm
        k = np.arange(n)
        return d * (k % (n // d)) + (k * d) // n


HE_RUS = {
    "242": HeRu(20, [(-122, -2), (2, 122)], [-116, -90, -48, -22, 22, 48, 90, 116], 60, (26, 9), (13, 9),
                9, 9, 0),
    "106u": HeRu(20, [(17, 122)], [22, 48, 90, 116], 24, (17, 6), (17, 3), 6, 3, 6),
    "484": HeRu(40, [(-244, -3), (3, 244)], [-238, -212, -170, -144, -104, -78, -36, -10,
                                             10, 36, 78, 104, 144, 170, 212, 238], 120, None, None, 12, 9, 1),
    "996": HeRu(80, [(-500, -3), (3, 500)], [-468, -400, -334, -266, -226, -158, -92, -24,
                                             24, 92, 158, 226, 266, 334, 400, 468], 240, None, None, 20, 14, 2),
}


# Resource units of HE MU and HE TB PPDUs (802.11ax 27.3.2.2): tone ranges per bandwidth,
# indexed from the lowest frequency, and the pilot tones (27.3.11.13).
def _ru(*spans):
    return [tuple(x) for x in spans]


HE_RU_RANGES = {
    (20, 26): [_ru((-121, -96)), _ru((-95, -70)), _ru((-68, -43)), _ru((-42, -17)), _ru((-16, -4), (4, 16)),
               _ru((17, 42)), _ru((43, 68)), _ru((70, 95)), _ru((96, 121))],
    (20, 52): [_ru((-121, -70)), _ru((-68, -17)), _ru((17, 68)), _ru((70, 121))],
    (20, 106): [_ru((-122, -17)), _ru((17, 122))],
    (20, 242): [_ru((-122, -2), (2, 122))],
    (40, 26): [_ru((a, b)) for a, b in ((-243, -218), (-217, -192), (-189, -164), (-163, -138), (-136, -111),
                                         (-109, -84), (-83, -58), (-55, -30), (-29, -4), (4, 29), (30, 55),
                                         (58, 83), (84, 109), (111, 136), (138, 163), (164, 189), (192, 217),
                                         (218, 243))],
    (40, 52): [_ru((a, b)) for a, b in ((-243, -192), (-189, -138), (-109, -58), (-55, -4), (4, 55), (58, 109),
                                         (138, 189), (192, 243))],
    (40, 106): [_ru((-243, -138)), _ru((-109, -4)), _ru((4, 109)), _ru((138, 243))],
    (40, 242): [_ru((-244, -3)), _ru((3, 244))],
    (40, 484): [_ru((-244, -3), (3, 244))],
    (80, 26): [_ru((a, b)) for a, b in ((-499, -474), (-473, -448), (-445, -420), (-419, -394), (-392, -367),
                                         (-365, -340), (-339, -314), (-311, -286), (-285, -260), (-257, -232),
                                         (-231, -206), (-203, -178), (-177, -152), (-150, -125), (-123, -98),
                                         (-97, -72), (-69, -44), (-43, -18))] + [_ru((-16, -4), (4, 16))] +
              [_ru((a, b)) for a, b in ((18, 43), (44, 69), (72, 97), (98, 123), (125, 150), (152, 177),
                                         (178, 203), (206, 231), (232, 257), (260, 285), (286, 311), (314, 339),
                                         (340, 365), (367, 392), (394, 419), (420, 445), (448, 473), (474, 499))],
    (80, 52): [_ru((a, b)) for a, b in ((-499, -448), (-445, -394), (-365, -314), (-311, -260), (-257, -206),
                                         (-203, -152), (-123, -72), (-69, -18), (18, 69), (72, 123), (152, 203),
                                         (206, 257), (260, 311), (314, 365), (394, 445), (448, 499))],
    (80, 106): [_ru((a, b)) for a, b in ((-499, -394), (-365, -260), (-257, -152), (-123, -18), (18, 123),
                                          (152, 257), (260, 365), (394, 499))],
    (80, 242): [_ru((-500, -259)), _ru((-258, -17)), _ru((17, 258)), _ru((259, 500))],
    (80, 484): [_ru((-500, -17)), _ru((17, 500))],
    (80, 996): [_ru((-500, -3), (3, 500))],
}
_PIL_SMALL = {20: [10, 22, 36, 48, 62, 76, 90, 102, 116],
              40: [10, 24, 36, 50, 64, 78, 90, 104, 116, 130, 144, 158, 170, 184, 198, 212, 224, 238],
              80: [10, 24, 38, 50, 64, 78, 92, 104, 118, 130, 144, 158, 172, 184, 198, 212, 226, 238, 252, 266,
                   280, 292, 306, 320, 334, 346, 360, 374, 388, 400, 414, 428, 442, 454, 468, 482, 496]}
_PIL_LARGE = {20: [22, 48, 90, 116], 40: [10, 36, 78, 104, 144, 170, 212, 238],
              80: [24, 50, 92, 118, 158, 184, 226, 252, 266, 292, 334, 360, 400, 426, 468, 494]}
_PIL_996 = [24, 92, 158, 226, 266, 334, 400, 468]
# size: (N_SD_short, BCC interleaver (N_COL, N_ROW/N_BPSCS), same with DCM, D_TM, D_TM with DCM, radiotap code)
HE_RU_SIZE = {26: (6, (8, 3), (4, 3), 1, 1, 4), 52: (12, (16, 3), (8, 3), 3, 1, 5),
              106: (24, (17, 6), (17, 3), 6, 3, 6), 242: (60, (26, 9), (13, 9), 9, 9, 7),
              484: (120, None, None, 12, 9, 8), 996: (240, None, None, 20, 14, 9)}
_HE_RU_CACHE = {}


def he_ru(bw, size, idx):
    """HeRu for resource unit idx (0-based, lowest frequency first) of a size in a bw-MHz PPDU, or None."""
    key = (bw, size, idx)
    if key in _HE_RU_CACHE:
        return _HE_RU_CACHE[key]
    spans = HE_RU_RANGES.get((bw, size))
    ru = None
    if spans is not None and 0 <= idx < len(spans):
        tones = set(k for a, b in spans[idx] for k in range(a, b + 1))
        src = _PIL_996 if size == 996 else (_PIL_SMALL[bw] if size <= 52 else _PIL_LARGE[bw])
        pil = sorted(k for p in src for k in (-p, p) if k in tones)
        sd_short, bcc, bcc_dcm, dtm, dtm_dcm, code = HE_RU_SIZE[size]
        ru = HeRu(bw, spans[idx], pil, sd_short, bcc, bcc_dcm, dtm, dtm_dcm, code)
        ru.size, ru.idx = size, idx
        # offset of the RU in 26-tone RU units from the low edge (radiotap RU allocation offset)
        ru.off26 = sum(1 for sp in HE_RU_RANGES[(bw, 26)] if max(b for _, b in sp) < min(tones))
    _HE_RU_CACHE[key] = ru
    return ru


def he_ru_from_trigger(bw, v):
    """RU of the 7-bit RU index of a Trigger frame User Info RU Allocation subfield (B7-B1)."""
    for base, size, cnt in ((0, 26, 37), (37, 52, 16), (53, 106, 8), (61, 242, 4), (65, 484, 2), (67, 996, 1)):
        if base <= v < base + cnt:
            return he_ru(bw, size, v - base)
    return None


_SIGB_PAIR_SLOT = (0, 2, 5, 7)
_SIGB_52_IDX = {0: 0, 2: 1, 5: 2, 7: 3}


def sigb_allocation(v):
    """HE-SIG-B RU Allocation subfield (802.11ax Table 27-26) -> list of (size, slot, users) in one
    242-tone region (slots are 26-tone positions 0-8), or ('none', size) for an RU with no User
    fields in this content channel, or None if reserved."""
    if v < 16:
        out = []
        for p in range(4):
            if (v >> (3 - p)) & 1:
                out.append((52, _SIGB_PAIR_SLOT[p], 1))
            else:
                out += [(26, _SIGB_PAIR_SLOT[p], 1), (26, _SIGB_PAIR_SLOT[p] + 1, 1)]
        out.append((26, 4, 1))
        return sorted(out, key=lambda e: e[1])
    if v < 96:
        # 5-bit pattern + 3-bit y2y1y0: (y + 1) User fields on the 106-tone RU
        y = (v & 7) + 1
        return {2: [(52, 0, 1), (52, 2, 1), (106, 5, y)],
                3: [(106, 0, y), (52, 5, 1), (52, 7, 1)],
                4: [(26, s, 1) for s in range(5)] + [(106, 5, y)],
                5: [(26, 0, 1), (26, 1, 1), (52, 2, 1), (26, 4, 1), (106, 5, y)],
                6: [(52, 0, 1), (26, 2, 1), (26, 3, 1), (26, 4, 1), (106, 5, y)],
                7: [(52, 0, 1), (52, 2, 1), (26, 4, 1), (106, 5, y)],
                8: [(106, 0, y)] + [(26, s, 1) for s in range(4, 9)],
                9: [(106, 0, y), (26, 4, 1), (26, 5, 1), (26, 6, 1), (52, 7, 1)],
                10: [(106, 0, y), (26, 4, 1), (52, 5, 1), (26, 7, 1), (26, 8, 1)],
                11: [(106, 0, y), (26, 4, 1), (52, 5, 1), (52, 7, 1)]}[v >> 3]
    if 96 <= v < 112:
        return [(106, 0, ((v >> 2) & 3) + 1), (106, 5, (v & 3) + 1)]
    if v == 112:
        return [(52, 0, 1), (52, 2, 1), (52, 5, 1), (52, 7, 1)]
    if v == 113:
        return []
    if v == 114:
        return [("none", 484)]
    if v == 115:
        return [("none", 996)]
    if 128 <= v < 192:
        return [(106, 0, ((v >> 3) & 7) + 1), (26, 4, 1), (106, 5, (v & 7) + 1)]
    if 192 <= v < 216:
        size = (242, 484, 996)[(v - 192) // 8]
        return [(size, 0, (v & 7) + 1)]
    return None


def sigb_ru(bw, region, size, slot):
    """Absolute RU (he_ru) of an entry of region `region` (20 MHz sub-channel, lowest first)."""
    if size == 26:
        idx = 9 * region + slot + (1 if bw == 80 and region >= 2 else 0)
    elif size == 52:
        idx = 4 * region + _SIGB_52_IDX[slot]
    elif size == 106:
        idx = 2 * region + (0 if slot == 0 else 1)
    elif size == 242:
        idx = region
    elif size == 484:
        idx = region // 2
    else:
        idx = 0
    return he_ru(bw, size, idx)


_P4 = np.array([[1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1], [-1, 1, 1, 1]], float)
HE_PMAT = {1: np.ones((1, 1)), 2: np.array([[1, -1], [1, 1]], float), 4: _P4,
           8: np.block([[_P4, _P4], [_P4, -_P4]])}   # P matrices of the HE-LTF (6 LTFs: not supported)
HE_NLTF = {0: 1, 1: 2, 2: 4, 3: 6, 4: 8}
HE_MU_GI_LTF = {0: (4, 16), 1: (2, 16), 2: (2, 32), 3: (4, 64)}   # HE MU: (LTF size, GI @ 20 MSps)
HE_TB_GI_LTF = {0: (1, 32), 1: (2, 32), 2: (4, 64)}               # Trigger frame GI And HE-LTF Type


def parse_trigger(data):
    """Fields of a Trigger frame (802.11ax 9.3.1.22) needed to decode the HE TB PPDU it solicits.
    data includes the FCS. Returns None if the frame is not a Trigger frame or is cut short."""
    if len(data) < 28 or (data[0] & 0x0C) != 0x04 or (data[0] >> 4) != 2:
        return None
    body = data[:-4]
    v = int.from_bytes(body[16:24], "little")
    t = dict(type=v & 0xF, ul_len=(v >> 4) & 0xFFF, ul_bw=(v >> 18) & 3, gi_ltf=(v >> 20) & 3,
             nltf_f=(v >> 23) & 7, stbc=(v >> 26) & 1, extra=(v >> 27) & 1, a=((v >> 34) & 3) or 4,
             pe=(v >> 36) & 1, doppler=(v >> 53) & 1, users=[])
    # length of the Trigger Dependent User Info per trigger type (None: depends on the field)
    dep = {0: 1, 1: 1, 2: None, 3: 0, 4: 0, 5: None, 6: 0}.get(t["type"])
    if t["type"] == 7:     # NFRP: no data in the response
        return t
    i = 24
    while i + 5 <= len(body):
        w = int.from_bytes(body[i:i + 5], "little")
        aid = w & 0xFFF
        if aid == 4095:
            break
        n = dep
        if n is None:
            # MU-BAR / GCR MU-BAR: BAR Control (2 octets) + BAR Information
            if i + 7 > len(body):
                break
            bc = int.from_bytes(body[i + 5:i + 7], "little")
            bar_type, tid_info = (bc >> 1) & 0xF, (bc >> 12) & 0xF
            n = 2 + {0: 2, 1: 2, 2: 2, 3: 4 * (tid_info + 1), 6: 8}.get(bar_type, -1000)
            if n < 0:
                break
        if aid != 2007:       # 2007: EHT Special User Info field
            ru = (w >> 12) & 0xFF
            u = dict(aid=aid, s80=ru & 1, ru_idx=ru >> 1, ldpc=(w >> 20) & 1, mcs=(w >> 21) & 0xF,
                     dcm=(w >> 25) & 1, ss=(w >> 26) & 7, nss=((w >> 29) & 7) + 1, n_ra=1)
            if aid in (0, 2045):  # random access RUs: B26-B30 = number of RA-RUs - 1
                u.update(ss=0, nss=1, n_ra=((w >> 26) & 0x1F) + 1)
            t["users"].append(u)
        i += 5 + n
    return t


def qam_points(nbpsc):
    """All constellation points of the 802.11 Gray mapping and their bit labels (b0 first)."""
    if nbpsc == 1:
        return np.array([-1.0 + 0j, 1.0 + 0j]), np.array([[0], [1]], np.uint8)
    h = nbpsc // 2
    m = 1 << h
    norm = math.sqrt(2 * (m * m - 1) / 3)
    gray_to_level = {i ^ (i >> 1): i for i in range(m)}
    pts, labels = [], []
    for v in range(1 << nbpsc):
        bits = [(v >> (nbpsc - 1 - j)) & 1 for j in range(nbpsc)]
        gi = int("".join(map(str, bits[:h])), 2)
        gq = int("".join(map(str, bits[h:])), 2)
        pts.append(complex(2 * gray_to_level[gi] - (m - 1), 2 * gray_to_level[gq] - (m - 1)) / norm)
        labels.append(bits)
    return np.array(pts), np.array(labels, np.uint8)


def demap_gen(z, nbpsc):
    """Max-log soft bits (positive -> 1) for 802.11 Gray-mapped BPSK..1024-QAM."""
    if nbpsc == 1:
        return z.real[:, None]
    h = nbpsc // 2
    s = math.sqrt(2 * ((1 << nbpsc) - 1) / 3)

    def axis(x):
        out, prev = [x], x
        for i in range(1, h):
            prev = (1 << (h - i)) - np.abs(prev)
            out.append(prev)
        return out

    return np.stack(axis(z.real * s) + axis(z.imag * s), 1)


def dcm_candidates(nbpsc):
    """Candidate maps from the lower-half DCM symbol to the upper-half one, as
    (conjugate, alternate sign by tone, negate, per-axis level permutation)."""
    import itertools
    perms = [None]
    if nbpsc == 4:
        perms = [p for p in itertools.permutations(range(4))]
    out = []
    for conj in (False, True):
        for alt in (False, True):
            for neg in (False, True):
                for perm in perms:
                    out.append((conj, alt, neg, perm))
    return out


def dcm_apply(cand, pts, kidx, nbpsc):
    """Upper-half images (len(kidx), M) of constellation points pts under candidate cand."""
    conj, alt, neg, perm = cand
    v = np.conj(pts) if conj else pts.copy()
    if perm is not None:
        norm = math.sqrt(10)
        lv = np.array([-3, -1, 1, 3])

        def remap(x):
            idx = np.round((x * norm + 3) / 2).astype(int)
            return lv[np.array(perm)[idx]] / norm

        v = remap(v.real) + 1j * remap(v.imag)
    if neg:
        v = -v
    v = np.broadcast_to(v, (len(kidx), len(pts))).copy()
    if alt:
        v = v * np.where(kidx % 2 == 0, 1.0, -1.0)[:, None]
    return v


# ---------------------------------------------------------------------------
# OFDM receiver (detection and legacy part at 20 MSps; HT/VHT at 20/40/80 MSps)
# ---------------------------------------------------------------------------


class OfdmRx:
    BK = 3  # FFT window starts this many (20 MSps) samples inside the cyclic prefix

    def __init__(self, keep_bad, stats):
        self.keep_bad = keep_bad
        self.stats = stats
        self.he_log = None      # csv writer for --he-log
        self.triggers = []      # recent Trigger frames: dicts from parse_trigger plus "t_end" (s)
        self.he_where = None    # (channel, seconds of sample 0 of this block's 20 MSps stream)
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
        t0 = self._tsec(0)
        self.triggers = [t for t in self.triggers if t["t_end"] > t0 - 0.02]
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
            self._note_triggers(got, end)
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
            # RL-SIG test, weighted by |H|^2 on both symbols so tones in a channel null
            # (common with a second transmit antenna's cyclic shift) do not dominate
            corr = abs(np.vdot(z0 * wts, z2 * wts)) / (np.linalg.norm(z0 * wts) * np.linalg.norm(z2 * wts) + 1e-30)
            if corr > 0.8:  # RL-SIG repeats L-SIG: HE PPDU
                return self._he(pk, y, H, nv, t_sig)

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

    # -- HE (802.11ax) SU / ER SU -----------------------------------------

    def _sig52(self, y, start, H, w64, pol):
        """One pre-HE BPSK/QBPSK symbol on the 52 HE-SIG-A tones. Tones +-27/+-28 have
        no L-LTF estimate: their weight is 0 (erasures)."""
        bk = self.BK
        Y = np.fft.fft(y[start + 16 - bk:start + 80 - bk])
        zp = Y[PILOT_IDX] / H[PILOT_IDX]
        cpl = zp * np.conj(PILOT_PSI * PILOT_POL[pol])
        ph0 = np.angle(cpl.sum())
        res = np.angle(cpl * np.exp(-1j * ph0))
        wp = np.abs(H[PILOT_IDX]) ** 2
        slope = float(np.sum(wp * res * PILOT_K) / np.sum(wp * PILOT_K ** 2))
        idx = HE_SIGA_K % 64
        z = Y[idx] / H[idx] * np.exp(-1j * (ph0 + slope * HE_SIGA_K))
        return z, w64[idx]

    def _he(self, pk, y, H, nv, t_sig):
        length = pk["l_len"]
        mod3 = length % 3
        if mod3 == 0:
            self._count("he_bad_l_length")
            return [], pk["l_end"]
        w64 = np.abs(H) ** 2 / nv
        k64 = np.arange(64)
        k64 = np.where(k64 < 32, k64, k64 - 64)
        w64[np.abs(k64) > 26] = 0.0   # no L-LTF estimate on +-27/+-28: erasures (H there is a placeholder 1)
        nsyms = 4 if mod3 == 2 else 2
        if len(y) < t_sig + 80 * (2 + nsyms) + 16:
            return [], pk["l_end"]
        zs = [self._sig52(y, t_sig + 80 * (2 + i), H, w64, 2 + i) for i in range(nsyms)]
        er = False
        if mod3 == 2:
            z1, w1 = zs[1]
            if np.sum(w1 * z1.imag ** 2) > 2 * np.sum(w1 * z1.real ** 2):
                er = True   # HE ER SU: second HE-SIG-A symbol is a QBPSK repetition
        if er:
            (z0, w), (z1, _), (z2, _), (z3, _) = zs
            llr = np.concatenate([(z0.real * w)[PERM52] + z1.imag * w, (z2.real * w)[PERM52] + z3.real * w])
        else:
            (z0, w), (z1, _) = zs[:2]
            llr = np.concatenate([(z0.real * w)[PERM52], (z1.real * w)[PERM52]])
        # Terminated decoding: tones +-27/+-28 are erasures, and when the tail's own coded bits
        # land on them the last bits would otherwise be left undetermined.
        b = VITERBI.decode(llr, terminated=True)
        # CRC: the first four bits (c7..c4, c7 first) of the HT-SIG CRC-8 over bits 0-41
        if not np.array_equal(b[42:46], msb_bits(crc8(b[:42]), 8)[:4]):
            self._count("he_siga_crc_fail")
            return [], pk["l_end"]
        if self.he_log is not None:
            self._log_siga(pk, b, mod3, er)
        if mod3 == 2 and not er:
            return self._he_mu(pk, b)
        if not er and not b[0]:
            return self._he_tb(pk, b)
        f = dict(beam_change=int(b[1]), ul_dl=int(b[2]), mcs=bits_to_int(b[3:7]), dcm=bool(b[7]),
                 bss_color=bits_to_int(b[8:14]), spatial_reuse=bits_to_int(b[15:19]),
                 bwf=bits_to_int(b[19:21]), gi_ltf=bits_to_int(b[21:23]), nsts_f=bits_to_int(b[23:26]),
                 txop=bits_to_int(b[26:33]), ldpc=bool(b[33]), extra=bool(b[34]), stbc=bool(b[35]),
                 beamformed=bool(b[36]), a=bits_to_int(b[37:39]) or 4, pe_disamb=bool(b[39]),
                 doppler=bool(b[41]), er=er)
        # GI+LTF = 3 with both DCM and STBC set signals 4x HE-LTF + 0.8 us GI; in that case
        # neither DCM nor STBC is applied to the Data field (802.11ax Table 27-18).
        short4x = f["gi_ltf"] == 3 and f["dcm"] and f["stbc"]
        if short4x:
            f["dcm"] = f["stbc"] = False
        if f["doppler"]:
            self._count("he_doppler_unsupported")
            return [], pk["l_end"]
        nsts = f["nsts_f"] + 1
        nss = nsts // 2 if f["stbc"] else nsts
        if nss != 1 or nsts > 2:
            self._count("he_unsupported_mimo")
            return [], pk["l_end"]
        if er:
            ru = {0: "242", 1: "106u"}.get(f["bwf"])
        else:
            ru = {0: "242", 1: "484", 2: "996"}.get(f["bwf"])
        mcs = f["mcs"]
        if ru is None:
            self._count("he160_unsupported")
            return [], pk["l_end"]
        reason = None
        if mcs > 11:
            reason = "mcs_reserved"
        elif er and mcs > 2:
            reason = "er_mcs"
        elif f["dcm"] and mcs not in (0, 1, 3, 4):
            reason = "dcm_mcs"
        elif not f["ldpc"] and mcs > 9:
            reason = "bcc_mcs"
        elif not f["ldpc"] and HE_RUS[ru].bcc is None:
            reason = "bcc_wide"
        if reason:
            self._count("he_invalid_siga_" + reason)
            return [], pk["l_end"]
        mult, gi = HE_GI_LTF[f["gi_ltf"]]
        if short4x:
            gi = 16   # 4x HE-LTF with 0.8 us GI
        nltf = 2 if nsts == 2 else 1
        nsiga = 4 if er else 2
        ltf0 = 160 + 80 * nsiga + 80
        data0 = ltf0 + nltf * (gi + 64 * mult)
        tsym = 256 + gi
        m = 1 if mod3 == 2 else 2
        txtime_us = (length + 3 + m) // 3 * 4 + 20
        # TXTIME - 20 us is the time after the L-SIG; data0 counts from the start of the L-SIG
        nsym = ((txtime_us - 20) * 20 - (data0 - 80)) // tsym - int(f["pe_disamb"])
        if nsym <= 0:
            self._count("he_no_data_symbols")   # e.g. sounding NDPs
            return [], pk["l_end"]
        nbpsc, cr = HE_MCS[mcs]
        p = dict(f, ru=ru, nsts=nsts, nbpsc=nbpsc, cr=cr, mult=mult, gi=gi, nltf=nltf, ltf0=ltf0,
                 data0=data0, nsym=nsym, tsym=tsym)
        return self._he_data(pk, p)

    # -- HE MU (downlink OFDMA) and HE TB (uplink OFDMA) --------------------

    def _tsec(self, s20):
        """Capture time in seconds of 20 MSps index s20 of the current channel stream."""
        ctx = self.ctx
        if ctx is not None and hasattr(ctx, "_raw"):
            return ctx._raw(s20) / ctx.fs_nom
        return s20 / 20e6

    def _note_triggers(self, frames, end):
        for fr in frames:
            if not fr.fcs_ok:
                continue
            t = parse_trigger(fr.data)
            if t is None:
                continue
            t["t_end"] = self._tsec(end)
            t["raw"] = fr.data
            if any(o["raw"] == fr.data and abs(o["t_end"] - t["t_end"]) < 5e-6 for o in self.triggers):
                continue
            self._count("trigger_frames")
            self.triggers.append(t)

    def _he_slice(self, pk, bw, n_after):
        """CFO-corrected samples of the PPDU at bw MHz from 160 (20 MSps) samples before the L-SIG
        to n_after samples after its start, with the L-LTF channel estimate on 64*u bins."""
        u = bw // 20
        bk = self.BK * u
        out = []
        for st, s_w, w_w, fc in self._stream(pk, bw):
            s0 = s_w - 160 * u
            need = (160 + n_after) * u
            if s0 < 0 or s0 + need > len(st):
                continue
            y = np.asarray(st[s0:s0 + need], complex) * np.exp(-1j * w_w * np.arange(need))
            o = 160 * u
            n = 64 * u
            ref = np.zeros(n)
            for j in range(u):
                ref[(np.arange(-32, 32) + 64 * j - 32 * (u - 1)) % n] = LTF64[np.arange(-32, 32) % 64]
            used = ref != 0
            Y1 = np.fft.fft(y[o - 128 * u - bk:o - 64 * u - bk])
            Y2 = np.fft.fft(y[o - 64 * u - bk:o - bk])
            H = np.ones(n, complex)
            H[used] = (Y1[used] + Y2[used]) / 2 / ref[used]
            nv = max(float(np.mean(np.abs(Y1[used] - Y2[used]) ** 2) / 2), 1e-30)
            out.append(dict(y=y, o=o, u=u, bk=bk, H=H, nv=nv, fc=fc, bw=bw))
        return out

    def _presym(self, sl, n, pol):
        """Pre-HE symbol n (0 = L-SIG) on the 52 HE-SIG tones of every 20 MHz sub-channel:
        (z (u, 52), weights (u, 52), pilot amplitude (u,)), or None past the slice."""
        y, o, u, bk, H, nv = sl["y"], sl["o"], sl["u"], sl["bk"], sl["H"], sl["nv"]
        st = o + 80 * n * u + 16 * u - bk
        if st + 64 * u > len(y):
            return None
        Y = np.fft.fft(y[st:st + 64 * u])
        zs, ws, amps = [], [], []
        pv = PILOT_PSI * PILOT_POL[pol % 127]
        for j in range(u):
            off = 64 * j - 32 * (u - 1)
            pb = (PILOT_K + off) % (64 * u)
            db = (HE_SIGA_K + off) % (64 * u)
            hp = H[pb]
            wp = np.abs(hp) ** 2
            cpl = Y[pb] / hp * pv
            ph0 = float(np.angle(np.sum(cpl * wp)))
            res = np.angle(cpl * np.exp(-1j * ph0))
            slope = float(np.sum(wp * res * PILOT_K) / max(float(np.sum(wp * PILOT_K ** 2)), 1e-30))
            zs.append(Y[db] / H[db] * np.exp(-1j * (ph0 + slope * HE_SIGA_K)))
            w = np.abs(H[db]) ** 2 / nv
            w[np.abs(HE_SIGA_K) > 26] = 0.0
            ws.append(w)
            amps.append(abs(np.sum(cpl * np.exp(-1j * ph0) * wp)) / max(float(np.sum(wp)), 1e-30))
        return np.array(zs), np.array(ws), np.array(amps)

    def _sigb(self, sl, f, bw):
        """Decode HE-SIG-B. Returns (list of user dicts, None) or (None, reason)."""
        u = bw // 20
        mcs = f["sigb_mcs"]
        if mcs > 5:
            return None, "he_sigb_bad_mcs"
        if f["sigb_dcm"]:
            return None, "he_sigb_dcm_unsupported"
        nb, cr = HE_MCS[mcs]
        ndbps = int(52 * nb * cr)
        perm = interleave_perm(52 * nb, nb, True)
        cache = {}

        def sym(n):
            if n not in cache:
                cache[n] = self._presym(sl, 4 + n, 4 + n)
            return cache[n]

        amp = [None]

        def decode(c, nsym, flip):
            llrs = []
            for n in range(nsym):
                r = sym(n)
                if r is None:
                    return None
                z, w, a = r
                if amp[0] is None:
                    amp[0] = max(float(np.median(a)), 1e-9)
                acc = 0
                for j in (range(c, u, 2) if u > 1 else [0]):
                    zz = z[j] / amp[0]
                    if flip:
                        zz = zz.copy()
                        zz[27::2] *= -1
                    acc = acc + (demap_gen(zz, nb) * w[j][:, None]).reshape(-1)[perm]
                llrs.append(acc)
            full = depuncture(np.concatenate(llrs), cr, nsym * ndbps)
            if full is None:
                return None
            return VITERBI.decode(full)

        def crc_ok(bits, n):
            return np.array_equal(bits[n:n + 4], msb_bits(crc8(bits[:n]), 8)[:4])

        ncc = 1 if u == 1 else 2
        comp = f["sigb_comp"]
        flip = False
        if comp:
            nuser = f["sigb_n"] + 1
            counts = [nuser] if ncc == 1 else [(nuser + 1) // 2, nuser // 2]
            ncommon = 0
            ru_full = he_ru(bw, {20: 242, 40: 484, 80: 996}[bw], 0)
            cc_rus = [[(ru_full, n)] for n in counts]
        else:
            nalloc = 1 if bw < 80 else 2
            ncf = 8 * nalloc + (1 if bw == 80 else 0)
            ncommon = ncf + 10
            nsym0 = -(-ncommon // ndbps)
            commons = []
            for c in range(ncc):
                bits = None
                for fl in ((flip,) if c else (False, True)):
                    bb = decode(c, nsym0, fl)
                    if bb is not None and crc_ok(bb, ncf):
                        bits, flip = bb, fl
                        break
                if bits is None:
                    return None, "he_sigb_common_crc_fail"
                commons.append(bits)
            cc_rus = []
            for c in range(ncc):
                bits = commons[c]
                regions = [0] if bw == 20 else ([c] if bw == 40 else [c, c + 2])
                lst = []
                for i, r in enumerate(regions):
                    alloc = sigb_allocation(bits_to_int(bits[8 * i:8 * i + 8]))
                    if alloc is None:
                        return None, "he_sigb_reserved_alloc"
                    for e in alloc:
                        if e[0] == "none":
                            continue
                        size, slot, n = e
                        ru = sigb_ru(bw, r, size, slot)
                        if ru is None:
                            return None, "he_sigb_reserved_alloc"
                        lst.append((ru, n))
                if bw == 80 and c == 0 and commons[0][16]:
                    lst.append((he_ru(80, 26, 18), 1))   # centre 26-tone RU: assumed last in content channel 1
                cc_rus.append(lst)
            counts = [sum(n for _, n in lst) for lst in cc_rus]
        need = max(ncommon + 52 * (n // 2) + 31 * (n % 2) for n in counts)
        nsym_req = max(1, -(-need // ndbps))
        nsym = nsym_req if (comp or f["sigb_n"] == 15) else f["sigb_n"] + 1
        if nsym < nsym_req:
            return None, "he_sigb_too_short"
        f["n_sigb"] = nsym
        per_ru = {}
        for lst in cc_rus:
            for ru, n in lst:
                per_ru[id(ru)] = per_ru.get(id(ru), 0) + n
        def parse(fl):
            users, nfail = [], 0
            for c in range(ncc):
                bits = decode(c, nsym, fl)
                if bits is None:
                    return None, 10 ** 6
                pos = ncommon
                slots = [ru for ru, n in cc_rus[c] for _ in range(n)]
                k = 0
                while k < len(slots):
                    pair = min(2, len(slots) - k)
                    blk = bits[pos:pos + 21 * pair + 4]
                    ok = len(blk) == 21 * pair + 4 and crc_ok(blk, 21 * pair)
                    if not ok:
                        nfail += pair
                    for q in range(pair if ok else 0):
                        fb = blk[21 * q:21 * q + 21]
                        ru = slots[k + q]
                        mumimo = per_ru[id(ru)] > 1
                        usr = dict(ru=ru, sta_id=bits_to_int(fb[0:11]), mcs=bits_to_int(fb[15:19]),
                                   ldpc=bool(fb[20]), mumimo=mumimo)
                        if not mumimo:
                            usr.update(nsts=bits_to_int(fb[11:14]) + 1, beamformed=bool(fb[14]),
                                       dcm=bool(fb[19]))
                        users.append(usr)
                    pos += 21 * pair + 10
                    k += pair
            return users, nfail

        users, nfail = parse(flip)
        if nfail:
            # the alternating sign on the upper half of HE-SIG-B (seen in one reference
            # implementation) cannot always be told from the Common field alone
            u2, f2 = parse(not flip)
            if f2 < nfail:
                users, nfail, flip = u2, f2, not flip
                self._count("he_sigb_upper_alternating")
        if nfail:
            self._count("he_sigb_user_crc_fail", nfail)
        return users, None

    def _he_mu(self, pk, b):
        u2i = lambda lo, hi: bits_to_int(b[lo:hi])
        f = dict(ul_dl=int(b[0]), sigb_mcs=u2i(1, 4), sigb_dcm=bool(b[4]), bss_color=u2i(5, 11),
                 spatial_reuse=u2i(11, 15), bwf=u2i(15, 18), sigb_n=u2i(18, 22), sigb_comp=bool(b[22]),
                 gi_ltf=u2i(23, 25), doppler=bool(b[25]), txop=u2i(26, 33), nltf_f=u2i(34, 37),
                 extra=bool(b[37]), stbc=bool(b[38]), a=u2i(39, 41) or 4, pe_disamb=bool(b[41]))
        self._count("he_mu_ppdu")
        if f["doppler"]:
            self._count("he_doppler_unsupported")
            return [], pk["l_end"]
        if f["bwf"] > 2:
            self._count("he_mu_bw_unsupported")   # 160 MHz or preamble puncturing
            return [], pk["l_end"]
        if f["stbc"]:
            self._count("he_mu_stbc_unsupported")
            return [], pk["l_end"]
        bw = (20, 40, 80)[f["bwf"]]
        nltf = HE_NLTF.get(f["nltf_f"])
        if nltf not in HE_PMAT:
            self._count("he_mu_nltf_unsupported")
            return [], pk["l_end"]
        mult, gi = HE_MU_GI_LTF[f["gi_ltf"]]
        length = pk["l_len"]
        txtime_us = (length + 3 + 1) // 3 * 4 + 20
        n_after = (txtime_us - 20) * 20 + 80
        slices = self._he_slice(pk, bw, n_after + 16)
        if not slices:
            self._count("wide_out_of_span" if bw > 20 else "he_short_capture")
            return [], pk["l_end"]
        users, why = None, "he_short_capture"
        for sl in slices:
            users, why = self._sigb(sl, f, bw)
            if users is not None:
                break
        if users is None:
            self._count(why)
            return [], pk["l_end"]
        self._count("he_sigb_ok")
        ltf0 = 160 + 160 + 80 * f["n_sigb"] + 80
        data0 = ltf0 + nltf * (gi + 64 * mult)
        tsym = 256 + gi
        nsym = ((txtime_us - 20) * 20 - (data0 - 80)) // tsym - int(f["pe_disamb"])
        end20 = pk["t_sig"] + n_after
        if self.ctx is not None and bw > 20:
            self.ctx.claim(pk["s"], pk["s"] + end20 + 80, [(None, None, None, sl["fc"])], bw)
        if nsym <= 0:
            self._count("he_no_data_symbols")
            return [], pk["s"] + end20
        any_ldpc = any(us.get("ldpc") for us in users)
        com = dict(nsym=nsym, a=f["a"], extra=f["extra"] and any_ldpc, mult=mult, gi=gi, nltf=nltf,
                   ltf0=ltf0, data0=data0, tsym=tsym, bss_color=f["bss_color"], ul_dl=f["ul_dl"],
                   spatial_reuse=f["spatial_reuse"], txop=f["txop"], pe_disamb=f["pe_disamb"])
        todo = []
        for us in users:
            if us["sta_id"] == 2046:
                continue                      # unassigned RU
            if us["mumimo"]:
                self._count("he_mu_mimo_user_unsupported")
                continue
            if us["nsts"] != 1:
                self._count("he_unsupported_mimo")
                continue
            todo.append(dict(us, ss=0))
        frames = self._he_ofdma_data(pk, sl, bw, com, todo, "mu", common_track=True)
        return frames, pk["s"] + end20

    def _he_tb(self, pk, b):
        self._count("he_tb_ppdu")
        t_start = self._tsec(pk["s"] + pk["t_sig"] - 320)
        best = None
        for t in self.triggers:
            dt = t_start - t["t_end"]
            if t["ul_len"] == pk["l_len"] and 2e-6 < dt < 60e-6 and t["type"] not in (3, 7):
                if best is None or abs(dt - 16e-6) < abs(best[0] - 16e-6):
                    best = (dt, t)
        if best is None:
            self._count("he_tb_no_trigger")
            return [], pk["l_end"]
        t = best[1]
        if t["doppler"]:
            self._count("he_doppler_unsupported")
            return [], pk["l_end"]
        if t["stbc"]:
            self._count("he_tb_stbc_unsupported")
            return [], pk["l_end"]
        if t["ul_bw"] > 2 or t["gi_ltf"] not in HE_TB_GI_LTF:
            self._count("he_tb_bw_unsupported" if t["ul_bw"] > 2 else "he_tb_bad_trigger")
            return [], pk["l_end"]
        bw = (20, 40, 80)[t["ul_bw"]]
        nltf = HE_NLTF.get(t["nltf_f"])
        if nltf not in HE_PMAT:
            self._count("he_tb_nltf_unsupported")
            return [], pk["l_end"]
        mult, gi = HE_TB_GI_LTF[t["gi_ltf"]]
        length = pk["l_len"]
        txtime_us = (length + 3 + 2) // 3 * 4 + 20
        n_after = (txtime_us - 20) * 20 + 80
        ltf0 = 160 + 160 + 160              # L-SIG + RL-SIG, HE-SIG-A, 8 us HE-STF
        data0 = ltf0 + nltf * (gi + 64 * mult)
        tsym = 256 + gi
        nsym = ((txtime_us - 20) * 20 - (data0 - 80)) // tsym - int(t["pe"])
        slices = self._he_slice(pk, bw, n_after + 16)
        if not slices:
            self._count("wide_out_of_span" if bw > 20 else "he_short_capture")
            return [], pk["l_end"]
        sl = slices[0]
        end20 = pk["t_sig"] + n_after
        if self.ctx is not None and bw > 20:
            self.ctx.claim(pk["s"], pk["s"] + end20 + 80, [(None, None, None, sl["fc"])], bw)
        if nsym <= 0:
            self._count("he_no_data_symbols")
            return [], pk["s"] + end20
        todo, seen = [], {}
        for us in t["users"]:
            if us["s80"]:
                continue
            for r in range(us["n_ra"]):
                ru = he_ru_from_trigger(bw, us["ru_idx"] + r)
                if ru is None:
                    continue
                seen[id(ru)] = seen.get(id(ru), 0) + 1
                todo.append(dict(ru=ru, sta_id=us["aid"], mcs=us["mcs"], ldpc=bool(us["ldpc"]),
                                 dcm=bool(us["dcm"]), nsts=us["nss"], ss=us["ss"], beamformed=False))
        keep = []
        for us in todo:
            if seen[id(us["ru"])] > 1:
                self._count("he_mu_mimo_user_unsupported")
            elif us["nsts"] != 1:
                self._count("he_unsupported_mimo")
            else:
                keep.append(us)
        any_ldpc = any(us["ldpc"] for us in keep)
        com = dict(nsym=nsym, a=t["a"], extra=bool(t["extra"]), mult=mult, gi=gi, nltf=nltf, ltf0=ltf0,
                   data0=data0, tsym=tsym, bss_color=bits_to_int(b[1:7]), ul_dl=1, spatial_reuse=0,
                   txop=bits_to_int(b[26:33]), pe_disamb=bool(t["pe"]))
        del any_ldpc
        frames = self._he_ofdma_data(pk, sl, bw, com, keep, "tb", common_track=False)
        return frames, pk["s"] + end20

    def _he_chan1(self, sl, ru, com, row, nb, beamformed):
        """Channel of one space-time stream (P-matrix row `row`) on every tone of RU ru."""
        y, o, u, bk = sl["y"], sl["o"], sl["u"], sl["bk"]
        N, mult, nltf = ru.N, com["mult"], com["nltf"]
        nl = N * mult // 4
        gi = com["gi"] * u
        step = 4 // mult
        L = HE_LTF[(ru.bw, mult)]
        scale = math.sqrt(4 / mult)
        kp = ru.tones[(ru.tones % step == 0) & (L[ru.tone_i] != 0)]
        bins = (kp // step) % nl
        pil = np.isin(kp, ru.pil_k)
        P = HE_PMAT[nltf]
        acc = np.zeros(len(kp), complex)
        for i in range(nltf):
            st = o + com["ltf0"] * u + i * (gi + nl) + gi - bk
            if st + nl > len(y):
                return None
            Yi = np.fft.fft(y[st:st + nl])[bins] * scale / L[kp % N]
            # pilot tones of the HE-LTF follow the first row of P for every stream
            acc += Yi * np.where(pil, P[0, i], P[row, i])
        e = acc / nltf
        kfull = np.arange(N)
        kfull = np.where(kfull < N // 2, kfull, kfull - N)
        ramp = np.exp(-2j * np.pi * kfull * bk / N)
        span_hz = 0.0 if beamformed else (470e3 if nb <= 6 else (160e3 if nb == 8 else 0.0))
        span = int(round(span_hz / (78.125e3 * step)))
        return self._he_fill(e / ramp[kp % N], kp, ru, span) * ramp

    def _he_ofdma_data(self, pk, sl, bw, com, users, fmt, common_track):
        """Decode the single-stream users of an HE MU or HE TB PPDU (one RU each)."""
        from types import SimpleNamespace
        if not users:
            return []
        y, o, u, bk = sl["y"], sl["o"], sl["u"], sl["bk"]
        N = 256 * u
        nsym, tsym, gi = com["nsym"], com["tsym"], com["gi"]
        starts = o + (com["data0"] + np.arange(nsym) * tsym) * u
        if starts[-1] + gi * u - bk + N > len(y):
            self._count("he_short_capture")
            return []
        idx = starts[:, None] + (gi * u - bk) + np.arange(N)[None, :]
        Y = np.fft.fft(y[idx], axis=1)
        kfull = np.arange(N)
        kfull = np.where(kfull < N // 2, kfull, kfull - N)
        jobs = []
        for us in users:
            ru, mcs = us["ru"], us["mcs"]
            bad = (mcs > 11 or (us["dcm"] and mcs not in (0, 1, 3, 4)) or
                   (not us["ldpc"] and (mcs > 9 or ru.bcc is None)))
            if bad:
                self._count("he_%s_user_invalid" % fmt)
                continue
            nb, cr = HE_MCS[mcs]
            p = dict(nbpsc=nb, cr=cr, dcm=us["dcm"], stbc=False, ldpc=us["ldpc"], mcs=mcs, nsym=nsym,
                     a=com["a"], extra=com["extra"], bcc_init=True, beamformed=us["beamformed"])
            lay = self._he_layout(ru, p)
            if lay is None or lay["n_pld"] <= 16:
                self._count("he_%s_user_invalid" % fmt)
                continue
            H = self._he_chan1(sl, ru, com, us["ss"], nb, us["beamformed"])
            if H is None:
                continue
            jobs.append((us, ru, p, lay, H))
        if not jobs:
            return []
        if common_track:
            # one transmitter: pilots of every RU share the common phase and timing drift
            pi = np.concatenate([ru.pil_i for _, ru, _, _, _ in jobs])
            pk_ = np.concatenate([ru.pil_k for _, ru, _, _, _ in jobs])
            heff = np.zeros(N, complex)
            for _, ru, _, _, H in jobs:
                heff[ru.pil_i] = H[ru.pil_i]
            ph, sl_ = self._he_track(Y, SimpleNamespace(pil_i=pi, pil_k=pk_), heff)
            Yc = Y * np.exp(-1j * (ph[:, None] + sl_[:, None] * kfull[None, :]))
        out = []
        for us, ru, p, lay, H in jobs:
            if not common_track:
                ph, sl_ = self._he_track(Y, ru, H)   # each HE TB transmitter drifts on its own
                Yc = Y * np.exp(-1j * (ph[:, None] + sl_[:, None] * kfull[None, :]))
            mpdus = self._he_payload(Yc, [H], ru, p, lay)
            if mpdus is None:
                continue
            good = sum(fcs_ok(m) for m in mpdus)
            self._count("he_%s_user_ok" % fmt if good else "he_%s_user_fail" % fmt)
            if not mpdus:
                self._count("ampdu_no_delimiters")
            kw = dict(mcs=us["mcs"], sgi=False, bw=bw, stbc=False, ldpc=us["ldpc"], center_mhz=sl["fc"],
                      beamformed=us["beamformed"], ldpc_extra=com["extra"], dcm=us["dcm"], he_er=False,
                      he_ru="%d:%d" % (ru.size, ru.idx), he_fmt=fmt, he_ru_off26=ru.off26, he_ru_code=ru.ru_code,
                      he_sta_id=us["sta_id"], bss_color=com["bss_color"], ul_dl=com["ul_dl"], beam_change=0,
                      spatial_reuse=com["spatial_reuse"], txop=com["txop"], gi_samples=com["gi"],
                      ltf_mult=com["mult"], n_ltf=com["nltf"], pre_fec_a=com["a"], pe_disamb=com["pe_disamb"],
                      nsts=1)
            out += self._emit(mpdus, pk["s"], "he", pk["meta"], **kw)
        return out

    def _log_siga(self, pk, b, mod3, er):
        ch, t0 = self.he_where or (0, 0.0)
        u = lambda lo, hi: bits_to_int(b[lo:hi])
        self.he_log.writerow([
            "%.6f" % (t0 + pk["s"] / 20e6), ch, pk["l_len"], mod3, int(er),
            "".join(map(str, b[:26])), "".join(map(str, b[26:52])),
            int(b[0]), u(3, 7), int(b[7]), u(19, 21), u(21, 23), u(23, 26),
            int(b[33]), int(b[35]), int(b[41])])

    def _he_layout(self, ru, p):
        """Coded and information bit counts of 802.11ax 27.3.12 (pre-FEC padding factor a)."""
        dcm = p["dcm"]
        nsd = ru.nsd // 2 if dcm else ru.nsd
        sd_short = ru.sd_short // 2 if dcm else ru.sd_short
        nb, cr = p["nbpsc"], p["cr"]
        ncbps, ncbps_s = nsd * nb, sd_short * nb
        ndbps, ndbps_s = int(ncbps * cr), int(ncbps_s * cr)
        m = 2 if p["stbc"] else 1

        def last(a, full, short):
            return a * short if a < 4 else full

        nsym, a = p["nsym"], p["a"]
        av_final = (nsym - m) * ncbps + m * last(a, ncbps, ncbps_s)
        nsym_i, a_i = nsym, a
        if p["extra"]:
            if a == 1:
                a_i, nsym_i = 4, nsym - m
            else:
                a_i = a - 1
        if not p["ldpc"]:
            if p.get("bcc_init") and p["extra"]:
                # MU / TB PPDU whose LDPC users needed the extra symbol segment: a BCC user's
                # coded bits end at the initial symbol count; the rest is post-FEC padding
                nsym, a = nsym_i, a_i
            out = dict(ncbps=ncbps, ncbps_last=last(a, ncbps, ncbps_s), m=m, nsd=nsd,
                       ndbps=ndbps, ndbps_last=last(a, ndbps, ndbps_s), nsym_used=nsym,
                       n_pld=(nsym - m) * ndbps + m * last(a, ndbps, ndbps_s))
            return out
        out = dict(ncbps=ncbps, ncbps_last=last(a, ncbps, ncbps_s), m=m, nsd=nsd, av_final=av_final,
                   ndbps=ndbps, ndbps_last=last(a, ndbps, ndbps_s))
        n_pld = (nsym_i - m) * ndbps + m * last(a_i, ndbps, ndbps_s)
        av_init = (nsym_i - m) * ncbps + m * last(a_i, ncbps, ncbps_s)
        if n_pld <= 0 or av_init <= 0:
            return None
        out["n_pld"] = n_pld
        out["lay"] = ldpc_layout_explicit(n_pld, av_init, av_final, cr)
        return out

    def _he_chan(self, y, ru, p, bk):
        """Channel per space-time stream from the HE-LTF(s), on every tone of the RU,
        plus the effective channel seen by the pilots (same pilots on every stream)."""
        N, mult = ru.N, p["mult"]
        u = ru.u
        nl = N * mult // 4
        gi = p["gi"] * u
        step = 4 // mult
        L = HE_LTF[(ru.bw, mult)]
        scale = math.sqrt(4 / mult)           # 1x/2x HE-LTF put more power on fewer tones
        kp = ru.tones[(ru.tones % step == 0) & (L[ru.tone_i] != 0)]
        bins = (kp // step) % nl
        Ys = []
        for i in range(p["nltf"]):
            st = (p["ltf0"] * u) + i * (gi + nl) + gi - bk
            Ys.append(np.fft.fft(y[st:st + nl])[bins] * scale / L[kp % N])
        kfull = np.arange(N)
        kfull = np.where(kfull < N // 2, kfull, kfull - N)
        ramp = np.exp(-2j * np.pi * kfull * bk / N)
        if p["nsts"] == 1:
            ests = [Ys[0]]
        else:
            ests = [(Ys[0] - Ys[1]) / 2, (Ys[0] + Ys[1]) / 2]
        pil_set = set(ru.pil_k.tolist())
        is_pil = np.array([k in pil_set for k in kp])
        nb = p["nbpsc"]
        span_hz = 0.0 if p["beamformed"] else (470e3 if nb <= 6 else (160e3 if nb == 8 else 0.0))
        span = int(round(span_hz / (78.125e3 * step)))
        out = []
        for e in ests:
            # pilot tones of a 2-stream HE-LTF follow R, not P: leave them out of the per-stream fit
            use = ~is_pil if p["nsts"] == 2 else np.ones(len(kp), bool)
            out.append(self._he_fill(e[use] / ramp[kp[use] % N], kp[use], ru, span) * ramp)
        # pilots: first HE-LTF symbol alone gives the effective channel under either mapping
        heff = self._he_fill(Ys[0] / ramp[kp % N], kp, ru, span) * ramp if p["nsts"] == 2 else out[0]
        return out, heff

    @staticmethod
    def _he_fill(g, kp, ru, span):
        """Smooth estimates g on tones kp (triangular, +-span estimate points) and
        interpolate them linearly onto every tone of each contiguous run."""
        H = np.zeros(ru.N, complex)
        # Remove the mean phase slope across tones (multipath delay, cyclic shift of a second
        # stream) before smoothing and interpolation, so sparse 1x/2x estimates interpolate well.
        tilt = 0.0
        if len(kp) > 1:
            dk = np.diff(kp)
            near = dk <= 4
            if near.any():
                tilt = float(np.angle(np.sum((g[1:] * np.conj(g[:-1]))[near]))) / float(np.mean(dk[near]))
        g = g * np.exp(-1j * tilt * kp)
        for run in ru.runs:
            sel = (kp >= run[0]) & (kp <= run[-1])
            ks, gs = kp[sel], g[sel]
            if len(ks) == 0:
                continue
            if span > 0 and len(ks) > 1:
                acc = np.zeros(len(gs), complex)
                wsum = np.zeros(len(gs))
                for d in range(-span, span + 1):
                    wt = span + 1 - abs(d)
                    lo, hi = max(0, -d), min(len(gs), len(gs) - d)
                    if hi <= lo:
                        continue
                    acc[lo:hi] += wt * gs[lo + d:hi + d]
                    wsum[lo:hi] += wt
                gs = acc / wsum
            if len(ks) == 1:
                vals = np.full(len(run), gs[0])
            else:
                vals = np.interp(run, ks, gs.real) + 1j * np.interp(run, ks, gs.imag)
            H[run % ru.N] = vals * np.exp(1j * tilt * run)
        return H

    def _he_track(self, Y, ru, heff):
        """Per-symbol common phase and timing slope from the pilots without assuming pilot
        values: squared BPSK pilots give the phase modulo pi, and continuity from the
        HE-LTF (phase 0) resolves it."""
        hp = heff[ru.pil_i]
        wp = np.abs(hp) ** 2
        kp = ru.pil_k.astype(float)
        E = Y[:, ru.pil_i] / np.where(wp > 0, hp, 1)
        nsym = Y.shape[0]
        ph = np.zeros(nsym)
        sl = np.zeros(nsym)
        phase, slope = 0.0, 0.0
        den = max(float(np.sum(wp * kp ** 2)), 1e-30)
        for n in range(nsym):
            e = E[n] * np.exp(-1j * slope * kp)
            c = 0.5 * np.angle(np.sum(e ** 2 * wp))
            c += np.pi * np.round((phase - c) / np.pi)
            d = (np.angle(e) - c + np.pi / 2) % np.pi - np.pi / 2
            slope += 0.3 * float(np.sum(wp * kp * d)) / den
            phase = c
            ph[n], sl[n] = c, slope
        return ph, sl

    def _he_data(self, pk, p):
        ru = HE_RUS[p["ru"]]
        bw, u, N = ru.bw, ru.u, ru.N
        cands = self._stream(pk, bw)
        if not cands:
            self._count("wide_out_of_span" if bw > 20 else "nonlegacy_no_stream")
            return [], pk["l_end"]
        lay = self._he_layout(ru, p)
        if lay is None or lay["n_pld"] <= 16:
            self._count("he_invalid_siga_length")
            return [], pk["l_end"]
        nsym, tsym, gi = p["nsym"], p["tsym"], p["gi"]
        end20 = pk["t_sig"] + p["data0"] + nsym * tsym
        if self.ctx is not None and bw > 20:
            self.ctx.claim(pk["s"], pk["s"] + end20 + 80, cands, bw)
        bk = self.BK * u
        best = None
        for st, s_w, w_w, fc in cands:
            need = (p["data0"] + nsym * tsym + 16) * u
            if s_w + need > len(st):
                continue
            y = st[s_w:s_w + need] * np.exp(-1j * w_w * np.arange(need))
            Hs, heff = self._he_chan(y, ru, p, bk)
            pw = np.abs(Hs[0][ru.tone_i]) ** 2
            lo_p, hi_p = pw[ru.tones < 0].mean() if (ru.tones < 0).any() else 1, pw[ru.tones > 0].mean()
            score = abs(math.log10(max(lo_p, 1e-30) / max(hi_p, 1e-30)))
            if best is None or score < best[0]:
                best = (score, y, Hs, heff, fc)
        if best is None:
            return [], pk["l_end"]
        _, y, Hs, heff, fc = best
        starts = (p["data0"] + np.arange(nsym) * tsym) * u
        idx = starts[:, None] + (gi * u - bk) + np.arange(N)[None, :]
        Y = np.fft.fft(y[idx], axis=1)
        ph, sl = self._he_track(Y, ru, heff)
        kfull = np.arange(N)
        kfull = np.where(kfull < N // 2, kfull, kfull - N)
        Y = Y * np.exp(-1j * (ph[:, None] + sl[:, None] * kfull[None, :]))
        mpdus = self._he_payload(Y, Hs, ru, p, lay)
        if mpdus is None:
            return [], pk["l_end"]
        self._count("he_ppdu")
        if not mpdus:
            self._count("ampdu_no_delimiters")
        kw = dict(mcs=p["mcs"], sgi=False, bw=bw, stbc=p["stbc"], ldpc=p["ldpc"], center_mhz=fc,
                  beamformed=p["beamformed"], ldpc_extra=p["extra"], dcm=p["dcm"],
                  he_er=p["er"], he_ru=p["ru"], bss_color=p["bss_color"], ul_dl=p["ul_dl"],
                  beam_change=p["beam_change"], spatial_reuse=p["spatial_reuse"], txop=p["txop"],
                  gi_samples=gi, ltf_mult=p["mult"], n_ltf=p["nltf"], pre_fec_a=p["a"],
                  pe_disamb=p["pe_disamb"], nsts=p["nsts"])
        return self._emit(mpdus, pk["s"], "he", pk["meta"], **kw), pk["s"] + end20

    def _he_payload(self, Y, Hs, ru, p, lay):
        """Equalize, demap and decode one user's data on RU ru. Y: FFT of the data symbols with
        common phase removed (nsym, N). Returns the MPDUs, or None."""
        nuse = lay.get("nsym_used", Y.shape[0])
        Y = Y[:nuse]
        di = ru.data_i
        if p["stbc"]:
            H0, H1 = Hs[0][di], Hs[1][di]
            g = np.abs(H0) ** 2 + np.abs(H1) ** 2
            g = np.maximum(g, 1e-30)
            Y0, Y1 = Y[0::2, di], Y[1::2, di]
            npair = min(len(Y0), len(Y1))
            Y0, Y1 = Y0[:npair], Y1[:npair]
            z = np.empty((2 * npair, ru.nsd), complex)
            z[0::2] = (np.conj(H0) * Y0 + H1 * np.conj(Y1)) / g
            z[1::2] = (np.conj(H0) * Y1 - H1 * np.conj(Y0)) / g
            wts = g
        else:
            hd = Hs[0][di]
            z = Y[:, di] / np.where(np.abs(hd) > 0, hd, 1)
            wts = np.abs(hd) ** 2
        wts = wts / max(float(np.median(wts)), 1e-30)
        nb = p["nbpsc"]
        nrow = z.shape[0]
        if p["dcm"]:
            soft = self._he_dcm(z, wts, nb, ru)
            if soft is None:
                return None
        else:
            soft = demap_gen(z.reshape(-1), nb).reshape(nrow, ru.nsd, nb) * wts[None, :, None]
        m, cb, cb_last = lay["m"], lay["ncbps"], lay["ncbps_last"]

        def assemble(per_sym):
            rows = [per_sym[n, :cb] for n in range(nrow - m)]
            rows += [per_sym[n, :cb_last] for n in range(max(0, nrow - m), nrow)]
            return np.concatenate(rows)

        if p["ldpc"]:
            tm = ru.tone_map(p["dcm"])
            inv = np.empty_like(tm)
            inv[tm] = np.arange(len(tm))
            info, bad = None, None
            for mp in (tm, inv):
                stream = assemble(soft[:, mp, :].reshape(nrow, -1))
                if len(stream) < lay["av_final"]:
                    stream = np.concatenate([stream, np.zeros(lay["av_final"] - len(stream))])
                i2, b2 = ldpc_decode_stream(stream[:lay["av_final"]], lay["lay"])
                if bad is None or b2 < bad:
                    info, bad = i2, b2
                if b2 == 0:
                    break
            if bad:
                self._count("ldpc_codeword_fail", bad)
            bits = ofdm_descramble(info)
            psdu_bits = lay["n_pld"] - 16
        else:
            perm = ru.bcc_perm(nb, p["dcm"])
            per_sym = soft.reshape(nrow, -1)[:, perm]
            streams = [assemble(per_sym)]
            if cb % 2:
                # MCS 0 with DCM on a 242/106-tone RU has an odd N_CBPS, so N_DBPS = N_CBPS / 2 is
                # not an integer. Two readings: the code fills each symbol up to 2 * N_DBPS bits,
                # or runs on across symbols. Both are tried; the FCS decides.
                rows = [per_sym[n, :2 * lay["ndbps"]] for n in range(nrow - m)]
                rows += [per_sym[n, :2 * lay["ndbps_last"]] for n in range(max(0, nrow - m), nrow)]
                streams.insert(0, np.concatenate(rows))
            psdu_bits = lay["n_pld"] - 16 - 6
            mpdus, psdu = None, b""
            for stream in streams:
                full = depuncture(stream, p["cr"], lay["n_pld"])
                if full is None:
                    continue
                bits = ofdm_descramble(VITERBI.decode(full))
                ps = bits_to_bytes(bits[16:16 + 8 * (psdu_bits // 8)])
                got = split_ampdu(ps)
                if mpdus is None or sum(fcs_ok(x) for x in got) > sum(fcs_ok(x) for x in mpdus):
                    mpdus, psdu = got, ps
            if mpdus is None:
                self._count("he_short_capture")
                return None
        if p["ldpc"]:
            psdu = bits_to_bytes(bits[16:16 + 8 * (psdu_bits // 8)])
            mpdus = split_ampdu(psdu)
        return mpdus

    def _he_dcm(self, z, wts, nb, ru):
        """Combine the two DCM copies. The upper-half map is chosen per PPDU from the
        candidates that best fit the received symbols (only the BPSK rule is confirmed
        against a published source), then soft bits come from both copies (max-log)."""
        half = ru.nsd // 2
        zl, zu = z[:, :half], z[:, half:2 * half]
        wl, wu = wts[:half], wts[half:2 * half]
        pts, labels = qam_points(nb)
        kidx = np.arange(half) + half
        sel = zl[:8], zu[:8]
        best, best_cost = None, None
        for cand in dcm_candidates(nb):
            U = dcm_apply(cand, pts, kidx, nb)            # (half, M)
            d = (wl[None, :, None] * np.abs(sel[0][:, :, None] - pts[None, None, :]) ** 2 +
                 wu[None, :, None] * np.abs(sel[1][:, :, None] - U[None, :, :]) ** 2)
            cost = float(d.min(axis=2).sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost
        self._count("he_dcm_map_%s" % "".join(str(int(bool(v))) for v in best[:3]))
        U = dcm_apply(best, pts, kidx, nb)
        d = (wl[None, :, None] * np.abs(zl[:, :, None] - pts[None, None, :]) ** 2 +
             wu[None, :, None] * np.abs(zu[:, :, None] - U[None, :, :]) ** 2)
        soft = np.empty((z.shape[0], half, nb))
        for j in range(nb):
            one = labels[:, j] == 1
            soft[:, :, j] = d[:, :, ~one].min(axis=2) - d[:, :, one].min(axis=2)
        return soft / 4

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
    if fr.phy == "he":
        present |= 1 << 23
        if (8 + len(body)) % 2:
            body.append(0)
        fmt = {"mu": 2, "tb": 3}.get(fr.he_fmt, 1 if fr.he_er else 0)
        data1 = fmt | 0x0004 | 0x0010 | 0x0020 | 0x0040 | 0x0080 | 0x0100 | 0x0200 | 0x0400 | 0x4000 | 0x8000
        if fmt < 2:
            data1 |= 0x0008                     # beam change known (SU / ER SU only)
        data2 = 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020 | 0x0040
        if fr.he_ru_off26 >= 0:
            data2 |= 0x4000 | ((fr.he_ru_off26 & 0x3F) << 8)
        data3 = ((fr.bss_color & 0x3F) | (fr.beam_change << 6) | (fr.ul_dl << 7) | ((fr.mcs & 0xF) << 8) |
                 (int(fr.dcm) << 12) | (int(fr.ldpc) << 13) | (int(fr.ldpc_extra) << 14) | (int(fr.stbc) << 15))
        data4 = fr.spatial_reuse & 0xF
        ru = fr.he_ru_code if fr.he_ru_code >= 0 else {"242": 0, "484": 1, "996": 2, "106u": 6}.get(fr.he_ru, 0)
        gi = {16: 0, 32: 1, 64: 2}.get(fr.gi_samples, 0)
        ltf = {1: 1, 2: 2, 4: 3}.get(fr.ltf_mult, 0)
        nltf = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}.get(fr.n_ltf, 0)
        data5 = (ru | (gi << 4) | (ltf << 6) | (nltf << 8) | ((fr.pre_fec_a & 3) << 12) |
                 (int(fr.beamformed) << 14) | (int(fr.pe_disamb) << 15))
        data6 = (fr.nsts & 0xF) | ((fr.txop & 0x7F) << 8)
        body += struct.pack("<6H", data1, data2, data3, data4, data5, data6)
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
    ap.add_argument("--he-log", help="write the decoded HE-SIG-A of every HE PPDU that passes its CRC to this CSV")
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
    he_log_file = None
    if args.he_log:
        import csv
        he_log_file = open(args.he_log, "w", newline="")
        orx.he_log = csv.writer(he_log_file)
        orx.he_log.writerow(["t_s", "channel", "l_length", "l_length_mod3", "er", "siga1_b0_b25", "siga2_b0_b25",
                             "format", "mcs", "dcm", "bw", "gi_ltf", "nsts_field", "ldpc", "stbc", "doppler"])
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
                orx.he_where = (ch, a / fs)
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
    if he_log_file:
        he_log_file.close()

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