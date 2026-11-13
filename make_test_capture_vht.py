#!/usr/bin/env python3
"""Synthetic HT/VHT test capture for birria2pcap.py: 20/40/80 MHz, BCC or LDPC,
optional 2x1 STBC through two independent multipath channels, CFO and noise.

    python3 make_test_capture_vht.py t5g.cs16 5g [seed] [snr_db]   # 80 MSps at 5210 MHz
    python3 make_test_capture_vht.py t24.cs16 24 [seed] [snr_db]   # HT40 at 2437 MHz
    python3 birria2pcap.py t5g.cs16 -f 5210 -o t5g.pcap

Ground truth is written next to the capture as <file>.json. The transmit chain
is written from the standard separately from the receiver; only the LDPC
prototype tables (standard data) are taken from birria2pcap.py."""
import json, math, struct, sys, zlib
import numpy as np
from scipy.signal import resample_poly
import make_test_capture as G
from birria2pcap import LDPC_PROTO

PROTOS = {(n, r.numerator, r.denominator): p for (n, r), p in LDPC_PROTO.items()}


def perm(v, s):
    # (P^s v)[i] = v[(i + s) % Z]  for P^s = identity with columns shifted right by s
    return np.roll(v, -s)


class Encoder:
    def __init__(self, proto, n):
        self.P = proto
        self.Z = n // 24
        self.mb = proto.shape[0]
        self.kb = 24 - self.mb
        self.n = n
        self.k = self.kb * self.Z
        col = proto[:, self.kb]
        nz = [r for r in range(self.mb) if col[r] >= 0]
        assert len(nz) == 3 and nz[0] == 0 and nz[-1] == self.mb - 1, nz
        assert col[0] == col[-1], (col[0], col[-1])
        self.x = col[0]
        self.mid = nz[1]
        assert col[self.mid] == 0
        # remaining parity columns must be the dual diagonal of zero shifts
        for j in range(1, self.mb):
            c = proto[:, self.kb + j]
            assert c[j - 1] == 0 and c[j] == 0 and (c >= 0).sum() == 2, j

    def encode(self, info):
        Z, P = self.Z, self.P
        s = info.reshape(self.kb, Z)
        lam = np.zeros((self.mb, Z), np.uint8)
        for r in range(self.mb):
            for c in range(self.kb):
                if P[r, c] >= 0:
                    lam[r] ^= perm(s[c], P[r, c])
        p = np.zeros((self.mb, Z), np.uint8)
        # sum of all rows: lam_sum + p0 = 0 (P^x p0 appears twice, p_i telescope)
        p[0] = np.bitwise_xor.reduce(lam, axis=0)
        # row 0: lam0 + P^x p0 + p1 = 0
        p[1] = lam[0] ^ perm(p[0], self.x)
        for r in range(1, self.mb - 1):
            p[r + 1] = lam[r] ^ p[r] ^ (p[0] if r == self.mid else 0)
        cw = np.concatenate([info, p.reshape(-1)])
        return cw

    def check(self, cw):
        Z, P = self.Z, self.P
        v = cw.reshape(24, Z)
        for r in range(self.mb):
            acc = np.zeros(Z, np.uint8)
            for c in range(24):
                if P[r, c] >= 0:
                    acc ^= perm(v[c], P[r, c])
            if acc.any():
                return False
        return True

rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 11)

LNEG = G.L[:26]; LPOS = G.L[27:]; LBLK = LNEG + [1] + LPOS


def ltf_table(bw):
    if bw == 20:
        return -28, [1, 1] + LNEG + [0] + LPOS + [-1, -1]
    if bw == 40:
        return -58, LBLK + [-1, -1, -1, 1, 0, 0, 0, -1, 1, 1, -1] + LBLK
    a = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1]
    return -122, LBLK + a + LBLK + [1, -1, 1, -1, 0, 0, 0, 1, -1, -1, 1] + LBLK + a + LBLK


PIL = {20: [-21, -7, 7, 21], 40: [-53, -25, -11, 11, 25, 53], 80: [-103, -75, -39, -11, 11, 39, 75, 103]}
HTPIL = {(20, 1): [[1, 1, 1, -1]], (20, 2): [[1, 1, -1, -1], [1, -1, -1, 1]],
         (40, 1): [[1, 1, 1, -1, -1, 1]], (40, 2): [[1, 1, -1, -1, -1, -1], [1, 1, 1, -1, 1, 1]]}
VHTPIL = {20: [1, 1, 1, -1], 40: [1, 1, 1, -1, -1, 1], 80: [1, 1, 1, -1, -1, 1, 1, 1]}
P = [[1, -1], [1, 1]]
R = [1, -1]
ILV = {20: (13, 4), 40: (18, 6), 80: (26, 9)}
DTM = {20: 4, 40: 6, 80: 9}
RN = {'1/2': (1, 2), '2/3': (2, 3), '3/4': (3, 4), '5/6': (5, 6)}
MCS = [(1, '1/2'), (2, '1/2'), (2, '3/4'), (4, '1/2'), (4, '3/4'), (6, '2/3'), (6, '3/4'), (6, '5/6'),
       (8, '3/4'), (8, '5/6')]


def used_tones(bw):
    k0, v = ltf_table(bw)
    return [k0 + i for i, x in enumerate(v) if x]


def data_tones(bw):
    return [k for k in used_tones(bw) if k not in PIL[bw]]


def gamma(bw, k):
    if bw == 40:
        return 1j if k > 0 else 1
    if bw == 80:
        return -1 if k >= -64 else 1
    return 1


def ofdm(bw, sc, gi):
    N = 64 * bw // 20
    F = np.zeros(N, complex)
    for k, v in sc.items():
        F[k % N] = v * gamma(bw, k)
    t = np.fft.ifft(F) * N
    return t


def add_gi(t, gi):
    return np.concatenate([t[-gi:], t])


def dup_legacy(bw, sc20):
    if bw == 20:
        return dict(sc20)
    offs = [-32, 32] if bw == 40 else [-96, -32, 32, 96]
    out = {}
    for o in offs:
        for k, v in sc20.items():
            out[k + o] = v
    return out


def qam(bits, nb):
    """802.11 Gray mapping for BPSK..256-QAM (b0 = first bit, MSB of the Gray index)."""
    if nb == 1:
        return [2 * int(b) - 1 + 0j for b in bits]
    h = nb // 2
    M = 2 ** h
    norm = math.sqrt(2 * (M * M - 1) / 3)
    inv = {i ^ (i >> 1): i for i in range(M)}
    out = []
    for i in range(0, len(bits), nb):
        b = bits[i:i + nb]
        gi = int(''.join(str(int(v)) for v in b[:h]), 2); gq = int(''.join(str(int(v)) for v in b[h:]), 2)
        out.append(complex(-(M - 1) + 2 * inv[gi], -(M - 1) + 2 * inv[gq]) / norm)
    return out


def interleave(bits, bw, nb):
    ncol, rowf = ILV[bw]
    ncbps = len(bits); s = max(nb // 2, 1); out = [0] * ncbps
    for k in range(ncbps):
        i = rowf * nb * (k % ncol) + k // ncol
        j = s * (i // s) + (i + ncbps - (ncol * i) // ncbps) % s
        out[j] = bits[k]
    return out


def crc8(bits):
    return G.crc8_ht(bits)


# ---------------- LDPC PPDU encoding (19.3.11.7.5), independent implementation -----

def ldpc_params(n_pld, n_avbits, R_):
    a, b = R_; r = a / b
    if n_avbits <= 648:
        ncw, L = 1, (1296 if n_avbits >= n_pld + 912 * (1 - r) else 648)
    elif n_avbits <= 1296:
        ncw, L = 1, (1944 if n_avbits >= n_pld + 1464 * (1 - r) else 1296)
    elif n_avbits <= 1944:
        ncw, L = 1, 1944
    elif n_avbits <= 2592:
        ncw, L = 2, (1944 if n_avbits >= n_pld + 2916 * (1 - r) else 1296)
    else:
        ncw, L = math.ceil(n_pld / (1944 * r)), 1944
    k = L * a // b
    shrt = max(0, ncw * k - n_pld)
    punc = max(0, ncw * L - n_avbits - shrt)
    extra = ((punc > 0.1 * ncw * L * (1 - r) and shrt < 1.2 * punc * r / (1 - r)) or punc > 0.3 * ncw * L * (1 - r))
    return ncw, L, k, shrt, punc, extra


def ldpc_encode(payload, ncw, L, k, shrt, punc, rep, R_):
    enc = Encoder(PROTOS[(L,) + R_], L)
    out = []; pos = 0
    sp = lambda t, i: t // ncw + (1 if i < t % ncw else 0)
    for i in range(ncw):
        s, p, r = sp(shrt, i), sp(punc, i), sp(rep, i)
        info = np.zeros(k, np.uint8)
        info[:k - s] = payload[pos:pos + k - s]; pos += k - s
        cw = enc.encode(info)
        kept = list(cw[:k - s]) + list(cw[k:L - p])
        reps = [kept[j % len(kept)] for j in range(r)]
        out += kept + reps
    return out


# ---------------- PPDU builder -----

def build(fmt, bw, mcs, payload, sgi=False, stbc=False, ldpc=False, agg=False):
    """Returns a list of per-antenna baseband arrays at bw MSps."""
    u = bw // 20; N = 64 * u
    nsts = 2 if stbc else 1; m_stbc = 2 if stbc else 1
    nb, cr = MCS[mcs]; R_ = RN[cr]
    dk = data_tones(bw); nsd = len(dk)
    ncbps = nsd * nb; ndbps = ncbps * R_[0] // R_[1]
    nltf = 2 if nsts == 2 else 1
    tsym = 3.6 if sgi else 4.0
    gi = 8 * u if sgi else 16 * u
    # ---- data bits
    if fmt == 'ht':
        psdu = payload
        if ldpc:
            n_pld = 8 * len(psdu) + 16
            n_av = ncbps * m_stbc * math.ceil(n_pld / (ncbps * R_[0] / R_[1] * m_stbc))
            ncw, L, k, shrt, punc, extra = ldpc_params(n_pld, n_av, R_)
            if extra:
                n_av += ncbps * m_stbc; punc = max(0, ncw * L - n_av - shrt)
            rep = max(0, round(n_av - ncw * L * (1 - R_[0] / R_[1]) - n_pld))
            bits = [0] * 16 + G.bits_lsb(psdu)
            sb = [a ^ b for a, b in zip(bits, G.scrambler_seq([1, 1, 0, 1, 0, 0, 1], len(bits)))]
            coded = ldpc_encode(np.array(sb, np.uint8), ncw, L, k, shrt, punc, rep, R_)
            nsym = n_av // ncbps
            assert len(coded) == n_av, (len(coded), n_av)
        else:
            nsym = m_stbc * math.ceil((16 + 8 * len(psdu) + 6) / (m_stbc * ndbps))
            bits = [0] * 16 + G.bits_lsb(psdu) + [0] * 6
            bits += [0] * (nsym * ndbps - len(bits))
            sb = [a ^ b for a, b in zip(bits, G.scrambler_seq([1, 0, 0, 1, 1, 0, 1], len(bits)))]
            t0 = 16 + 8 * len(psdu); sb[t0:t0 + 6] = [0] * 6
            coded = G.puncture(G.conv(sb), cr)
        length_field = len(psdu)
    else:  # vht, payload is the A-MPDU (APEP)
        apep = len(payload)
        sigb_len = math.ceil(apep / 4)
        nbits_sigb = {20: 17, 40: 19, 80: 21}[bw]; nres = {20: 3, 40: 2, 80: 2}[bw]
        sigb = [(sigb_len >> i) & 1 for i in range(nbits_sigb)] + [1] * nres
        sigb_crc = crc8(sigb)
        service = [0] * 8 + sigb_crc
        if ldpc:
            nsym_init = m_stbc * math.ceil((8 * apep + 16) / (m_stbc * ndbps))
            n_pld = nsym_init * ndbps
            n_av = nsym_init * ncbps
            ncw, L, k, shrt, punc, extra = ldpc_params(n_pld, n_av, R_)
            nsym = nsym_init + (m_stbc if extra else 0)
            if extra:
                n_av += ncbps * m_stbc; punc = max(0, ncw * L - n_av - shrt)
            rep = max(0, round(n_av - ncw * L * (1 - R_[0] / R_[1]) - n_pld))
            psdu_len = (n_pld - 16) // 8
        else:
            nsym = m_stbc * math.ceil((8 * apep + 16 + 6) / (m_stbc * ndbps))
            psdu_len = (nsym * ndbps - 16 - 6) // 8
            extra = False
        pad = psdu_len - apep
        eof = b''
        while pad >= 4:
            v = 1  # EOF=1, length 0
            c = crc8([(v >> i) & 1 for i in range(16)])
            eof += struct.pack('<HBB', v, sum(b << i for i, b in enumerate(c)), 0x4E); pad -= 4
        psdu = payload + eof + b'\0' * pad
        bits = service + G.bits_lsb(psdu)
        total = (n_pld if ldpc else nsym * ndbps - 6)
        bits += [0] * (total - len(bits))
        sb = [a ^ b for a, b in zip(bits, G.scrambler_seq([0, 1, 1, 1, 0, 0, 1], len(bits)))]
        if ldpc:
            coded = ldpc_encode(np.array(sb, np.uint8), ncw, L, k, shrt, punc, rep, R_)
            assert len(coded) == n_av, (len(coded), n_av)
        else:
            coded = G.puncture(G.conv(sb + [0] * 6), cr)
    # ---- per-symbol constellation points (nsym, nsd), stream 0 view
    syms = []
    tm = None
    if ldpc and fmt == 'vht':
        d = DTM[bw]; kk = np.arange(nsd); tm = d * (kk % (nsd // d)) + (kk * d) // nsd
    for n in range(nsym):
        chunk = coded[n * ncbps:(n + 1) * ncbps]
        if not ldpc:
            chunk = interleave(chunk, bw, nb)
        pts = np.array(qam(chunk, nb))
        if tm is not None:
            mapped = np.zeros(nsd, complex); mapped[tm] = pts; pts = mapped
        syms.append(pts)
    syms = np.array(syms)
    # ---- STBC
    sts = [syms]
    if stbc:
        a0 = syms.copy(); a1 = np.zeros_like(syms)
        a1[0::2] = -np.conj(syms[1::2]); a1[1::2] = np.conj(syms[0::2])
        sts = [a0, a1]
    # ---- legacy fields (duplicated, CSD on antenna 2)
    lsig_rate = [1, 1, 0, 1]
    if fmt == 'ht':
        txtime = 20 + 8 + 4 + 4 * nltf + (4 * math.ceil(tsym * nsym / 4) if sgi else 4 * nsym)
    else:
        txtime = 20 + 8 + 4 + 4 * nltf + 4 + (4 * math.ceil(tsym * nsym / 4) if sgi else 4 * nsym)
    llen = math.ceil((txtime - 20) / 4) * 3 - 3
    STF20 = {k: np.sqrt(13 / 6) * v * (1 + 1j) for k, v in G.S.items()}
    LTF20 = {k: G.L[k + 26] for k in range(-26, 27) if G.L[k + 26]}
    sig = G.interleave(G.conv(G.lsig(lsig_rate, llen)), 48, 1, False)

    def sig20(pts, pol):
        sc = dict(zip(G.KL, pts)); sc.update(dict(zip([-21, -7, 7, 21], [p * pol for p in (1, 1, 1, -1)])))
        return sc
    legacy_syms = []  # (subcarrier dict, gi) per symbol
    legacy_syms.append(('stf', dup_legacy(bw, STF20)))
    legacy_syms.append(('ltf', dup_legacy(bw, LTF20)))
    legacy_syms.append(('sym', dup_legacy(bw, sig20(G.qam(sig, 1), G.POL[0]))))
    if fmt == 'ht':
        h1 = [(mcs >> i) & 1 for i in range(7)] + [1 if bw == 40 else 0] + [(length_field >> i) & 1 for i in range(16)]
        h2 = [1, 1, 1, int(agg), int(stbc), 0, int(ldpc), int(sgi), 0, 0]
        hb = G.conv(h1 + h2 + crc8(h1 + h2) + [0] * 6)
    else:
        a1 = [{20: 0, 40: 1, 80: 2}[bw] & 1, {20: 0, 40: 1, 80: 2}[bw] >> 1, 1, int(stbc)] + [0] * 6
        a1 += [((nsts - 1) >> i) & 1 for i in range(3)] + [(77 >> i) & 1 for i in range(9)] + [0, 1]
        a2 = [int(sgi), int(sgi and nsym % 10 == 9), int(ldpc), int(bool(extra))] + [(mcs >> i) & 1 for i in range(4)] + [0, 1]
        hb = G.conv(a1 + a2 + crc8(a1 + a2) + [0] * 6)
    for kk in range(2):
        il = G.interleave(hb[48 * kk:48 * kk + 48], 48, 1, False)
        pts = G.qam(il, 1)
        if fmt == 'ht' or kk == 1:
            pts = [1j * v for v in pts]
        legacy_syms.append(('sym', dup_legacy(bw, sig20(pts, G.POL[1 + kk]))))
    # ---- per antenna waveform
    k_used = used_tones(bw); k0, ltfv = ltf_table(bw); LT = {k0 + i: v for i, v in enumerate(ltfv) if v}
    ants = []
    for a in range(nsts):
        out = []
        lcs = -4 * u * a      # -200 ns legacy CSD
        ncs = -8 * u * a      # -400 ns non-legacy CSD
        for kind, sc in legacy_syms:
            t = np.roll(ofdm(bw, sc, 16 * u), lcs)
            if kind == 'stf':
                out.append(np.tile(t, 3)[:160 * u][- 160 * u:] if False else np.concatenate([t[-32 * u:], t, t])[:160 * u])
            elif kind == 'ltf':
                out.append(np.concatenate([t[-32 * u:], t, t]))
            else:
                out.append(add_gi(t, 16 * u))
        # HT/VHT-STF
        t = np.roll(ofdm(bw, dup_legacy(bw, STF20), 16 * u), ncs)
        out.append(add_gi(t, 16 * u))
        # LTFs
        pil = set(PIL[bw])
        for li in range(nltf):
            sc = {}
            for k, v in LT.items():
                if fmt == 'vht' and k in pil:
                    sc[k] = v * R[li]
                else:
                    sc[k] = v * (P[a][li] if nsts == 2 else 1)
            out.append(add_gi(np.roll(ofdm(bw, sc, 16 * u), ncs), 16 * u))
        # VHT-SIG-B
        if fmt == 'vht':
            sb_bits = sigb + [0] * 6
            sb_bits = sb_bits * {20: 1, 40: 2, 80: 4}[bw] + ([0] if bw == 80 else [])
            il = interleave(G.conv(sb_bits), bw, 1)
            pts = qam(il, 1)
            sc = dict(zip(dk, pts))
            sc.update(dict(zip(PIL[bw], [v * G.POL[3] for v in VHTPIL[bw]])))
            out.append(add_gi(np.roll(ofdm(bw, sc, 16 * u), ncs), 16 * u))
        # data
        npil = len(PIL[bw])
        for n in range(nsym):
            sc = dict(zip(dk, sts[a][n]))
            if fmt == 'vht':
                pv = [VHTPIL[bw][(n + m) % npil] * G.POL[(n + 4) % 127] for m in range(npil)]
            else:
                pat = HTPIL[(bw, nsts)][a]
                pv = [pat[(n + m) % npil] * G.POL[(n + 3) % 127] for m in range(npil)]
            sc.update(dict(zip(PIL[bw], pv)))
            out.append(add_gi(np.roll(ofdm(bw, sc, gi), ncs), gi))
        ants.append(np.concatenate(out))
    return ants


def eof_ampdu(mpdus):
    return G.ampdu(mpdus)


def main():
    out = sys.argv[1]
    band = sys.argv[2]
    fs = 80e6
    if band == '5g':
        center = 5210e6
        dur = 0.05
    else:
        center = 2437e6
        dur = 0.02
    total = int(fs * dur)
    sig = np.zeros(total, complex)
    expected = []
    seq = [1000]

    def frame(n):
        seq[0] += 1
        return G.data(n, seq[0] % 4096)
    plan = []
    if band == '5g':
        for mcs, sgi, stbc, ldpc in ((0, False, False, False), (4, True, False, False), (7, False, False, False),
                                      (9, False, False, False), (3, False, False, True), (8, True, False, True),
                                      (2, False, True, False), (5, False, True, True)):
            plan.append(('vht', 80, 5210, mcs, sgi, stbc, ldpc))
        for fc, mcs, sgi, stbc, ldpc in ((5190, 1, False, False, False), (5230, 8, False, False, False),
                                         (5190, 6, True, False, True), (5230, 4, False, True, False)):
            plan.append(('vht', 40, fc, mcs, sgi, stbc, ldpc))
        for fc, mcs, sgi, stbc, ldpc in ((5180, 0, False, False, False), (5200, 5, True, False, False),
                                         (5220, 8, False, False, False), (5240, 7, False, False, True),
                                         (5200, 3, False, True, True)):
            plan.append(('vht', 20, fc, mcs, sgi, stbc, ldpc))
        for fc, mcs, sgi, stbc, ldpc in ((5190, 7, False, False, False), (5230, 3, False, False, True),
                                         (5190, 2, False, True, False), (5230, 5, True, False, False)):
            plan.append(('ht', 40, fc, mcs, sgi, stbc, ldpc))
        for fc, mcs, sgi, stbc, ldpc in ((5180, 4, False, False, True), (5240, 1, False, True, False),
                                         (5220, 6, True, True, True)):
            plan.append(('ht', 20, fc, mcs, sgi, stbc, ldpc))
    else:
        plan += [('ht', 40, 2447, 7, False, False, False), ('ht', 40, 2427, 4, False, False, True),
                 ('ht', 40, 2447, 2, True, True, False), ('ht', 20, 2437, 3, False, True, True)]
    clock = 100e-6
    for fmt, bw, fcm, mcs, sgi, stbc, ldpc in plan:
        if fmt == 'vht':
            mp = [frame(int(rng.integers(40, 700))) for _ in range(int(rng.integers(1, 4)))]
            payload = eof_ampdu(mp)
            truth = mp
            agg = True
        else:
            agg = bool(rng.integers(0, 2))
            if agg:
                mp = [frame(int(rng.integers(40, 500))) for _ in range(2)]
                payload = eof_ampdu(mp); truth = mp
            else:
                payload = frame(int(rng.integers(40, 900))); truth = [payload]
        ants = build(fmt, bw, mcs, payload, sgi, stbc, ldpc, agg)
        rx = 0
        for a, x in enumerate(ants):
            up = resample_poly(x, 80 // bw, 1) if bw < 80 else x
            h = np.zeros(10, complex); h[0] = 1
            h[rng.integers(3, 10)] = 0.3 * np.exp(1j * rng.uniform(0, 6.28))
            h *= np.exp(1j * rng.uniform(0, 6.28))
            y = np.convolve(up, h)[:len(up)]
            rx = rx + y if isinstance(rx, np.ndarray) else y
        rx = rx / np.sqrt(np.mean(np.abs(rx) ** 2)) * 0.12 * 10 ** (rng.uniform(-4, 0) / 20)
        cfo = rng.uniform(-60e3, 60e3)
        f_off = fcm * 1e6 - center + cfo
        start = int(clock * fs)
        if start + len(rx) > total:
            print('ran out of room at', fmt, bw, mcs); break
        n = np.arange(len(rx))
        sig[start:start + len(rx)] += rx * np.exp(2j * np.pi * f_off * (start + n) / fs)
        for f in truth:
            expected.append(dict(fmt=fmt, bw=bw, fc=fcm, mcs=mcs, sgi=sgi, stbc=stbc, ldpc=ldpc, t=start / fs, hex=f.hex()))
        clock += len(rx) / fs + 40e-6
    snr_db = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    noise = (rng.normal(size=total) + 1j * rng.normal(size=total)) * 0.12 * 10 ** (-snr_db / 20) / np.sqrt(2)
    sig += noise
    iq = np.empty(2 * total, np.int16)
    iq[0::2] = np.clip(np.round(sig.real * 8192), -8192, 8191)
    iq[1::2] = np.clip(np.round(sig.imag * 8192), -8192, 8191)
    iq.astype('<i2').tofile(out)
    json.dump(expected, open(out + '.json', 'w'))
    print('wrote', out, 'center', center, len(plan), 'PPDUs', len(expected), 'MPDUs, used %.1f ms' % (clock * 1e3))


if __name__ == '__main__':
    main()
