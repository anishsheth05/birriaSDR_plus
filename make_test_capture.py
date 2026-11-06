#!/usr/bin/env python3
"""Synthetic 802.11 transmitter for testing birria2pcap.py. Written from the
standard's equations, sharing no code with the receiver."""
import struct, sys, zlib, json
import numpy as np
from scipy.signal import resample_poly

rng = np.random.default_rng(7)

def bits_lsb(data):
    return [ (b >> i) & 1 for b in data for i in range(8) ]

def add_fcs(body):
    return body + struct.pack('<I', zlib.crc32(body))

# ---------------- OFDM ----------------
L = [1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,
     1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1]
S = {-24:1,-20:-1,-16:1,-12:-1,-8:-1,-4:1,4:-1,8:-1,12:1,16:1,20:1,24:1}

def ifft64(sc):  # sc dict k->value
    F = np.zeros(64, complex)
    for k, v in sc.items():
        F[k % 64] = v
    return np.fft.ifft(F) * 64 / np.sqrt(52)

def stf():
    sym = ifft64({k: np.sqrt(13/6) * v * (1+1j) for k, v in S.items()})
    return np.tile(sym, 3)[:160]

def ltf(ht=False):
    sc = {k: L[k+26] for k in range(-26, 27) if L[k+26]}
    if ht:
        sc.update({-28: 1, -27: 1, 27: -1, 28: -1})
        sym = ifft64(sc)
        return np.concatenate([sym[-16:], sym])
    sym = ifft64(sc)
    return np.concatenate([sym[-32:], sym, sym])

def scrambler_seq(state, n):
    # state: 7-bit list x1..x7; output = x7 ^ x4, shifted into x1
    x = list(state); out = []
    for _ in range(n):
        o = x[6] ^ x[3]
        out.append(o)
        x = [o] + x[:6]
    return out

POL = [1 - 2*b for b in scrambler_seq([1]*7, 127)]

def conv(bits):
    st = [0]*6; out = []
    g0 = [1,0,1,1,0,1,1]; g1 = [1,1,1,1,0,0,1]
    for b in bits:
        reg = [b] + st
        out.append(sum(r*g for r, g in zip(reg, g0)) % 2)
        out.append(sum(r*g for r, g in zip(reg, g1)) % 2)
        st = reg[:6]
    return out

PUNCT = {'1/2': [1,1], '2/3': [1,1,1,0], '3/4': [1,1,1,0,0,1], '5/6': [1,1,1,0,0,1,1,0,0,1]}

def puncture(c, r):
    p = PUNCT[r]
    return [v for i, v in enumerate(c) if p[i % len(p)]]

def interleave(bits, ncbps, nbpsc, ht):
    s = max(nbpsc // 2, 1); out = [0]*ncbps
    for k in range(ncbps):
        if ht:
            i = 4*nbpsc*(k % 13) + k // 13
            j = s*(i // s) + (i + ncbps - (13*i) // ncbps) % s
        else:
            i = (ncbps//16)*(k % 16) + k // 16
            j = s*(i // s) + (i + ncbps - (16*i) // ncbps) % s
        out[j] = bits[k]
    return out

def qam(bits, nbpsc):
    lv = {1: {(0,):-1, (1,):1},
          2: {(0,):-1, (1,):1},
          4: {(0,0):-3,(0,1):-1,(1,1):1,(1,0):3},
          6: {(0,0,0):-7,(0,0,1):-5,(0,1,1):-3,(0,1,0):-1,(1,1,0):1,(1,1,1):3,(1,0,1):5,(1,0,0):7}}
    norm = {1:1, 2:np.sqrt(2), 4:np.sqrt(10), 6:np.sqrt(42)}[nbpsc]
    out = []
    for i in range(0, len(bits), nbpsc):
        b = tuple(bits[i:i+nbpsc])
        if nbpsc == 1:
            out.append(lv[1][b] + 0j)
        else:
            h = nbpsc // 2
            out.append((lv[nbpsc][b[:h]] + 1j*lv[nbpsc][b[h:]]) / norm)
    return out

def ofdm_sym(data, k_list, pilots, gi=16):
    sc = dict(zip(k_list, data))
    sc.update(dict(zip([-21,-7,7,21], pilots)))
    t = ifft64(sc)
    return np.concatenate([t[-gi:], t])

KL = [k for k in range(-26,27) if k not in (0,-21,-7,7,21)]
KH = [k for k in range(-28,29) if k not in (0,-21,-7,7,21)]

def encode_field(bits24_or_48):
    return conv(bits24_or_48)

def lsig(rate_bits, length):
    b = list(rate_bits) + [0] + [(length >> i) & 1 for i in range(12)]
    b.append(sum(b) % 2)
    b += [0]*6
    return b

def data_bits(psdu, ndbps, seed):
    nb = 16 + 8*len(psdu) + 6
    nsym = -(-nb // ndbps)
    bits = [0]*16 + bits_lsb(psdu) + [0]*6
    bits += [0]*(nsym*ndbps - len(bits))
    sc = scrambler_seq(seed, len(bits))
    sb = [a ^ b for a, b in zip(bits, sc)]
    t0 = 16 + 8*len(psdu)
    sb[t0:t0+6] = [0]*6
    return sb, nsym

LEG = {6:('1101',1,'1/2'),9:('1111',1,'3/4'),12:('0101',2,'1/2'),18:('0111',2,'3/4'),
       24:('1001',4,'1/2'),36:('1011',4,'3/4'),48:('0001',6,'2/3'),54:('0011',6,'3/4')}
RNUM = {'1/2':0.5,'2/3':2/3,'3/4':0.75,'5/6':5/6}

def legacy_ppdu(psdu, mbps):
    rb, nbpsc, r = LEG[mbps]
    rate_bits = [int(c) for c in rb]
    ncbps = 48*nbpsc; ndbps = int(ncbps*RNUM[r])
    sig = interleave(conv(lsig(rate_bits, len(psdu))), 48, 1, False)
    out = [stf(), ltf(), ofdm_sym(qam(sig,1), KL, [p*POL[0] for p in (1,1,1,-1)])]
    sb, nsym = data_bits(psdu, ndbps, [1,0,1,1,1,0,1])
    coded = puncture(conv(sb), r)
    for n in range(nsym):
        chunk = interleave(coded[n*ncbps:(n+1)*ncbps], ncbps, nbpsc, False)
        out.append(ofdm_sym(qam(chunk, nbpsc), KL, [p*POL[(n+1)%127] for p in (1,1,1,-1)]))
    return np.concatenate(out)

def crc8_ht(bits):
    reg = [1]*8
    for b in bits:
        fb = reg[7] ^ b  # reg[7] = c7
        reg = [fb] + reg[:7]
        reg[1] ^= fb; reg[2] ^= fb  # x^1, x^2 taps
    return [1 - v for v in reg[::-1]]  # c7..c0

HTM = [(1,'1/2'),(2,'1/2'),(2,'3/4'),(4,'1/2'),(4,'3/4'),(6,'2/3'),(6,'3/4'),(6,'5/6')]

def ht_ppdu(psdu, mcs, sgi=False, aggregation=False):
    nbpsc, r = HTM[mcs]
    ncbps = 52*nbpsc; ndbps = int(ncbps*RNUM[r])
    nsym = -(-(16 + 8*len(psdu) + 6) // ndbps)
    tsym = 3.6 if sgi else 4.0
    txtime = 20 + 8 + 4 + 4 + 4*np.ceil(tsym*nsym/4)
    llen = int(np.ceil((txtime - 20)/4))*3 - 3
    sig = interleave(conv(lsig([1,1,0,1], llen)), 48, 1, False)
    out = [stf(), ltf(), ofdm_sym(qam(sig,1), KL, [p*POL[0] for p in (1,1,1,-1)])]
    h1 = [(mcs >> i) & 1 for i in range(7)] + [0] + [(len(psdu) >> i) & 1 for i in range(16)]
    h2 = [1, 1, 1, int(aggregation), 0, 0, 0, int(sgi), 0, 0]
    crc = crc8_ht(h1 + h2)
    hb = conv(h1 + h2 + crc + [0]*6)
    for k in range(2):
        il = interleave(hb[48*k:48*k+48], 48, 1, False)
        out.append(ofdm_sym([1j*v for v in qam(il,1)], KL, [p*POL[1+k] for p in (1,1,1,-1)]))
    out.append(stf()[:80])
    out.append(ltf(ht=True))
    sb, nsym2 = data_bits(psdu, ndbps, [0,1,1,0,1,0,1])
    assert nsym2 == nsym
    coded = puncture(conv(sb), r)
    psi = [1,1,1,-1]
    for n in range(nsym):
        chunk = interleave(coded[n*ncbps:(n+1)*ncbps], ncbps, nbpsc, True)
        pil = [psi[(n+m) % 4]*POL[(n+3) % 127] for m in range(4)]
        out.append(ofdm_sym(qam(chunk, nbpsc), KH, pil, gi=8 if sgi else 16))
    return np.concatenate(out)

def ampdu(mpdus):
    out = b''
    for i, m in enumerate(mpdus):
        ln = len(m)
        v = ((ln & 0xFFF) << 4) | (((ln >> 12) & 3) << 2)
        bits = [(v >> k) & 1 for k in range(16)]
        c = crc8_ht(bits)  # c7..c0, c7 transmitted first -> bit 0 of the byte
        cbyte = sum(b << k for k, b in enumerate(c))
        d = struct.pack('<HBB', v, cbyte, 0x4E)
        pad = b'\0' * ((-ln) % 4) if i < len(mpdus)-1 else b''
        out += d + m + pad
    return out

# ---------------- DSSS ----------------
BARKER = [1,-1,1,1,-1,1,1,1,-1,-1,-1]

def crc16_ccitt(bits):
    reg = 0xFFFF
    for b in bits:
        fb = ((reg >> 15) & 1) ^ b
        reg = (reg << 1) & 0xFFFF
        if fb: reg ^= 0x1021
    reg ^= 0xFFFF
    return [(reg >> (15-i)) & 1 for i in range(16)]  # c15 first

def dsss_scramble(bits, init):
    z = list(init)  # previous 7 outputs, z[-1] newest
    out = []
    for b in bits:
        o = b ^ z[-4] ^ z[-7]
        out.append(o); z.append(o)
    return out

def dsss_ppdu(psdu, mbps, short):
    sig = {1:0x0A, 2:0x14, 5.5:0x37, 11:0x6E}[mbps]
    n = len(psdu)
    if mbps == 1: lus = n*8
    elif mbps == 2: lus = n*4
    elif mbps == 5.5: lus = int(np.ceil(n*8/5.5))
    else: lus = int(np.ceil(n*8/11))
    service = 0x04
    if mbps == 11 and lus*11/8 - n >= 1: service |= 0x80
    hdr = [(sig>>i)&1 for i in range(8)] + [(service>>i)&1 for i in range(8)] + [(lus>>i)&1 for i in range(16)]
    hdr += crc16_ccitt(hdr)
    if short:
        pre = [0]*56 + [(0x05CF>>i)&1 for i in range(16)]
        init = [0,0,1,1,0,1,1][::-1]
    else:
        pre = [1]*128 + [(0xF3A0>>i)&1 for i in range(16)]
        init = [1,1,0,1,1,0,0][::-1]
    allbits = pre + hdr + bits_lsb(psdu)
    z = dsss_scramble(allbits, init)
    npre = len(pre)
    chips = []; ph = 0.0
    def barker_sym(phase):
        chips.extend(np.exp(1j*phase)*np.array(BARKER))
    # DBPSK preamble (+ long header)
    nb = npre + (48 if not short else 0)
    for b in z[:nb]:
        ph += np.pi*b; barker_sym(ph)
    pos = nb
    dq = {(0,0):0, (0,1):np.pi/2, (1,1):np.pi, (1,0):3*np.pi/2}
    if short:
        for i in range(24):
            ph += dq[(z[pos], z[pos+1])]; pos += 2; barker_sym(ph)
    if mbps == 1:
        for b in z[pos:]:
            ph += np.pi*b; barker_sym(ph)
    elif mbps == 2:
        for i in range(pos, len(z), 2):
            ph += dq[(z[i], z[i+1])]; barker_sym(ph)
    else:
        q = [0, np.pi/2, np.pi, 3*np.pi/2]
        step = 8 if mbps == 11 else 4
        for s, i in enumerate(range(pos, len(z), step)):
            d = z[i:i+step]
            ph += dq[(d[0], d[1])] + (np.pi if s % 2 else 0)
            if mbps == 11:
                p2, p3, p4 = q[d[2]*2+d[3]], q[d[4]*2+d[5]], q[d[6]*2+d[7]]
            else:
                p2, p3, p4 = d[2]*np.pi + np.pi/2, 0.0, d[3]*np.pi
            e = lambda a: np.exp(1j*a)
            chips.extend([e(ph+p2+p3+p4), e(ph+p3+p4), e(ph+p2+p4), -e(ph+p4),
                          e(ph+p2+p3), e(ph+p3), -e(ph+p2), e(ph)])
    return np.repeat(np.array(chips), 2)  # 22 MSps

# ---------------- frames ----------------
def mac(fc, a1, a2, a3, body, seq=0):
    return add_fcs(struct.pack('<HH', fc, 0) + a1 + a2 + a3 + struct.pack('<H', seq << 4) + body)

def beacon(ssid, bssid):
    body = struct.pack('<QHH', 123456789, 100, 0x0431) + bytes([0, len(ssid)]) + ssid.encode() + bytes([1, 4, 0x82, 0x84, 0x8b, 0x96])
    return mac(0x0080, b'\xff'*6, bssid, bssid, body)

def data(n, seq):
    return mac(0x0208, bytes.fromhex('020000000001'), bytes.fromhex('020000000002'), bytes.fromhex('020000000002'),
               bytes(rng.integers(0, 256, n, dtype=np.uint8)), seq)

def main():
    out = sys.argv[1]
    ppm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    fs = 80e6; center = 2437e6
    dur = 0.06
    total = int(fs*dur)
    sig = np.zeros(total, complex)
    expected = []
    t = 200e-6
    plan = []
    seq = 1
    for ch, cfo in ((6, 30e3), (1, -45e3), (11, 70e3)):
        plan.append((ch, cfo, 'dsss', (1, False), beacon('Birria-%d' % ch, bytes.fromhex('0a0b0c0d0e%02x' % ch))))
        for r in (6, 9, 12, 18, 24, 36, 48, 54):
            plan.append((ch, cfo, 'ofdm', r, data(int(rng.integers(30, 600)), seq))); seq += 1
        for mcs in range(8):
            plan.append((ch, cfo, 'ht', (mcs, mcs % 2 == 1, False), data(int(rng.integers(30, 900)), seq))); seq += 1
        plan.append((ch, cfo, 'ht', (5, False, True), None))
        for r, sh in ((2, True), (5.5, False), (11, True), (11, False), (1, True)):
            plan.append((ch, cfo, 'dsss', (r, sh), data(int(rng.integers(30, 400)), seq))); seq += 1
    # schedule: each channel has its own timeline so frames overlap in time across channels
    clock = {1: 150e-6, 6: 170e-6, 11: 190e-6}
    for ch, cfo, phy, p, frame in plan:
        if phy == 'ofdm':
            bb = legacy_ppdu(frame, p); fsb = 20e6; frames_here = [frame]
        elif phy == 'ht':
            mcs, sgi, agg = p
            if agg:
                m = [data(int(rng.integers(40, 300)), seq + i) for i in range(3)]; seq += 3
                frame = ampdu(m); frames_here = m
            else:
                frames_here = [frame]
            bb = ht_ppdu(frame, mcs, sgi, agg); fsb = 20e6
        else:
            bb = dsss_ppdu(frame, p[0], p[1]); fsb = 22e6; frames_here = [frame]
        x = resample_poly(bb, 40, 10) if fsb == 20e6 else resample_poly(bb, 40, 11)
        x = x / np.sqrt(np.mean(np.abs(x)**2))
        gain = 10**(rng.uniform(-6, 0)/20) * 0.12
        h = np.zeros(9, complex); h[0] = 1; h[8] = 0.2*np.exp(1j*1.1)  # 100 ns echo
        x = np.convolve(x, h)[:len(x)] * gain
        f_off = (2407 + 5*ch)*1e6 - center + cfo
        start = int(clock[ch]*fs) + int(rng.integers(0, 80))
        n = np.arange(len(x))
        if ppm:
            from scipy.signal import resample
            x = resample(x, int(round(len(x)/(1+ppm*1e-6))))
            n = np.arange(len(x))
        x = x * np.exp(2j*np.pi*f_off*(start + n)/fs + 1j*rng.uniform(0, 6.28))
        if start + len(x) > total:
            break
        sig[start:start+len(x)] += x
        for f in frames_here:
            expected.append(dict(ch=ch, phy=phy, p=str(p), t=start/fs, hex=f.hex()))
        clock[ch] += len(x)/fs + rng.uniform(30e-6, 120e-6)
    noise = (rng.normal(size=total) + 1j*rng.normal(size=total)) * 0.12/np.sqrt(2) * 10**(-28/20) * 2
    sig += noise
    iq = np.empty(2*total, np.int16)
    iq[0::2] = np.clip(np.round(sig.real*8192), -8192, 8191)
    iq[1::2] = np.clip(np.round(sig.imag*8192), -8192, 8191)
    iq.astype('<i2').tofile(out)
    json.dump(expected, open(out + '.json', 'w'))
    print('wrote', out, total, 'samples,', len(expected), 'frames; last clock', clock)

if __name__ == '__main__':
    main()
