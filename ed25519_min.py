"""Minimal pure-Python Ed25519 (RFC 8032 reference algorithm). Slow (~0.1 s/op) but dependency-free.
Used only to sign/verify the small canonical manifest of an organ bundle."""
import hashlib

p = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
d = -121665 * pow(121666, p - 2, p) % p
I = pow(2, (p - 1) // 4, p)


def _h(m):
    return hashlib.sha512(m).digest()


def _xrecover(y):
    xx = (y * y - 1) * pow(d * y * y + 1, p - 2, p)
    x = pow(xx, (p + 3) // 8, p)
    if (x * x - xx) % p != 0:
        x = x * I % p
    if x % 2 != 0:
        x = p - x
    return x


By = 4 * pow(5, p - 2, p) % p
Bx = _xrecover(By)
B = (Bx, By, 1, Bx * By % p)


def _add(P, Q):
    x1, y1, z1, t1 = P; x2, y2, z2, t2 = Q
    A = (y1 - x1) * (y2 - x2) % p; Bb = (y1 + x1) * (y2 + x2) % p
    C = 2 * t1 * t2 * d % p; D = 2 * z1 * z2 % p
    E, F, G, H = Bb - A, D - C, D + C, Bb + A
    return (E * F % p, G * H % p, F * G % p, E * H % p)


def _mul(P, e):
    Q = (0, 1, 1, 0)
    while e > 0:
        if e & 1:
            Q = _add(Q, P)
        P = _add(P, P); e >>= 1
    return Q


def _enc(P):
    x, y, z, _ = P
    zi = pow(z, p - 2, p); x, y = x * zi % p, y * zi % p
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _dec(s):
    y = int.from_bytes(s, "little"); sign = y >> 255; y &= (1 << 255) - 1
    x = _xrecover(y)
    if (x & 1) != sign:
        x = p - x
    P = (x, y, 1, x * y % p)
    if (-x * x + y * y - 1 - d * x * x * y * y) % p != 0:
        raise ValueError("bad point")
    return P


def _secret(seed):
    h = _h(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8; a |= 1 << 254
    return a, h[32:]


def public_key(seed: bytes) -> bytes:
    a, _ = _secret(seed)
    return _enc(_mul(B, a))


def sign(seed: bytes, msg: bytes) -> bytes:
    a, prefix = _secret(seed)
    A = _enc(_mul(B, a))
    r = int.from_bytes(_h(prefix + msg), "little") % L
    R = _enc(_mul(B, r))
    k = int.from_bytes(_h(R + A + msg), "little") % L
    return R + int.to_bytes((r + k * a) % L, 32, "little")


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        if len(sig) != 64:
            return False
        R, A = _dec(sig[:32]), _dec(pub)
        s = int.from_bytes(sig[32:], "little")
        if s >= L:
            return False
        k = int.from_bytes(_h(sig[:32] + pub + msg), "little") % L
        return _enc(_mul(B, s)) == _enc(_add(R, _mul(A, k)))
    except Exception:
        return False
