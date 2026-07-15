"""
ristikko.medium -- Kaiku's entropic wave medium, extended with a DIRECTION
degree of freedom so a lag operator has an arrow to read.

Two independent Z2 labels are encodable on the input field:
  P  parity of three carrier SIGNS (phases +/-)  -- odd under psi -> -psi
  D  propagation DIRECTION of the data carriers  -- odd under time reversal
P is invisible to |psi|^2 and to the lag operator (both factors flip);
readable only via an EXTERNAL fixed-phase reference (homodyne).
D is invisible to |psi|^2 of a symmetric config; readable via the
lag/current operator Im(psi psi*_lag). The reference carrier is ALWAYS a
right-mover at k_ref regardless of D, so the FWM channel that reads P
survives (or does not) the direction flip -- the composition question.
"""
import numpy as np

C = 1.0
DT = 0.2
KS = [0.23, 0.29, 0.37, 0.43]      # [reference, k1, k2, k3];  k1+k2-k3 = k_ref


def dx(f):
    return 0.5 * (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1))


def lap(f):
    return np.roll(f, -1, axis=-1) + np.roll(f, 1, axis=-1) - 2.0 * f


def make_field(signed_data, direction, N, width=60.0, center_frac=0.5, ref_on=True):
    B = signed_data.shape[0]
    x = np.arange(N)[None, :]
    env = np.exp(-(x - N * center_frac) ** 2 / (2 * width ** 2))
    data = np.zeros((B, N), dtype=np.complex128)
    for j in range(3):
        data += signed_data[:, j:j + 1] * env * np.exp(1j * KS[j + 1] * x)
    ref = np.zeros((B, N), dtype=np.complex128)
    if ref_on:
        refamp = np.abs(signed_data).mean(axis=1, keepdims=True)
        ref = refamp * env * np.exp(1j * KS[0] * x)
    psi = data + ref
    d = direction[:, None].astype(float)
    psid = (-C * d) * dx(data) + (-C) * dx(ref)
    return psi, psid


def propagate(psi, psid, mode, alpha=5.0, kappa=5.0, steps=2000, record_every=None):
    old = psi - DT * psid
    frames = []
    for t in range(steps):
        g = dx(psi)
        if mode == 'linear':
            a = lap(psi)
        elif mode == 'entropic':
            a = dx(1.0 / (1.0 + alpha * np.abs(g) ** 2) * g)
        elif mode == 'intensity':
            a = lap(psi) / (1.0 + kappa * np.abs(psi) ** 2)
        else:
            raise ValueError(mode)
        old, psi = psi, 2 * psi - old + DT ** 2 * a
        if record_every and t % record_every == 0:
            frames.append(np.abs(psi[0, ::4]) ** 2)
    if record_every:
        return psi, np.array(frames)
    return psi


NB = 24


def _bin(v):
    edges = np.linspace(0, v.shape[-1], NB + 1).astype(int)
    return np.stack([v[:, a:b].sum(1) for a, b in zip(edges, edges[1:])], 1)


def feat_auto(final):
    return _bin(np.abs(final) ** 2)


def feat_lag(final, taus=(3, 7, 15, 31)):
    outs = []
    for tau in taus:
        L = np.roll(final, tau, axis=1)
        outs.append(_bin(np.imag(final * np.conj(L))))
        outs.append(_bin(np.real(final * np.conj(L))))
    return np.concatenate(outs, 1)


def feat_homodyne(final):
    x = np.arange(final.shape[-1])[None, :]
    ref = np.exp(1j * KS[0] * x)
    return np.concatenate([_bin(np.real(final * np.conj(ref))),
                           _bin(np.imag(final * np.conj(ref)))], 1)


def feat_both(final):
    return np.concatenate([feat_lag(final), feat_homodyne(final)], 1)
