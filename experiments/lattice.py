"""
RISTIKKO -- the reference lattice of wave-computing capabilities.
==========================================================================
One wave field carries two INDEPENDENT conjugation-odd labels:
  P  parity of 3 carrier signs   -- odd under global phase flip psi -> -psi
  D  a time-reversal-odd arrow    -- odd under conjugation / mover-swap

Each is invisible to |psi|^2 and recoverable only through a cross-term
against a reference -- but a DIFFERENT reference, because they are odd under
DIFFERENT Z2 symmetries (Ristikko's parent result, Kaiku EXP4):
  P needs an EXTERNAL fixed-phase reference (homodyne); a lagged self-
    reference inherits the flip and stays blind.
  D needs a LAGGED reference (the Chiral Eye Im(psi psi*_lag)); the
    external carrier's magnitude is blind to it.

The lattice: classify P and D from four readouts of the same field.
                       sees P?   sees D?
  auto      |psi|^2      no        no       <- no reference
  lag       Chiral Eye   no        YES      <- lag reference (Geometric Neuron)
  homodyne  ext carrier  YES       no       <- homodyne reference (Kaiku)
  both      lag+homodyne YES       YES      <- the composition question

TWO ARROW ENCODINGS (see PROVENANCE in README):
  prop    D = propagation direction. Partially hidden: |psi|^2 sees it via
          translation (auto-D leaks ~0.90).
  hidden  D = small current imbalance on a balanced standing envelope.
          Fully hidden: no net translation, auto-D -> chance. Faithful
          analog of the GN arrow (|.|^2-invisible by V13). Found AFTER
          seeing the prop leak -- disclosed post-hoc.

=== REGISTERED (before any run) ===
C1  homodyne reads P (>=0.90), blind to D (<=0.65).
C2  lag reads D (>=0.90), blind to P (<=0.65).
C3  COMPOSITION: both reads P AND D, no destructive interference:
      both.P >= homodyne.P - 0.05  AND  both.D >= lag.D - 0.05.
    KILL: either drops > 0.05.
C4  linear control: P unreadable everywhere (<=0.65); D still lag-readable
    (>=0.90) -- the arrow exists before the nonlinearity that reads P.

Do not hype. Do not lie. Just show.
"""
import numpy as np, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ristikko.medium import (dx, C as CC, KS, make_field, propagate,
                             feat_auto, feat_lag, feat_homodyne, feat_both)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N, STEPS, A, JIT, EPS = 768, 2000, 1.5, 0.15, 0.15
FEATS = {'auto': feat_auto, 'lag': feat_lag, 'homodyne': feat_homodyne, 'both': feat_both}


def make_hidden(signed, D, width=60.0):
    B = signed.shape[0]
    x = np.arange(N)[None, :]
    env = np.exp(-(x - N * 0.5) ** 2 / (2 * width ** 2))
    data = np.zeros((B, N), dtype=np.complex128)
    for j in range(3):
        data += signed[:, j:j + 1] * env * np.exp(1j * KS[j + 1] * x)
    Ds = np.where(D == 1, -1.0, 1.0)[:, None]
    right, left = data, np.conj(data)
    psi = (1 + EPS * Ds) * right + (1 - EPS * Ds) * left
    psid = (1 + EPS * Ds) * (-CC * dx(right)) + (1 - EPS * Ds) * (+CC * dx(left))
    return psi, psid


def batch(n, rng):
    bits = rng.integers(0, 2, size=(n, 3))
    P = bits.sum(1) % 2
    amps = A + JIT * rng.standard_normal((n, 3))
    signed = amps * np.where(bits == 1, -1.0, 1.0)
    D = rng.integers(0, 2, size=n)
    return signed, P, D


def acc(Ftr, ytr, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    return float(LogisticRegression(max_iter=3000).fit(sc.transform(Ftr), ytr).score(sc.transform(Fte), yte))


def field_for(signed, D, mode, encoding):
    if encoding == 'hidden':
        psi, psid = make_hidden(signed, D)
    else:
        direction = np.where(D == 1, -1.0, 1.0)
        psi, psid = make_field(signed, direction, N, ref_on=False)
    return propagate(psi, psid, mode, steps=STEPS)


def run(mode, encoding, seed=1, ntr=500, nte=250):
    rng = np.random.default_rng(seed)
    s1, P1, D1 = batch(ntr, rng)
    s2, P2, D2 = batch(nte, rng)
    f1, f2 = field_for(s1, D1, mode, encoding), field_for(s2, D2, mode, encoding)
    table = {nm: {'P': acc(fn(f1), P1, fn(f2), P2), 'D': acc(fn(f1), D1, fn(f2), D2)}
             for nm, fn in FEATS.items()}
    return table, float(np.abs(f1).max())


if __name__ == '__main__':
    t0 = time.time()
    out = {}
    for encoding in ['hidden', 'prop']:
        out[encoding] = {}
        for mode in ['entropic', 'linear']:
            table, mx = run(mode, encoding)
            out[encoding][mode] = {'table': table, 'max_abs_field': mx}
            print(f"\n=== {encoding} / {mode} (max|psi|={mx:.0f}) ({time.time()-t0:.0f}s)")
            print(f"{'readout':10s}{'P':>8s}{'D':>8s}")
            for nm, a in table.items():
                print(f"{nm:10s}{a['P']:8.3f}{a['D']:8.3f}")

    E = out['hidden']['entropic']['table']
    Lc = out['hidden']['linear']['table']
    verdict = dict(
        C1=bool(E['homodyne']['P'] >= 0.90 and E['homodyne']['D'] <= 0.65),
        C2=bool(E['lag']['D'] >= 0.90 and E['lag']['P'] <= 0.65),
        C3_compose=bool(E['both']['P'] >= E['homodyne']['P'] - 0.05
                        and E['both']['D'] >= E['lag']['D'] - 0.05),
        C3_P_drop=float(E['homodyne']['P'] - E['both']['P']),
        C3_D_drop=float(E['lag']['D'] - E['both']['D']),
        C4=bool(max(Lc['auto']['P'], Lc['lag']['P'], Lc['homodyne']['P'], Lc['both']['P']) <= 0.65
                and Lc['lag']['D'] >= 0.90),
        auto_D_leak_hidden=float(E['auto']['D']),
        auto_D_leak_prop=float(out['prop']['entropic']['table']['auto']['D']))
    print("\n" + json.dumps(verdict, indent=1))
    json.dump(dict(results=out, verdict=verdict,
                   config=dict(N=N, steps=STEPS, A=A, jitter=JIT, eps=EPS,
                               alpha=5.0, seed=1, encodings=['hidden', 'prop'])),
              open(os.path.join(os.path.dirname(__file__), '..', 'results', 'lattice.json'), 'w'), indent=1)
