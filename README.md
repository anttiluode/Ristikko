# Ristikko

*(Finnish: **lattice / grid**.)*

The composition test between two wave-computing frameworks — the Geometric
Neuron's lagged **Chiral Eye** and Kaiku's external **homodyne** carrier —
and the 2×2 lattice it completes. Built with [Kaiku](https://github.com/anttiluode/Kaiku)
as its parent.

> **Do not hype. Do not lie. Just show.**

## The one-sentence result

A single wave field can carry two quantities that `|ψ|²` is blind to — the
**parity** of its carrier signs (odd under a global phase flip) and an
**arrow** (odd under time reversal) — and the two references that read them
**compose cleanly**: a readout given both a lag reference and an external
homodyne reference recovers *both* sectors at once, at 1.000 / 1.000, with a
measured composition drop of **0.000** on each. The two references help;
they never fight.

## Why this was not obvious

It rests on Kaiku's EXP4, the parent result: parity and the arrow are odd
under **different Z₂ symmetries**, so they need **different** references.

- **Parity** is odd under `ψ → −ψ` (a global U(1) flip). A *lagged
  self-reference* `Im(ψ ψ̄_lag)` inherits the flip in **both** factors and
  stays blind — the Geometric Neuron's own Chiral Eye reads parity at
  chance (0.53, measured). Only an **external** fixed-phase carrier, which
  does not flip with the data, breaks the symmetry.
- **The arrow** is odd under time reversal / conjugation. The external
  carrier's magnitude is blind to it; the **lagged** reference reads it,
  precisely because reversal swaps the lag.

So a priori the two detection channels could have interfered — the caustic
four-wave-mixing regime that makes parity readable runs at max|ψ| ≈ 1000,
violent enough to scramble the delicate current information the arrow lives
in. The registered question was whether the arrow *survives that soup* and
whether the two references *coexist*. Both answers are yes.

## The lattice (hidden-arrow encoding, entropic medium, registered)

|  readout | reference | reads P (parity) | reads D (arrow) |
|---|---|---|---|
| **auto** `|ψ|²` | none | 0.484 — blind | 0.608 — ~blind |
| **lag** `Im(ψ ψ̄_lag)` | lagged self (Geometric Neuron) | 0.528 — blind | **1.000 — SEES** |
| **homodyne** `Re(ψ r̄)` | external carrier (Kaiku) | **1.000 — SEES** | 0.464 — blind |
| **both** | lag + homodyne | **1.000 — SEES** | **1.000 — SEES** |

All four registered predictions verified (C1–C4). **Composition (C3): both.P
− homodyne.P = 0.000, both.D − lag.D = 0.000.** The fourth box — a readout
that sees the arrow of time *and* solves odd-order logic simultaneously — is
real and constructive.

**Linear-medium control (C4):** parity is unreadable in *every* box (no
four-wave mixing exists to make it), while the arrow stays lag-readable at
1.000. This proves the arrow is present *before* the nonlinearity that reads
parity — the lattice is not an artifact of the caustic dynamics folding
labels into everything.

## Provenance — the honest bit

The **prop** encoding (arrow = literal propagation direction) was registered
and run first. It leaks: `|ψ|²` reads a right-mover-vs-left-mover from net
translation, so **auto-D = 0.90**, not chance — the "no reference, blind to
both" box fails, because propagation direction is *not* a purely
phase-hidden quantity. This is faithfully in `results/lattice.json` and the
table below.

The **hidden** encoding (arrow = a small current imbalance `ε=0.15` on a
balanced right+left standing envelope, zero net translation) was written
**after** seeing that leak, specifically to make the arrow `|ψ|²`-invisible
the way the Geometric Neuron's arrow is by construction (V13). With it,
auto-D drops to **0.608** and the null box nearly closes. The composition
verdict C3 was registered before any run and holds identically in **both**
encodings (drop 0.000 hidden, ≤0.016 prop), so the headline result does not
depend on the post-hoc encoding choice; only the cleanliness of the auto
null does. Disclosed, not sanded.

| encoding / medium | auto-D | reading |
|---|---|---|
| prop / entropic | 0.900 | arrow visible in intensity via translation |
| hidden / entropic | 0.608 | near-chance; residual is the ε imbalance |
| prop & hidden / linear | 1.000 | arrow trivially readable, no soup to hide it |

## What Ristikko is, and is not

**Is:** the meter that says *which conjugation-odd sectors a wave readout can
resolve, and whether references for different sectors coexist.* It places
two independent frameworks — the Geometric Neuron line and Kaiku — as the two
single-reference rows of one grid, and measures that the composed readout is
strictly at least as good as either.

**Is not:** a faster computer, a learned model, or a claim about biology. The
readouts are fixed linear projections; the only trained object is a logistic
head. The arrow here is a wave-medium analog, not the Geometric Neuron's
lag-covariance operator on neural data — the bridge is the *blindness
structure* (Kaiku EXP4), not an identity of objects. A genuinely hidden
arrow needs `ε → 0`, where lag-readability and auto-invisibility trade off;
mapping that trade-off is the open seam.

## The open seam

The `ε` knob is an uncertainty-flavored dial: large `ε` makes the arrow
lag-readable but slightly `|ψ|²`-visible; small `ε` hides it from `|ψ|²` but
weakens the lag signal. Whether there is a floor `ε` below which *no* readout
resolves D — a genuine resolution limit on the arrow, set by the medium's
noise and the caustic scrambling — is the falsifiable next question. That is
the wave-medium version of the Geometric Neuron's standing question about
how sharply the skew operator can be read at all.

## Files

```
ristikko/medium.py        the medium, direction DOF, the four feature operators
experiments/lattice.py    the registered lattice, both encodings, both media
results/lattice.json      every cell, every verdict, both encodings
app.py                    the HuggingFace Space -- the lattice, live
```

Run: `pip install -r requirements.txt && python app.py`

## Lineage

Parent: **Kaiku** (the interference soma + the EXP4 blindness theorem).
Grandparent: the **Geometric Neuron** line (`C_τ = S ⊕ A`; the arrow is the
skew half; V13's reciprocity wall). Ristikko is the small repo that shows
their two references are the two rows of one lattice, and that the lattice's
fourth box is inhabitable.

Do not hype. Do not lie. Just show.
