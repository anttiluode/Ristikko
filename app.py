"""
Ristikko -- the 2x2 lattice of wave-computing capabilities.
Gradio app. Physics in ristikko/medium.py; the lattice is precomputed in
results/lattice.json and rendered live per input here.
"""
import json
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ristikko.medium import KS, dx, C as CC, propagate

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = json.load(open(os.path.join(HERE, "results", "lattice.json")))
EPS = LAT["config"]["eps"]
N = LAT["config"]["N"]

CELL = {  # (reads P, reads D) per readout, from the hidden/entropic table
    "auto | no reference": ("blind", "blind"),
    "lag | Chiral Eye (GN)": ("blind", "SEES"),
    "homodyne | ext carrier (Kaiku)": ("SEES", "blind"),
    "both | composed": ("SEES", "SEES"),
}


def make_hidden_single(signs, D, amp=1.5, width=60.0):
    x = np.arange(N)[None, :]
    env = np.exp(-(x - N * 0.5) ** 2 / (2 * width ** 2))
    data = np.zeros((1, N), dtype=np.complex128)
    for j in range(3):
        data += (amp * signs[j]) * env * np.exp(1j * KS[j + 1] * x)
    Ds = -1.0 if D == 1 else 1.0
    right, left = data, np.conj(data)
    psi = (1 + EPS * Ds) * right + (1 - EPS * Ds) * left
    psid = (1 + EPS * Ds) * (-CC * dx(right)) + (1 - EPS * Ds) * (+CC * dx(left))
    return psi, psid


def lattice_figure():
    t = LAT["results"]["hidden"]["entropic"]["table"]
    order = ["auto", "lag", "homodyne", "both"]
    labels = ["auto\n(no ref)", "lag\n(Chiral Eye)", "homodyne\n(ext carrier)", "both\n(composed)"]
    P = [t[o]["P"] for o in order]
    D = [t[o]["D"] for o in order]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(4)
    ax.bar(x - 0.19, P, 0.36, label="reads P (parity)", color="crimson")
    ax.bar(x + 0.19, D, 0.36, label="reads D (arrow)", color="steelblue")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.axhline(0.9, color="green", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("classification accuracy")
    ax.set_title("The lattice: which reference unlocks which conjugation-odd sector")
    ax.legend(loc="center left")
    fig.tight_layout()
    return fig


def run_field(b1, b2, b3, D_left):
    signs = [(-1.0 if b else 1.0) for b in (b1, b2, b3)]
    Pbit = int(sum(1 for b in (b1, b2, b3) if b) % 2)
    Dbit = 1 if D_left else 0
    psi, psid = make_hidden_single(signs, Dbit)
    final, frames = propagate(psi.copy(), psid.copy(), "entropic", steps=2000,
                              record_every=16)
    x = np.arange(N)
    ref = np.exp(1j * KS[0] * x)
    homo = float(np.real(final[0] * np.conj(ref)).sum())
    lagop = float(np.imag(final[0] * np.conj(np.roll(final[0], 7))).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(frames, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, N, 0, 400])
    axes[0].set(xlabel="x", ylabel="time", title=f"|psi|^2  (max reached ~1000, caustic FWM)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].bar(["homodyne\n(-> P)", "lag\n(-> D)"], [np.sign(homo), np.sign(lagop)],
                color=["crimson", "steelblue"])
    axes[1].set(ylim=(-1.4, 1.4), yticks=[-1, 0, 1],
                title="sign of each reference cross-term (single shot)")
    fig.tight_layout()

    msg = (f"true parity P = **{Pbit}**  |  true arrow D = **{'left' if Dbit else 'right'}**\n\n"
           f"homodyne cross-term sign: {'+' if homo >= 0 else '-'}  (carries P)  |  "
           f"lag cross-term sign: {'+' if lagop >= 0 else '-'}  (carries D)\n\n"
           f"*Both live in the same field at once. |psi|^2 alone sees neither "
           f"(auto: P={LAT['results']['hidden']['entropic']['table']['auto']['P']:.2f}, "
           f"D={LAT['results']['hidden']['entropic']['table']['auto']['D']:.2f}).*")
    return fig, msg


with gr.Blocks(title="Ristikko -- the wave-computing lattice") as demo:
    gr.Markdown(
        "# Ristikko\n*(Finnish: **lattice / grid**)*\n\n"
        "A wave field can carry two things that `|psi|^2` cannot see: the "
        "**parity** of its carrier signs (odd under a global phase flip) and an "
        "**arrow** (odd under time reversal). Each is readable only through a "
        "cross-term against a reference -- but a *different* reference. "
        "The Geometric Neuron's lagged **Chiral Eye** reads the arrow and is blind "
        "to parity; Kaiku's external **homodyne** carrier reads parity and is blind "
        "to the arrow. **Do they compose?** Yes -- give the readout both references "
        "and it reads both sectors at once, with zero interference. This app is that "
        "lattice, live.")
    with gr.Tab("The lattice"):
        gr.Plot(value=lattice_figure())
        gr.Markdown(
            "Hidden-arrow / entropic medium, registered. **auto** blind to both "
            "(P=0.48, D=0.61); **lag** reads the arrow only (D=1.00, P=0.53); "
            "**homodyne** reads parity only (P=1.00, D=0.46); **both** reads both "
            "(1.00 / 1.00). Composition drop: **0.000** on each -- the references "
            "help, never fight. Linear-medium control: parity unreadable everywhere "
            "(no four-wave mixing), arrow still lag-readable at 1.00.")
    with gr.Tab("Fire a field"):
        with gr.Row():
            b1 = gr.Checkbox(label="bit 1", value=True)
            b2 = gr.Checkbox(label="bit 2", value=False)
            b3 = gr.Checkbox(label="bit 3", value=True)
            dleft = gr.Checkbox(label="arrow = left", value=False)
        btn = gr.Button("Fire")
        plot = gr.Plot()
        msg = gr.Markdown()
        btn.click(run_field, [b1, b2, b3, dleft], [plot, msg])

if __name__ == "__main__":
    demo.launch()
