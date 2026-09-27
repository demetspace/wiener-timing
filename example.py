"""
Example: recover a known time lag between two simulated light curves.

A random light curve with a QPO-like (Lorentzian) power spectrum is simulated
(Timmer & König 1995). Band 1 is a copy of band 2 delayed by `true_lag`,
Poisson noise is added to both, and the Wiener deconvolved response is
computed. The response should peak at +true_lag (band 1 lags band 2).

Run with:  python example.py
"""

import numpy as np
import matplotlib.pyplot as plt

from wiener_timing import poisson_noise_level, wiener_response

rng = np.random.default_rng(42)

# --- simulation settings (same as the toy models in Section 3.1) ---------
duration = 100.0        # s
nbins = 16 * 1024
rate = 2048.0           # counts / s
true_lag = 0.1          # s, band 1 lags band 2
dt = duration / nbins

# --- simulate an underlying light curve (Timmer & König 1995) ------------
freqs = np.fft.rfftfreq(nbins, dt)
gamma = np.sqrt(nbins) / duration       # QPO width, Eq. 9 of the paper
f0 = gamma
psd = np.zeros_like(freqs)
psd[1:] = gamma / ((freqs[1:] - f0) ** 2 + (gamma / 2) ** 2)
amp = np.sqrt(psd / 2) * (rng.normal(size=freqs.size)
                          + 1j * rng.normal(size=freqs.size))
amp[0] = 0.0

# band 2 = original signal, band 1 = the same signal delayed by true_lag
signal2 = np.fft.irfft(amp, n=nbins)
signal1 = np.fft.irfft(amp * np.exp(-2j * np.pi * freqs * true_lag), n=nbins)


def to_counts(signal):
    """Scale a signal to ~30% rms around the mean rate, then Poisson-sample."""
    mean_counts = rate * dt
    lc = mean_counts * (1 + 0.3 * signal / signal.std())
    return rng.poisson(np.clip(lc, 0, None)).astype(float)


counts1 = to_counts(signal1)
counts2 = to_counts(signal2)

# --- Wiener deconvolved response ----------------------------------------
eta = poisson_noise_level(counts1)
lags, response = wiener_response(counts1, counts2, noise=eta, dt=dt)

peak = lags[np.argmax(response)]
print(f"input lag: {true_lag:.3f} s   recovered peak: {peak:.3f} s")

# --- plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(lags, response, lw=1, label="WDR")
ax.axvline(true_lag, color="r", ls="--", lw=1, label="input lag")
ax.set_xlim(-0.5, 0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Response")
ax.legend()
fig.tight_layout()
fig.savefig("example_response.png", dpi=150)
print("saved example_response.png")
