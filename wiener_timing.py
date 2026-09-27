"""
wiener_timing
=============

Wiener deconvolution for astronomical timing analysis: estimate the time lag
and the response (transfer) function between two light curves.

This is the method described in

    Kırmızıbayrak D. & Heyl J., "Estimating Astronomical Time Lags and
    Response with Wiener Deconvolution", MNRAS (submitted)

The Wiener deconvolved response (WDR) between band 1 and band 2 is

    R(f) = F1(f) F2*(f) / ( |F1(f)|^2 + eta )

where F1, F2 are the Fourier transforms of the two light curves and eta is
the noise power to be filtered out (see `poisson_noise_level` and
`white_noise_level`). The inverse Fourier transform of R(f) gives the
response in the time domain.

Sign convention: with the ordering above, a peak at POSITIVE time means
band 1 lags behind band 2. Swap the two inputs to reverse the convention.
"""

import numpy as np

__all__ = [
    "bump_window",
    "resample_to_grid",
    "poisson_noise_level",
    "white_noise_level",
    "wiener_response",
]


def bump_window(t, k=8, t0=None, duration=None):
    """Exponentially decaying "bump" window (Eq. 7 of the paper).

        w(t) = exp[ 1 - 1 / (1 - (2 (t - t0) / T)^k) ]

    Parameters
    ----------
    t : array_like
        Time stamps.
    k : int, optional
        Even exponent setting how quickly the window falls to zero at the
        edges. Larger k gives a flatter top and steeper edges. Default 8.
    t0 : float, optional
        Mid-point of the observation. Defaults to the centre of `t`.
    duration : float, optional
        Total duration T of the observation. Defaults to the span of `t`.

    Returns
    -------
    w : ndarray
        Window values between 0 and 1 (1 at the centre, 0 at the edges).
    """
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 0.5 * (t.min() + t.max())
    if duration is None:
        duration = t.max() - t.min()

    x = (2.0 * (t - t0) / duration) ** k
    w = np.zeros_like(t)
    inside = x < 1.0
    w[inside] = np.exp(1.0 - 1.0 / (1.0 - x[inside]))
    return w


def resample_to_grid(t1, y1, t2, y2, nbins):
    """Put two light curves on a common, evenly spaced time grid.

    The grid spans only the time range covered by BOTH light curves, so no
    values are extrapolated. Choose `nbins` so that the bin size is not finer
    than the original sampling, and check that neither light curve has large
    gaps: linear interpolation across a gap invents data and would result in false detections. 

    Returns
    -------
    t, y1_grid, y2_grid : ndarray
        Common time grid and the two resampled light curves.
    """
    t_start = max(np.min(t1), np.min(t2))
    t_end = min(np.max(t1), np.max(t2))
    if t_end <= t_start:
        raise ValueError("The two light curves do not overlap in time.")
    t = np.linspace(t_start, t_end, nbins)
    return t, np.interp(t, t1, y1), np.interp(t, t2, y2)


def poisson_noise_level(counts_per_bin):
    """Noise level for Poisson counting noise (Eq. 11 / Appendix B).

    eta = N, the total number of counts in the light curve (band 1).
    `counts_per_bin` must be COUNTS per bin, not a count rate.
    """
    return float(np.sum(counts_per_bin))


def white_noise_level(sigma, nbins):
    """Noise level for white (Gaussian) noise (Eq. 11 / Appendix B).

    eta = nbins * sigma^2, where sigma^2 is the variance of the measurement
    noise per bin.
    """
    return float(nbins) * float(sigma) ** 2


def wiener_response(y1, y2, noise, dt, window_k=8, subtract_mean=True):
    """Wiener deconvolved response between two evenly sampled light curves.

    Parameters
    ----------
    y1, y2 : array_like
        Light curves of band 1 and band 2 on the same, evenly spaced grid.
    noise : float
        Noise level eta to filter out (see `poisson_noise_level`,
        `white_noise_level`, or read it off the flat, high-frequency part of
        the band-1 power spectrum).
    dt : float
        Bin size (time units of the output lag axis).
    window_k : int or None, optional
        Exponent of the bump window. Use None for no window. Default 8.
    subtract_mean : bool, optional
        Subtract the mean of each light curve (before windowing). Default True.

    Returns
    -------
    lags : ndarray
        Time axis of the response, centred on zero.
    response : ndarray
        Wiener deconvolved response in the time domain, on `lags`.
        Positive lag = band 1 lags behind band 2.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if y1.shape != y2.shape:
        raise ValueError("y1 and y2 must have the same length.")
    n = y1.size

    if subtract_mean:
        y1 = y1 - y1.mean()
        y2 = y2 - y2.mean()

    if window_k is not None:
        w = bump_window(np.arange(n), k=window_k)
        y1 = y1 * w
        y2 = y2 * w

    f1 = np.fft.rfft(y1)
    f2 = np.fft.rfft(y2)

    ps1 = np.abs(f1) ** 2              # power spectrum of band 1
    cs12 = f1 * np.conj(f2)            # cross spectrum
    resp_f = cs12 / (ps1 + noise)      # Wiener deconvolved response, freq. domain

    resp_t = np.fft.irfft(resp_f, n=n)  # response in the time domain
    response = np.fft.fftshift(resp_t)
    lags = (np.arange(n) - n // 2) * dt
    return lags, response
