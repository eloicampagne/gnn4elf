from abc import ABC, abstractmethod
import os
from typing import Literal, Optional
import torch
from importlib import import_module
from typing import TypeVar
from collections.abc import Callable
from dataclasses import fields
from functools import partial
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

class PretrainedModel(ABC):
    """
    Classe de base abstraite pour tous les modèles pré-entraînés.
    Utilise un système de registre pour trouver les classes de modèle par leur nom.
    """
    REGISTRY: dict[str, "PretrainedModel"] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Méthode magique appelée quand une classe hérite de PretrainedModel.
        Elle enregistre la nouvelle classe de modèle dans le REGISTRY.
        """
        super().__init_subclass__(**kwargs)
        # On vérifie que la méthode register_name est bien implémentée
        if hasattr(cls, 'register_name'):
            model_name = cls.register_name()
            if model_name:
                print(f"--- INFO: Enregistrement du modèle '{model_name}' dans le registre. ---")
                cls.REGISTRY[model_name] = cls

    @classmethod
    @abstractmethod
    def register_name(cls) -> str:
        """
        Doit retourner le nom unique sous lequel le modèle est enregistré.
        Exemple: "TiRex"
        """
        pass

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Hook optionnel appelé après le chargement d'un checkpoint, mais avant
        le chargement du state_dict. Utile pour modifier les clés du state_dict.
        """
        pass


T = TypeVar("T", bound="PretrainedModel")
VERSION_DELIMITER = "-"

def create_incrementing_folder(base_path):
    """
    Creates a folder at the given base_path. If the folder already exists,
    it appends an incrementing number (e.g., _1, _2) until a unique name is found.

    Args:
        base_path (str): The desired path for the folder (e.g., "results/run").

    Returns:
        str: The actual path of the folder that was created.
    """
    # If the base path itself doesn't exist, create it and return
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Folder created: {base_path}")
        return base_path

    # If the base path exists, start incrementing
    counter = 1
    while True:
        # Generate the new path with the counter
        incremented_path = f"{base_path}_{counter}"
        
        # Check if this new path exists
        if not os.path.exists(incremented_path):
            # If not, create it and we're done
            os.makedirs(incremented_path)
            print(f"Folder created: {incremented_path}")
            return incremented_path
        
        # If it does exist, increment the counter and the loop will try the next number
        counter += 1

def plot_forecast(ctx, quantile_fc, real_future_values=None):
    median_forecast = quantile_fc[:, 4].numpy()
    lower_bound = quantile_fc[:, 0].numpy()
    upper_bound = quantile_fc[:, 8].numpy()

    original_x = range(len(ctx))
    forecast_x = range(len(ctx), len(ctx) + len(median_forecast))

    plt.figure(figsize=(12, 6))
    plt.plot(original_x, ctx, label="Ground Truth Context", color="#4a90d9")
    if real_future_values is not None:
        original_fut_x = range(len(ctx), len(ctx) + len(real_future_values))
        plt.plot(original_fut_x, real_future_values, label="Ground Truth Future", color="#4a90d9", linestyle=":")
    plt.plot(forecast_x, median_forecast, label="Forecast (Median)", color="#d94e4e", linestyle="--")
    plt.fill_between(
        forecast_x, lower_bound, upper_bound, color="#d94e4e", alpha=0.1, label="Forecast 10% - 90% Quantiles"
    )
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_forecast_2(forecast, real_future_values):

#     plt.figure(figsize=(12, 6))
#     plt.plot(original_fut_x, real_future_values, label="Ground Truth Future", color="#4a90d9", linestyle=":")
#     plt.plot(forecast_x, median_forecast, label="Forecast (Median)", color="#d94e4e", linestyle="--")
#     plt.fill_between(
#         forecast_x, lower_bound, upper_bound, color="#d94e4e", alpha=0.1, label="Forecast 10% - 90% Quantiles"
#     )
#     plt.xlim(left=0)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def frequency_resample(
    ts: torch.Tensor,
    prediction_length: int,
    patch_size: int = 64,
    peak_prominence: float = 0.1,
    selection_method: Literal["low_harmonic", "high_harmonic", "highest_amplitude"] = "low_harmonic",
    min_period: int | None = None,
    max_period: int = 1000,
    bandpass_filter: bool = True,
    nifr_enabled: bool = True,
    nifr_start_integer: int = 2,
    nifr_end_integer: int = 12,
    nifr_clamp_large_factors: bool = False,
) -> tuple[torch.Tensor, float, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Downsample a time series according to a frequency-based strategy and return a helper to upsample
    forecasts back to the original resolution.

    Parameters
    ----------
    ts : torch.Tensor
        1D time series of shape [T].
    prediction_length : int
        Requested forecast horizon; short horizons (<100) skip resampling.
    patch_size : int, default 64
        Nominal patch size used to align one dominant period to one patch.
    peak_prominence : float, default 0.1
        Threshold for FFT peak detection on the normalized spectrum.
    selection_method : {"low_harmonic", "high_harmonic", "highest_amplitude"}, default "low_harmonic"
        How to resolve two dominant peaks in ~2x harmonic relation.
    min_period : int or None, optional
        Minimum period to consider; if None, defaults to `patch_size`.
    max_period : int, default 1000
        Maximum period to consider for FFT peak search.
    bandpass_filter : bool, default True
        If True, suppresses very low frequencies before peak search.
    nifr_enabled : bool, default True
        Enable nearest-integer-fraction rounding of the factor.
    nifr_start_integer : int, default 2
        Smallest integer k used for 1/k grid when NIFR is enabled.
    nifr_end_integer : int, default 12
        Largest integer k used for 1/k grid when NIFR is enabled.
    nifr_clamp_large_factors : bool, default False
        If True, clamps large factors in [1, 1/nifr_start_integer] to 1.0.

    Returns
    -------
    resampled_ts : torch.Tensor
        The resampled input series.
    sample_factor : float
        Applied sampling factor (<= 1 means downsampling; 1.0 = identity).
    fc_resample_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that upsamples a forecast back to the original resolution using the inverse factor.

    Notes
    -----
    - For short horizons (prediction_length < 100), resampling is disabled and the factor is set to 1.0.
    - The factor is clamped to at most 1.0 to avoid upsampling the context.
    """
    sample_factor = frequency_factor(
        ts,
        max_period=max_period,
        min_period=min_period,
        bandpass_filter=bandpass_filter,
        selection_method=selection_method,
        peak_prominence=peak_prominence,
        patch_size=patch_size,
        nifr_enabled=nifr_enabled,
        nifr_start_integer=nifr_start_integer,
        nifr_end_integer=nifr_end_integer,
        nifr_clamp_large_factors=nifr_clamp_large_factors,
    )

    sample_factor = min(1, sample_factor)

    if prediction_length < 100:
        # do not resample for short forecasts
        sample_factor = 1.0

    fc_resample_factor = 1 / sample_factor
    fc_resample_fn = partial(resample, sample_rate=fc_resample_factor)
    resampled_ts = resample(ts, sample_rate=sample_factor)

    return resampled_ts, sample_factor, fc_resample_fn


def frequency_factor(
    ts: torch.Tensor,
    patch_size: int = 64,  # This doesn't have to match model patch size, but rather the 'target frequency'
    peak_prominence: float = 0.1,
    selection_method: Literal["low_harmonic", "high_harmonic", "highest_amplitude"] = "low_harmonic",
    min_period: int | None = None,
    max_period: int = 1000,
    bandpass_filter: bool = True,
    nifr_enabled: bool = False,
    nifr_start_integer: int = 2,
    nifr_end_integer: int = 12,
    nifr_clamp_large_factors: bool = False,
) -> float:
    """
    Estimate a sampling factor from the dominant frequency of a 1D series so that one period
    approximately fits into one patch of length `patch_size`.

    The factor is computed as `patch_size / period`, where `period = 1 / f*` and `f*`
    is the selected dominant frequency from the one-sided FFT of the series (NaNs are
    linearly interpolated for analysis). If two prominent peaks are detected whose
    frequencies are in roughly a 2x harmonic relation (ratio in [1.5, 2.5]),
    `selection_method` determines whether to select the lower or higher harmonic. A set of
    guards returns 1.0 (identity) for short series, invalid/non-finite results, or when no
    prominent peak is found. Optional nearest-integer-fraction rounding (NIFR) can snap the
    factor to the closest value in {1} ∪ {1/k | k ∈ [nifr_start_integer, nifr_end_integer]}.

    Parameters
    ----------
    ts : torch.Tensor
        Input 1D series (last dim is time). NaNs are linearly interpolated for FFT analysis only;
        the original series is not modified.
    patch_size : int, default 64
        Target number of samples per period.
    peak_prominence : float, default 0.1
        Minimum normalized spectrum height to treat a bin as a peak.
    selection_method : {"low_harmonic", "high_harmonic", "highest_amplitude"}, default "low_harmonic"
        Rule for picking between two ~2x related peaks.
    min_period : int or None, optional
        Minimum period to consider. If None, defaults to `patch_size`.
    max_period : int, default 1000
        Series shorter than `2 * max_period` return 1.0.
    bandpass_filter : bool, default True
        If True, very low frequencies below 1 / max_period are suppressed.
    nifr_enabled : bool, default False
        Enable nearest-integer-fraction rounding of the factor.
    nifr_start_integer : int, default 2
        Smallest integer k used for the 1/k grid when NIFR is enabled.
    nifr_end_integer : int, default 12
        Largest integer k used for the 1/k grid when NIFR is enabled.
    nifr_clamp_large_factors : bool, default False
        If True, clamps factors in [1, 1/nifr_start_integer] to 1.0.

    Returns
    -------
    float
        The sampling factor. Values <= 0 or non-finite are mapped to 1.0. If no valid
        dominant frequency is found or the series is too short, returns 1.0.

    Notes
    -----
    - The factor is computed as `patch_size / period`, where `period = 1 / f*` and `f*` is the selected
      dominant FFT frequency.
    - If two prominent peaks are detected ~2x apart, `selection_method` determines whether to select the lower or higher harmonic.
    - Optional nearest-integer-fraction rounding (NIFR) can snap the factor to the closest value in {1} ∪ {1/k | k ∈ [nifr_start_integer, nifr_end_integer]}.
    """
    if min_period is None:
        # NOTE: be careful when min_period is not matching patch_size, it can create unexpected scaling factors!
        min_period = patch_size

    # Ensure CPU numpy array for FFT analysis
    ts_np = ts.detach().cpu().numpy() if isinstance(ts, torch.Tensor) else np.asarray(ts)

    # NOTE: If the series is shorter than max_period *2, FFT may not be accurate, to avoid detecting these peaks, we don't scale
    if ts_np.size < max_period * 2:
        return 1.0

    freqs, specs, peak_idc = run_fft_analysis(
        ts_np,
        scaling="amplitude",
        peak_prominence=peak_prominence,
        min_period=min_period,
        max_period=max_period,
        bandpass_filter=bandpass_filter,
    )

    # No detectable peaks -> keep original sampling
    if peak_idc.size == 0:
        return 1.0

    # Choose initial candidate as the highest-amplitude peak
    chosen_idx = int(peak_idc[0])

    # If two peaks exist, check for ~2x harmonic relation and prefer the higher/lower one
    if peak_idc.size >= 2:
        idx_a = int(peak_idc[0])  # highest amplitude
        idx_b = int(peak_idc[1])  # second highest amplitude
        f_a = float(freqs[idx_a])
        f_b = float(freqs[idx_b])

        # Determine lower/higher frequency
        low_f = min(f_a, f_b)
        high_f = max(f_a, f_b)

        if low_f > 0:
            ratio = high_f / low_f
            # Roughly half relation
            if 1.5 <= ratio <= 2.5:
                if selection_method == "low_harmonic":
                    chosen_idx = idx_a if f_a < f_b else idx_b
                elif selection_method == "high_harmonic":
                    chosen_idx = idx_a if f_a > f_b else idx_b

    chosen_freq = float(freqs[chosen_idx])

    # Guard against zero or non-finite frequency
    if not np.isfinite(chosen_freq) or chosen_freq <= 0:
        return 1.0

    # Convert to period and compute scaling factor so one period fits one patch
    period = 1.0 / chosen_freq
    factor = resampling_factor(period, patch_size)
    factor = round(factor, 4)

    # Guard against factor being negative
    if not np.isfinite(factor) or factor <= 0:
        return 1.0

    # nearest interger fraction rounding (nifr)
    if nifr_enabled:
        int_fractions = np.concatenate([[1], 1 / np.arange(nifr_start_integer, nifr_end_integer + 1)])
        diff = np.abs(factor - int_fractions)
        min_diff_idc = np.argmin(diff)
        factor = int_fractions[min_diff_idc]

        if nifr_clamp_large_factors:
            # Clamp everything between 1 and 1/nifr_start_integer to 1, that is no scaling
            factor = factor if factor < int_fractions[1] else 1

    return float(factor)


def resample(ts: torch.Tensor, sample_rate: float, window_position: str = "center") -> torch.Tensor:
    """
    Resample the time series using NaN-tolerant window averaging with size 1/sample_rate.

    - If sample_rate > 1 the series is upsampled; windows may collapse to a single index.
    - If sample_rate < 1 the series is downsampled; windows span multiple indices.
    - If sample_rate == 1 the series is returned unchanged (cast to float for NaN support).

    Window alignment controlled by `window_position`:
    - "center": average over [c - L/2, c + L/2]
    - "left"  : average over [c - L,   c]
    - "right" : average over [c,       c + L]

    The window is truncated at the boundaries. NaNs are ignored via nan-mean semantics;
    if a window contains only NaNs, the output is NaN.

    Arguments:
    ----------
    ts: torch.Tensor of shape [..., T]
        The time series to be rescaled (last dim is time).
    sample_rate: float
        The factor determining the final number of timesteps in the series, i.e., T' = ceil(T * sample_rate).
    window_position: {"center", "left", "right"}
        Placement of the averaging window relative to each target coordinate.

    Returns:
    --------
    torch.Tensor of shape [..., ceil(T * sample_rate)] with dtype float.
    """
    # Validate inputs
    if sample_rate <= 0 or sample_rate == 1:
        # Invalid or no scaling; return original as float
        return ts.to(torch.float)

    src_num_timesteps = ts.shape[-1]
    tgt_num_timesteps = ceil(src_num_timesteps * sample_rate)

    # Do not change coordinate creation logic
    src_coords = torch.arange(src_num_timesteps, device=ts.device)
    tgt_coords = torch.linspace(0, src_num_timesteps - 1, tgt_num_timesteps, device=ts.device)

    if sample_rate == 1:
        return ts.to(torch.float)

    # Branch: upsampling -> linear interpolation between nearest neighbors (NaN-aware)
    if sample_rate > 1:
        # Neighbour indices for each target coordinate along the last dimension
        tgt_in_src_idx_lo = tgt_coords.floor().to(torch.long)
        tgt_in_src_idx_hi = tgt_coords.ceil().to(torch.long)

        # Distances in index space and offsets from lower index
        dist = src_coords[tgt_in_src_idx_hi] - src_coords[tgt_in_src_idx_lo]

        # Work in float for NaN support; gather neighbour values
        src_lo_vals = ts[..., tgt_in_src_idx_lo].to(torch.float)
        src_hi_vals = ts[..., tgt_in_src_idx_hi].to(torch.float)
        diff = src_hi_vals - src_lo_vals
        offset = tgt_coords - src_coords[tgt_in_src_idx_lo]

        # Allocate output
        tgt_values = torch.empty(*ts.shape[:-1], tgt_num_timesteps, dtype=torch.float, device=ts.device)

        # Masks
        exact_mask = dist == 0
        interp_mask = ~exact_mask

        # Exact source index -> take the source value
        if exact_mask.any():
            tgt_values[..., exact_mask] = src_lo_vals[..., exact_mask]

        # Linear interpolate where indices differ
        if interp_mask.any():
            tgt_values[..., interp_mask] = (
                diff[..., interp_mask] / dist[interp_mask].to(torch.float) * offset[interp_mask]
                + src_lo_vals[..., interp_mask]
            )

        # Propagate NaNs from either neighbour
        nan_mask = torch.isnan(src_lo_vals) | torch.isnan(src_hi_vals)
        if nan_mask.any():
            tgt_values[..., nan_mask] = torch.nan

        return tgt_values

    # Window length in source-index units
    L = 1.0 / sample_rate
    half_L = 0.5 * L

    if window_position == "center":
        left_f = tgt_coords - half_L
        right_f = tgt_coords + half_L
    elif window_position == "left":
        left_f = tgt_coords - L
        right_f = tgt_coords
    elif window_position == "right":
        left_f = tgt_coords
        right_f = tgt_coords + L
    else:
        raise ValueError("window_position must be one of {'center','left','right'}")

    # Convert to integer indices, inclusive bounds
    left_idx = torch.ceil(left_f).to(torch.long)
    right_idx = torch.floor(right_f).to(torch.long)

    # Clip to valid range and ensure non-empty windows (at least one index)
    left_idx = torch.clamp(left_idx, 0, src_num_timesteps - 1)
    right_idx = torch.clamp(right_idx, 0, src_num_timesteps - 1)
    right_idx = torch.maximum(right_idx, left_idx)

    # Prepare cumulative sums for fast [l, r] segment nan-mean along the last dim
    ts_float = ts.to(torch.float)
    valid_mask = ~torch.isnan(ts_float)

    values_filled = torch.where(valid_mask, ts_float, torch.zeros_like(ts_float))
    counts = valid_mask.to(torch.float)

    cumsum_vals = values_filled.cumsum(dim=-1)
    cumsum_cnts = counts.cumsum(dim=-1)

    # Pad a leading zero to make inclusive range sums easy: sum[l:r] = cs[r] - cs[l-1]
    pad_shape = (*ts.shape[:-1], 1)
    zeros_vals = torch.zeros(pad_shape, dtype=cumsum_vals.dtype, device=ts.device)
    zeros_cnts = torch.zeros(pad_shape, dtype=cumsum_cnts.dtype, device=ts.device)
    cumsum_vals = torch.cat([zeros_vals, cumsum_vals], dim=-1)
    cumsum_cnts = torch.cat([zeros_cnts, cumsum_cnts], dim=-1)

    # Build broadcastable indices for gather along the last dim
    prefix_shape = ts.shape[:-1]
    target_len = tgt_num_timesteps

    def _expand_index(idx: torch.Tensor) -> torch.Tensor:
        # idx shape [target_len] -> [..., target_len]
        view_shape = (1,) * len(prefix_shape) + (target_len,)
        return idx.view(view_shape).expand(*prefix_shape, target_len)

    # For inclusive [l, r], use cumsum at (r+1) and (l)
    r_plus1 = torch.clamp(right_idx + 1, 0, src_num_timesteps)
    l_idx = left_idx

    r_plus1_exp = _expand_index(r_plus1)
    l_exp = _expand_index(l_idx)

    seg_sums = cumsum_vals.gather(dim=-1, index=r_plus1_exp) - cumsum_vals.gather(dim=-1, index=l_exp)
    seg_cnts = cumsum_cnts.gather(dim=-1, index=r_plus1_exp) - cumsum_cnts.gather(dim=-1, index=l_exp)

    # Compute nan-mean: where count==0 -> NaN
    with torch.no_grad():
        safe_cnts = torch.where(seg_cnts > 0, seg_cnts, torch.ones_like(seg_cnts))
    averages = seg_sums / safe_cnts
    averages = torch.where(seg_cnts > 0, averages, torch.full_like(averages, float("nan")))

    return averages


def run_fft_analysis(
    y,
    dt: float = 1.0,
    window: str = "hann",
    detrend: bool = True,
    scaling: str = "amplitude",
    peak_prominence: float = 0.1,
    min_period: int = 64,
    max_period: int = 1000,
    bandpass_filter: bool = True,
):
    """
    Compute one-sided FFT frequencies and spectrum magnitude for a real 1D signal.

    Parameters
    ----------
    y : array_like
        1D time series (regularly sampled). NaNs will be linearly interpolated.
    dt : float
        Sampling period (time between samples). Frequencies are cycles per unit of dt.
    window : {'hann', None}
        Optional taper to reduce leakage.
    detrend : bool
        If True, remove the mean before FFT.
    scaling : {'amplitude', 'power', 'raw'}
        - 'amplitude': one-sided amplitude spectrum with window-power compensation.
        - 'power'    : one-sided power (not density) with window-power compensation.
        - 'raw'      : |rfft(yw)| (no normalization, mostly for debugging).
    peak_prominence : float
        Absolute threshold on the normalized spectrum for peak detection.

    Returns
    -------
    f : ndarray
        Frequencies (non-negative), length N//2 + 1, in cycles per unit (1/dt).
    spec : ndarray
        Spectrum corresponding to `f` under the chosen `scaling`.
    peaks_idx : ndarray
        Indices into f of detected peaks.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)
    n = y.size
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    # Fill NaNs linearly (handles edge NaNs as well)
    y = _nan_linear_interpolate(y)

    if detrend:
        y = y - np.mean(y)

    # Windowing
    if window == "hann":
        w = np.hanning(n)
        yw = y * w
        # average window power (for proper amplitude/power normalization)
        w_power = np.sum(w**2) / n
    elif window is None:
        yw = y
        w_power = 1.0
    else:
        raise ValueError("window must be either 'hann' or None")

    # FFT (one-sided)
    Y = np.fft.rfft(yw)
    f = np.fft.rfftfreq(n, d=dt)  # cycles per unit time

    if scaling == "raw":
        spec = np.abs(Y)
    elif scaling == "amplitude":
        # One-sided amplitude with window power compensation
        spec = np.abs(Y) / (n * np.sqrt(w_power))
        if n % 2 == 0:
            spec[1:-1] *= 2.0
        else:
            spec[1:] *= 2.0
    elif scaling == "power":
        # One-sided power (not PSD)
        spec = (np.abs(Y) ** 2) / (n**2 * w_power)
        if n % 2 == 0:
            spec[1:-1] *= 2.0
        else:
            spec[1:] *= 2.0
    else:
        raise ValueError("scaling must be 'amplitude', 'power', or 'raw'")

    # Normalize the spectrum by its maximum value
    if spec.max() > 0:
        spec = spec / spec.max()

    # Find peaks in the spectrum
    peaks_idx = custom_find_peaks(
        f,
        spec,
        max_peaks=2,
        prominence_threshold=peak_prominence,
        min_period=min_period,
        max_period=max_period,
        bandpass_filter=bandpass_filter,
    )

    return f, spec, peaks_idx


def _nan_linear_interpolate(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    if y.ndim != 1:
        y = y.reshape(-1)
    n = y.size
    mask = np.isfinite(y)
    if mask.all():
        return y
    if (~mask).all():
        return np.zeros(n, dtype=np.float32)
    idx = np.arange(n)
    y_interp = y.copy()
    y_interp[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return y_interp


def resampling_factor(inverted_freq, path_size):
    """
    Compute the resampling factor based on the inverted frequency and path size.
    """
    if inverted_freq <= 0:
        return 1.0
    factor = path_size / inverted_freq
    return factor


def custom_find_peaks(
    f,
    spec,
    *,
    max_peaks=5,
    prominence_threshold=0.1,
    min_period=64,
    max_period=1000,
    bandpass_filter=True,
):
    """
    Finds prominent peaks in a spectrum using a simple custom logic.

    A peak is a local maximum. A peak is considered prominent if its height
    (on a normalized spectrum) is greater than a given threshold.

    Parameters
    ----------
    f : np.ndarray
        Frequency array (currently unused but kept for API consistency).
    spec : np.ndarray
        The normalized spectrum.
    max_peaks : int
        The maximum number of peaks to return.
    prominence_threshold : float
        The minimum height for a peak to be considered prominent.

    Returns
    -------
    np.ndarray
        An array of indices of the detected peaks in the spectrum. Returns an
        empty array if no prominent peaks are found.
    """
    if len(spec) < 5:  # Need at least 5 points to exclude last two bins
        return np.array([], dtype=int)

    if bandpass_filter:  # only truly filter low frequencies, high frequencies are dealt with later
        min_freq = 1 / max_period
        freq_mask = f >= min_freq
        spec = spec * freq_mask

    # Find all local maxima, excluding the last two bins
    local_maxima_indices = []
    for i in range(1, len(spec) - 2):
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1]:
            local_maxima_indices.append(i)

    if not local_maxima_indices:
        return np.array([], dtype=int)

    # Filter by prominence (height)
    prominent_peaks = []
    for idx in local_maxima_indices:
        if spec[idx] > prominence_threshold:
            prominent_peaks.append((idx, spec[idx]))

    # If no peaks are above the threshold, return an empty list
    if not prominent_peaks:
        return np.array([], dtype=int)

    # Check for clear peaks below min_period (do lowpass filter)
    for idx, _ in prominent_peaks:
        period = 1 / f[idx]
        if period < min_period:
            return np.array([], dtype=int)

    # Filter by period
    period_filtered_peaks = []
    for idx, prominence in prominent_peaks:
        period = 1 / f[idx]

        if min_period <= period <= max_period:
            period_filtered_peaks.append((idx, prominence))

    if not period_filtered_peaks:
        return np.array([], dtype=int)

    # Sort by height and return the top `max_peaks`
    period_filtered_peaks.sort(key=lambda x: x[1], reverse=True)
    peak_indices = np.array([p[0] for p in period_filtered_peaks[:max_peaks]], dtype=int)

    return peak_indices


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def dataclass_from_dict(cls, dict: dict):
    class_fields = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict.items() if k in class_fields})

def skip_cuda():
    return os.getenv("TIREX_NO_CUDA", "False").lower() in ("true", "1", "t")


def parse_hf_repo_id(path):
    parts = path.split("/")
    return "/".join(parts[0:2])

def parse_model_string(model_string):
    if VERSION_DELIMITER in model_string:
        parts = model_string.split(VERSION_DELIMITER)
        model_id, version = parts[0], parts[0]
    else:
        model_id = model_string
        version = None

    return model_id, version

def load_tirex_from_checkpoint(
    checkpoint_path: str,
    model_id: str,
    device: str | None = None,
    backend: Literal["torch", "cuda"] = "torch",
) -> PretrainedModel:
    """
    Charge un modèle TiRex depuis un fichier de checkpoint (.ckpt) local.

    Args:
        checkpoint_path (str): Le chemin vers votre fichier 'tirex.ckpt'.
        model_id (str): L'identifiant du modèle à charger (ex: 'TiRex').
        device (str, optional): Le périphérique sur lequel charger le modèle ('cpu', 'cuda:0', etc.).
                                Si None, est détecté automatiquement.
        backend (Literal["torch", "cuda"], optional): Le backend à utiliser.

    Returns:
        Le modèle PyTorch chargé, prêt à être utilisé.
    """
    print("--- Début du chargement du modèle local. ---")

    # 1. Vérifier que le fichier existe
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fichier de checkpoint non trouvé : {checkpoint_path}")

    # 2. S'assurer que le module du modèle est importé pour remplir le registre
    #    Cette ligne est la clé ! Elle exécute tirex/models/tirex.py
    try:
        print(f"Tentative d'importation du module pour le modèle ID: '{model_id}'...")
        # On suppose que le fichier s'appelle comme le model_id en minuscules.
        # ex: 'TiRex' -> 'tirex.models.tirex'
        import_module(f"tirex.models.{model_id.lower()}")
    except ImportError as e:
        raise ImportError(
            f"Impossible d'importer le module du modèle 'tirex.models.{model_id.lower()}'. "
            f"Vérifiez que le fichier existe et que le nom correspond. Erreur: {e}"
        )

    # 3. Trouver la classe du modèle dans le registre maintenant rempli
    print(f"Recherche de '{model_id}' dans le registre. IDs disponibles: {list(PretrainedModel.REGISTRY.keys())}")
    model_cls = PretrainedModel.REGISTRY.get(model_id)
    if model_cls is None:
        raise ValueError(
            f"Modèle ID '{model_id}' non trouvé dans le registre après import. "
            "Vérifiez que la classe dans le fichier modèle a bien `register_name()` qui retourne cette valeur."
        )
    print(f"Classe du modèle trouvée : {model_cls.__name__}")

    # 4. Déterminer le périphérique
    if device is None:
        device = "cuda:0" if backend == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle sur le périphérique : {device}")

    # 5. Charger le fichier checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError("Le checkpoint est invalide. Il doit contenir les clés 'hyper_parameters' et 'state_dict'.")

    # 6. Instancier le modèle avec sa configuration
    hyper_params = checkpoint["hyper_parameters"]
    print(f"Hyper-paramètres trouvés dans le checkpoint : {hyper_params.keys()}")
    model = model_cls(backend=backend, **hyper_params)

    # 7. Appliquer les modifications de clés si nécessaire
    model.on_load_checkpoint(checkpoint)

    # 8. Charger les poids dans le modèle instancié
    model.load_state_dict(checkpoint["state_dict"])
    print("Poids du modèle chargés avec succès.")

    # 9. Mettre le modèle sur le bon périphérique et en mode évaluation
    model.to(device)
    model.eval()

    print(f"--- Modèle '{model_id}' chargé et prêt à l'emploi ! ---")
    return model
