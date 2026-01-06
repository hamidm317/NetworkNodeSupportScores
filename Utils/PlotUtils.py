import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from scipy.interpolate import griddata
import scipy.stats as sps

def plot_scalp_topography(
    ax,
    ch_xy,
    ch_values,
    vmin=None,
    vmax=None,
    cmap="viridis",
    show_colorbar=False,
    n_grid=200,
    head_radius=0.5,
    nose=True,
    ears=True,
    shading="gouraud",
):
    """
    Plot EEG scalp topography inside a circular head outline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on. The function will also return it.
    ch_xy : array-like, shape (n_channels, 2)
        2D positions for each channel in Cartesian coords (x, y), roughly within a head circle.
        Coordinates should be in the same units; typical range ~[-0.5, 0.5] if head_radius=0.5.
    ch_values : array-like, shape (n_channels,)
        Scalar value per channel to plot.
    vmin, vmax : float or None
        Color scale limits. If None, computed from ch_values (ignoring NaNs).
    cmap : str or Colormap
        Matplotlib colormap.
    show_colorbar : bool
        If True, attach a colorbar to the figure.
    n_grid : int
        Resolution of interpolation grid (n_grid x n_grid).
    head_radius : float
        Radius of the head circle used for masking and outline.
    nose, ears : bool
        If True, draw simple nose and ears glyphs.
    shading : {"gouraud","nearest","auto","flat"}
        Shading for pcolormesh-like smoothness. Only used if fall back to pcolormesh.
        For griddata with regular grid, we’ll use imshow for speed and smoothness.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used for plotting.
    im : matplotlib.image.AxesImage
        The image object returned by imshow (useful for colorbar).
    cbar : matplotlib.colorbar.Colorbar or None
        The colorbar if show_colorbar=True, else None.

    Notes
    -----
    - Interpolation is done with scipy.interpolate.griddata (linear).
    - Values outside the head circle are masked out to keep a crisp circular boundary.
    - Provide channel positions already projected to 2D (e.g., top-down).
    """
    ch_xy = np.asarray(ch_xy, dtype=float)
    ch_values = np.asarray(ch_values, dtype=float)

    # Color limits
    if vmin is None or vmax is None:
        finite_vals = ch_values[np.isfinite(ch_values)]
        if vmin is None:
            vmin = np.nanmin(finite_vals) if finite_vals.size else 0.0
        if vmax is None:
            vmax = np.nanmax(finite_vals) if finite_vals.size else 1.0
        if vmin == vmax:
            # avoid zero dynamic range
            vmin, vmax = vmin - 1e-9, vmax + 1e-9

    # Regular grid inside bounding square
    r = float(head_radius)
    x = np.linspace(-r, r, n_grid)
    y = np.linspace(-r, r, n_grid)
    X, Y = np.meshgrid(x, y)

    # Interpolate onto grid
    Zi = griddata(points=ch_xy, values=ch_values, xi=(X, Y), method="linear")

    # Mask outside head circle
    mask = (X**2 + Y**2) <= (r**2)
    Zi_masked = np.ma.array(Zi, mask=~mask)

    # Plot with imshow for smooth continuous look (set extent to match coordinates)
    im = ax.imshow(
        Zi_masked,
        extent=(-r, r, -r, r),
        origin="lower",
        interpolation="bicubic",  # smooth edges
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    # Draw head outline
    head = Circle((0, 0), r, edgecolor="k", facecolor="none", linewidth=1.5)
    ax.add_patch(head)

    # Optionally draw nose (simple triangle pointing +Y)
    if nose:
        nose_w = 0.08 * r
        nose_h = 0.14 * r
        nose_y = r
        nose_pts = np.array([
            [0.0, nose_y + nose_h],
            [-nose_w, nose_y],
            [ nose_w, nose_y],
        ])
        ax.add_patch(Polygon(nose_pts, closed=True, edgecolor="k", facecolor="k", linewidth=1.0))

    # Optionally draw ears (simple arcs/ellipses)
    if ears:
        ear_w = 0.04 * r
        ear_h = 0.14 * r
        ear_y0 = 0.0
        left_ear = Polygon([
            [-r, ear_y0 - ear_h],
            [-r - ear_w, ear_y0],
            [-r, ear_y0 + ear_h]
        ], closed=False, edgecolor="k", linewidth=1.2)
        right_ear = Polygon([
            [ r, ear_y0 - ear_h],
            [ r + ear_w, ear_y0],
            [ r, ear_y0 + ear_h]
        ], closed=False, edgecolor="k", linewidth=1.2)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)

    # Tidy axes
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    return ax, im, cbar

def mean_confidence_interval(data_matrix: np.ndarray, CoLe: float = 0.95):
    """
    Compute mean and confidence interval along the first axis of a NumPy array.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Input data with shape (Pop, ...). First axis = population axis.
    CoLe : float, default 0.95
        Confidence level (0 < CoLe < 1).
    
    Returns
    -------
    mean : np.ndarray
        Mean over the Pop axis, shape = data_matrix.shape[1:].
    y_U : np.ndarray
        Upper bound of confidence interval, same shape as mean.
    y_L : np.ndarray
        Lower bound of confidence interval, same shape as mean.
    """
    if not (0 < CoLe < 1):
        raise ValueError("Confidence level must be between 0 and 1.")
    
    n = data_matrix.shape[0]
    mean = np.mean(data_matrix, axis=0)
    std_err = np.std(data_matrix, axis=0, ddof=1) / np.sqrt(n)  # SEM
    
    # t critical value for two-tailed interval
    t_crit = sps.t.ppf((1 + CoLe) / 2.0, df=n-1)
    
    margin = t_crit * std_err
    y_U = mean + margin
    y_L = mean - margin
    
    return mean, y_U, y_L

import numpy as np
from scipy import stats as sps
from typing import Tuple, Literal

def distribution_bound(
    data_matrix: np.ndarray,
    coverage: float = 80.0,
    assumption: Literal['normal', 'empirical'] = 'normal',
    axis: int = 0,
    nan_policy: Literal['omit','propagate'] = 'omit',
    ddof: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute central band (upper/lower) that contains `coverage` percent of the distribution
    along `axis`. Two modes:
      - assumption='normal'   : uses mean ± z * std  (z from equal tails)
      - assumption='empirical': uses equal-tail empirical quantiles (q_low, q_high),
                                and returns the median as center.

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data. Population axis is given by `axis` (default 0).
    coverage : float, default 80.0
        Desired central coverage in percent (0 < coverage < 100),
        e.g., 80 → central 10–90% band.
    assumption : {'normal','empirical'}, default 'normal'
        'normal'   → μ ± z·σ
        'empirical'→ [Q_{(1-c)/2}, Q_{1-(1-c)/2}], center=median
    axis : int, default 0
        Axis to aggregate over (population axis).
    nan_policy : {'omit','propagate'}, default 'omit'
        If 'omit', ignore NaNs in statistics; if 'propagate', NaNs propagate.
    ddof : int, default 1
        Degrees of freedom for std in 'normal' mode.

    Returns
    -------
    center : np.ndarray
        Center of the band: mean (normal) or median (empirical),
        shape = data_matrix.shape with the `axis` removed.
    y_U : np.ndarray
        Upper bound of the central band, same shape as center.
    y_L : np.ndarray
        Lower bound of the central band, same shape as center.

    Notes
    -----
    - Equal-tail means tails trimmed equally in probability (e.g., 10% پایین + 10% بالا).
    - For plotting whiskers (e.g., 5–95%), set coverage=90.
    """
    if not (0.0 < coverage < 100.0):
        raise ValueError("coverage must be in (0, 100).")
    c = coverage / 100.0
    q_low = (1.0 - c) / 2.0
    q_high = 1.0 - q_low

    x = np.asarray(data_matrix)

    # Nan-safe reducers
    if nan_policy == 'omit':
        reducer_mean = np.nanmean
        reducer_std  = lambda a, axis: np.nanstd(a, axis=axis, ddof=ddof)
        reducer_med  = np.nanmedian
        quantile_fn  = np.nanquantile
    elif nan_policy == 'propagate':
        reducer_mean = lambda a, axis: np.mean(a, axis=axis)
        reducer_std  = lambda a, axis: np.std(a, axis=axis, ddof=ddof)
        reducer_med  = lambda a, axis: np.median(a, axis=axis)
        quantile_fn  = lambda a, q, axis: np.quantile(a, q, axis=axis)
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    if assumption == 'normal':
        center = reducer_mean(x, axis=axis)
        std = reducer_std(x, axis=axis)
        z = sps.norm.ppf(q_high)  # e.g., for 80% coverage, q_high=0.90 → z≈1.2816
        y_U = center + z * std
        y_L = center - z * std
        return center, y_U, y_L

    elif assumption == 'empirical':
        center = reducer_med(x, axis=axis)
        # Equal-tail empirical quantiles:
        y_L = quantile_fn(x, q_low, axis=axis)
        y_U = quantile_fn(x, q_high, axis=axis)
        return center, y_U, y_L

    else:
        raise ValueError("assumption must be 'normal' or 'empirical'.")
