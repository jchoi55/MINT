import pandas as pd
import matplotlib.patches as patches
import numpy as np

from scipy.interpolate import interp1d

from mint import const, plot_tools as pt
from mint.lattice_tools import advance_in_pos_and_momentum


def get_gyro_radius(E, B):
    return 3.3e2 * E / B  # cm (E in GeV, B in T)


def get_dtheta(s, R):
    return s / R


def propagate(x0, y0, px0, py0, dtheta, s):
    # r = np.sqrt(x0**2 + y0**2)
    theta_p = np.arctan2(py0, px0)
    p = np.sqrt(px0**2 + py0**2)
    pxf = p * np.cos(theta_p - dtheta)
    pyf = p * np.sin(theta_p - dtheta)

    if dtheta == 0:
        return x0 + s * np.cos(theta_p), y0 + s * np.sin(theta_p), pxf, pyf
    else:
        R = s / dtheta
        # coordinates centered around larmor circle
        x0_prime = R * np.cos(np.pi / 2 + theta_p)
        y0_prime = R * np.sin(np.pi / 2 + theta_p)

        xf_prime = R * np.cos(np.pi / 2 + theta_p - dtheta)
        yf_prime = R * np.sin(np.pi / 2 + theta_p - dtheta)

        dx = xf_prime - x0_prime
        dy = yf_prime - y0_prime

        return x0 + dx, y0 + dy, pxf, pyf


def plot_lattice(df, ax, units=1, draw_center_line=True):
    if draw_center_line:
        ax.plot(df["x"] * units, df["y"] * units, linewidth=0.5, c="black")

    # Minimum size of linear step
    ds = 0.1 * units

    # How tall is the magnet for x-y plane
    magnet_thickness = 1 * units
    n_elements = df.index.size
    ds = 0.1 * units

    for i in list(range(n_elements - 100, n_elements - 8)):
        x, y, s = df["x"][i] * units, df["y"][i] * units, df["L"][i] * units
        px, py = df["px"][i], df["py"][i]
        theta_p = np.arctan2(py, px)
        dtheta = df["ANGLE"][i]
        r_arc = s / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = max(int(s / ds), 30)
            x0, y0, px0, py0 = x, y, px, py
            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = propagate(
                    x0, y0, px0, py0, dtheta / n_discrete_bend, s / n_discrete_bend
                )
                theta_pn = np.arctan2(pyn, pxn)

                if df["KEYWORD"][i] == "SBEND" or df["KEYWORD"][i] == "RBEND":
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor=pt.cblind_safe_wheel[0],
                        facecolor=pt.cblind_safe_wheel[0],
                        zorder=0.5,
                        alpha=1,
                    )
                elif (
                    df["KEYWORD"][i] == "QUADRUPOLE"
                    or df["KEYWORD"][i] == "MULTIPOLE"
                    or df["KEYWORD"][i] == "RCOLLIMATOR"
                ):
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor=pt.cblind_safe_wheel[1],
                        facecolor=pt.cblind_safe_wheel[1],
                        zorder=0.51,
                        alpha=1,
                    )
                elif df["KEYWORD"][i] == "DRIFT":
                    rect = patches.Rectangle(
                        (x0, y0 - magnet_thickness * np.cos(theta_pn) / 2),
                        width=s / n_discrete_bend,
                        height=magnet_thickness,
                        angle=theta_pn * 180 / np.pi,
                        linewidth=0.5,
                        edgecolor="lightgrey",
                        facecolor="lightgrey",
                        zorder=0.5,
                        alpha=1,
                    )

                ax.add_patch(rect)
                x0, y0, px0, py0 = xn, yn, pxn, pyn

    return ax


# ds in meters
def create_smoothed_lattice_truncated(
    df,
    emittance_RMS=1e-6,
    midpoint=False,
    n_elements=None,
    rotated=False,
    if_sublattice=False,
    **kwargs,
):
    """
    Create a smooth lattice representation from an existing lattice DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the lattice data with columns 'x', 'y', 'L', 'px', 'py', 'ANGLE', 'BETX', 'BETY', 'GAMMAX', 'GAMMAY', 'DX', 'DPX'.
    emittance_RMS : float
        RMS emittance value to calculate beam sizes.
    n_elements : int
        Number of elements in the new smoother lattice.
    rotated : bool
        If True, rotate and shift the lattice so that the center of the straight section is at (0,0)
        and horizontal. Assumes symmetry about the middle element.
    if_sublattice : bool
        If True, treats the given lattice as just the interaction region and closes it by drawing
        straight lines from the rightmost point to (0, -4.321E8) and back to the leftmost point.
    Returns
    -------
    lattice_dict : dict
        Dictionary containing smoothed lattice data with keys 'x', 'y', 's', 'angle_of_central_p', 'beamsize_x', 'beamsize_y',
        'beamdiv_x', 'beamdiv_y', 'dispersion_Dx', 'dispersion_Dpx', 'inv_s'.
    """

    n_elements_current = df.index.size

    if n_elements <= n_elements_current:
        print(
            f"Warning: n_elements ({n_elements}) is less than or equal to the current number of elements ({n_elements_current})."
        )
        n_elements = n_elements_current

    # element of length in new smoother lattice
    ds = df["L"].sum() / n_elements

    # All desired units are cm or seconds or radians
    smooth_curve_x = np.array([])
    smooth_curve_y = np.array([])

    smooth_curve_s = np.array([])

    smooth_curve_angle_of_central_p = np.array([])

    smooth_curve_beamsize_x = np.array([])
    smooth_curve_beamsize_y = np.array([])

    smooth_curve_beamdiv_x = np.array([])
    smooth_curve_beamdiv_y = np.array([])

    smooth_curve_dispersion_Dx = np.array([])
    smooth_curve_dispersion_Dpx = np.array([])

    # Calculating the distance between elements
    # Note: depending on whether s is measured from the endpoints or midpoint, use L or S
    for i in range(0, n_elements_current):
        x, y, ell = df["x"][i], df["y"][i], df["L"][i]
        if midpoint:
            # case if midpoint is used for S
            s = df["S"][i] - (0.5 * ell)

        else:
            s = df["S"][i] - ell

        px, py = df["px"][i], df["py"][i]
        dtheta = df["ANGLE"][i]
        # theta_p = np.arctan2(py, px)
        # r_arc = l / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = int(ell / ds)
            if n_discrete_bend < 1:
                n_discrete_bend = 1
            x0, y0, px0, py0 = x, y, px, py

            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = advance_in_pos_and_momentum(
                    x0, y0, px0, py0, dtheta / n_discrete_bend, ell / n_discrete_bend
                )
                theta_pn = np.arctan2(pyn, pxn)

                # Arc length position: s is the start of element i, then we add the distance traversed
                # through the element for each subdivision
                s_position = s + (ell / n_discrete_bend) * (j + 1)
                smooth_curve_s = np.append(smooth_curve_s, s_position * const.m_to_cm)
                smooth_curve_x = np.append(smooth_curve_x, xn * const.m_to_cm)
                smooth_curve_y = np.append(smooth_curve_y, yn * const.m_to_cm)
                smooth_curve_angle_of_central_p = np.append(
                    smooth_curve_angle_of_central_p, theta_pn
                )

                smooth_curve_beamsize_x = np.append(
                    smooth_curve_beamsize_x,
                    np.sqrt(emittance_RMS * df["BETX"][i]) * const.m_to_cm,
                )
                smooth_curve_beamsize_y = np.append(
                    smooth_curve_beamsize_y,
                    np.sqrt(emittance_RMS * df["BETY"][i]) * const.m_to_cm,
                )

                smooth_curve_beamdiv_x = np.append(
                    smooth_curve_beamdiv_x,
                    np.arctan(np.sqrt(emittance_RMS * df["GAMMAX"][i])),
                )
                smooth_curve_beamdiv_y = np.append(
                    smooth_curve_beamdiv_y,
                    np.arctan(np.sqrt(emittance_RMS * df["GAMMAY"][i])),
                )

                smooth_curve_dispersion_Dx = np.append(
                    smooth_curve_dispersion_Dx, df["DX"][i]
                )
                smooth_curve_dispersion_Dpx = np.append(
                    smooth_curve_dispersion_Dpx, df["DPX"][i]
                )

                x0, y0, px0, py0 = xn, yn, pxn, pyn

    # Apply rotation and centering if requested
    if rotated:
        # Find the center of the lattice (middle element)
        center_idx = len(smooth_curve_s) // 2
        x_center = smooth_curve_x[center_idx]
        y_center = smooth_curve_y[center_idx]
        angle_at_center = smooth_curve_angle_of_central_p[center_idx]

        # Rotate all points by -angle_at_center to make center horizontal
        cos_angle = np.cos(-angle_at_center)
        sin_angle = np.sin(-angle_at_center)

        # Rotate around center point
        x_rotated = (smooth_curve_x - x_center) * cos_angle - (
            smooth_curve_y - y_center
        ) * sin_angle
        y_rotated = (smooth_curve_x - x_center) * sin_angle + (
            smooth_curve_y - y_center
        ) * cos_angle

        smooth_curve_x = x_rotated
        smooth_curve_y = y_rotated

        # Also rotate the angles
        smooth_curve_angle_of_central_p = (
            smooth_curve_angle_of_central_p - angle_at_center
        )

    # Close the lattice if if_sublattice is True
    if if_sublattice:
        # Find rightmost and leftmost points
        rightmost_idx = np.argmax(smooth_curve_x)
        leftmost_idx = np.argmin(smooth_curve_x)

        x_right = smooth_curve_x[rightmost_idx]
        y_right = smooth_curve_y[rightmost_idx]
        x_left = smooth_curve_x[leftmost_idx]
        y_left = smooth_curve_y[leftmost_idx]

        closing_point_x = 0.0
        closing_point_y = -4.321e5

        # Number of points for each closing segment
        n_closing = 100  # Can adjust this for finer resolution

        # Segment 1: From right to closing point
        x_seg1 = np.linspace(x_right, closing_point_x, n_closing)
        y_seg1 = np.linspace(y_right, closing_point_y, n_closing)

        # Segment 2: From closing point back to left
        x_seg2 = np.linspace(closing_point_x, x_left, n_closing)
        y_seg2 = np.linspace(closing_point_y, y_left, n_closing)

        # Calculate s values for closing segments
        # Segment 1 distance
        seg1_dist = np.sqrt(
            (x_seg1[-1] - x_seg1[0]) ** 2 + (y_seg1[-1] - y_seg1[0]) ** 2
        )
        s_seg1 = np.linspace(
            smooth_curve_s[-1], smooth_curve_s[-1] + seg1_dist, n_closing
        )

        # Segment 2 distance
        seg2_dist = np.sqrt(
            (x_seg2[-1] - x_seg2[0]) ** 2 + (y_seg2[-1] - y_seg2[0]) ** 2
        )
        s_seg2 = np.linspace(s_seg1[-1], s_seg1[-1] + seg2_dist, n_closing)

        # Calculate angles for closing segments
        angle_seg1 = np.arctan2(y_seg1[1:] - y_seg1[:-1], x_seg1[1:] - x_seg1[:-1])
        angle_seg1 = np.append(angle_seg1, angle_seg1[-1])  # Pad last value

        angle_seg2 = np.arctan2(y_seg2[1:] - y_seg2[:-1], x_seg2[1:] - x_seg2[:-1])
        angle_seg2 = np.append(angle_seg2, angle_seg2[-1])  # Pad last value

        # For beam parameters in closing segments, use values from the nearby endpoints
        beamsize_x_seg1 = np.full(n_closing, 0)
        beamsize_x_seg2 = np.full(n_closing, 0)

        beamsize_y_seg1 = np.full(n_closing, 0)
        beamsize_y_seg2 = np.full(n_closing, 0)

        beamdiv_x_seg1 = np.full(n_closing, 0)
        beamdiv_x_seg2 = np.full(n_closing, 0)

        beamdiv_y_seg1 = np.full(n_closing, 0)
        beamdiv_y_seg2 = np.full(n_closing, 0)

        dispersion_Dx_seg1 = np.full(n_closing, 0)
        dispersion_Dx_seg2 = np.full(n_closing, 0)

        dispersion_Dpx_seg1 = np.full(n_closing, 0)
        dispersion_Dpx_seg2 = np.full(n_closing, 0)

        # Append closing segments to the smooth curves
        smooth_curve_x = np.concatenate([smooth_curve_x, x_seg1, x_seg2])
        smooth_curve_y = np.concatenate([smooth_curve_y, y_seg1, y_seg2])
        smooth_curve_s = np.concatenate([smooth_curve_s, s_seg1, s_seg2])
        smooth_curve_angle_of_central_p = np.concatenate(
            [smooth_curve_angle_of_central_p, angle_seg1, angle_seg2]
        )

        smooth_curve_beamsize_x = np.concatenate(
            [smooth_curve_beamsize_x, beamsize_x_seg1, beamsize_x_seg2]
        )
        smooth_curve_beamsize_y = np.concatenate(
            [smooth_curve_beamsize_y, beamsize_y_seg1, beamsize_y_seg2]
        )
        smooth_curve_beamdiv_x = np.concatenate(
            [smooth_curve_beamdiv_x, beamdiv_x_seg1, beamdiv_x_seg2]
        )
        smooth_curve_beamdiv_y = np.concatenate(
            [smooth_curve_beamdiv_y, beamdiv_y_seg1, beamdiv_y_seg2]
        )
        smooth_curve_dispersion_Dx = np.concatenate(
            [smooth_curve_dispersion_Dx, dispersion_Dx_seg1, dispersion_Dx_seg2]
        )
        smooth_curve_dispersion_Dpx = np.concatenate(
            [smooth_curve_dispersion_Dpx, dispersion_Dpx_seg1, dispersion_Dpx_seg2]
        )

    lattice_dict = {}
    u = np.linspace(0, 1, len(smooth_curve_s))
    lattice_dict["x"] = interp1d(u, smooth_curve_x, bounds_error=False, fill_value=None)
    lattice_dict["y"] = interp1d(u, smooth_curve_y, bounds_error=False, fill_value=None)
    lattice_dict["s"] = interp1d(u, smooth_curve_s, bounds_error=False, fill_value=None)

    lattice_dict["angle_of_central_p"] = interp1d(
        u, smooth_curve_angle_of_central_p, bounds_error=False, fill_value=None
    )
    lattice_dict["beamsize_x"] = interp1d(
        u, smooth_curve_beamsize_x, bounds_error=False, fill_value=None
    )
    lattice_dict["beamsize_y"] = interp1d(
        u, smooth_curve_beamsize_y, bounds_error=False, fill_value=None
    )
    lattice_dict["beamdiv_x"] = interp1d(
        u, smooth_curve_beamdiv_x, bounds_error=False, fill_value=None
    )
    lattice_dict["beamdiv_y"] = interp1d(
        u, smooth_curve_beamdiv_y, bounds_error=False, fill_value=None
    )
    lattice_dict["dispersion_Dx"] = interp1d(
        u, smooth_curve_dispersion_Dx, bounds_error=False, fill_value=None
    )
    lattice_dict["dispersion_Dpx"] = interp1d(
        u, smooth_curve_dispersion_Dpx, bounds_error=False, fill_value=None
    )

    lattice_dict["inv_s"] = interp1d(
        smooth_curve_s, u, bounds_error=False, fill_value=None
    )

    lattice_dict["beam_p0"] = np.sqrt(
        float(df.attrs["ENERGY"]) ** 2 - const.m_mu**2
    )  # Default to 1.0 if not set

    # Update with user input
    lattice_dict.update(kwargs)

    return lattice_dict


####################################################################################################
# LINEAR OPTICS
####################################################################################################


def _unwrap_phase(phi: np.ndarray) -> np.ndarray:
    """Unwrap a phase-like quantity in turns or radians (MAD-X mux/muy are typically in turns)."""
    # np.unwrap assumes radians; if mux is in "turns" you can unwrap in turns by scaling.
    # But linear interpolation of mux is only used if you want it; here we keep it robust:
    # treat as turns -> convert to radians, unwrap, then back.
    phi = np.asarray(phi, dtype=float)
    phi_rad = 2.0 * np.pi * phi
    phi_unw = np.unwrap(phi_rad)
    return phi_unw / (2.0 * np.pi)


def create_smoothed_lattice(
    twiss_df: pd.DataFrame,
    emittance_RMS: float,
    n_elements: int = 120_000,
    rotated: bool = False,
    if_sublattice: bool = False,
    midpoint: bool = True,
    include_dispersion: bool = True,
    sigma_delta: float | None = None,
    # geometry inputs (if you already have them in the df, set these column names)
    x_col: str = "x",
    y_col: str = "y",
    px_col: str = "px",
    py_col: str = "py",
    angle_col: str = "ANGLE",
    # If your df has ALFX/ALFY; if not, we’ll infer gamma from GAMMAX/GAMMAY if provided
    use_alpha: bool = True,
    # constants container you use in your codebase
    **kwargs,
):
    """
    New-physics smoothed lattice (Twiss->gamma->beam envelopes + optional dispersion+sigma_delta),
    but returns the OLD interface: interp1d callables in a lattice_dict.

    Returns at least:
      x,y,s,angle_of_central_p,beamsize_x,beamsize_y,beamdiv_x,beamdiv_y,
      dispersion_Dx,dispersion_Dpx,inv_s,beam_p0
    plus useful extras for MC sampling (covariances, twiss, etc.) as interpolants.
    """

    df = twiss_df.copy()

    # ---- Required optics columns
    required = ["S", "L", "BETX", "BETY"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"TWISS table missing required column '{c}'")

    # ALFX/ALFY are needed if you want covariances and to compute gamma from Twiss.
    # If missing, we can fall back to GAMMAX/GAMMAY if present.
    have_alpha = ("ALFX" in df.columns) and ("ALFY" in df.columns)
    have_gamma = ("GAMMAX" in df.columns) and ("GAMMAY" in df.columns)

    if use_alpha and not have_alpha and not have_gamma:
        raise ValueError(
            "Need ALFX/ALFY or GAMMAX/GAMMAY columns to compute divergences (gamma)."
        )

    # ---- Optional geometry columns (x,y,px,py/angle)
    have_xy = (x_col in df.columns) and (y_col in df.columns)
    have_p = (px_col in df.columns) and (py_col in df.columns)
    have_angle = angle_col in df.columns

    # Sort by S
    s = np.asarray(df["S"], dtype=float)
    Lcol = np.asarray(df["L"], dtype=float)
    order = np.argsort(s)
    df = df.iloc[order].reset_index(drop=True)
    s = s[order]
    Lcol = Lcol[order]

    # Element edge positions if midpoint is true
    if midpoint:
        s_end = s + 0.5 * Lcol
    else:
        s_end = s

    C = float(np.max(s_end))
    if C <= 0:
        raise ValueError("Could not infer positive circumference from TWISS data.")

    # Periodic interpolation knots helper
    def periodic_knots(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ss = np.mod(s, C)
        idx = np.argsort(ss)
        ss = ss[idx]
        xx = np.asarray(arr, dtype=float)[idx]
        ss_ext = np.concatenate([ss - C, ss, ss + C])
        xx_ext = np.concatenate([xx, xx, xx])
        return ss_ext, xx_ext

    # Uniform grid in meters for physics; later convert to cm for output interpolants
    n_elements = int(n_elements)
    s_grid_m = np.linspace(0.0, C, n_elements, endpoint=False)

    def interp_on_grid(arr: np.ndarray) -> np.ndarray:
        ss_ext, xx_ext = periodic_knots(arr)
        return np.interp(s_grid_m, ss_ext, xx_ext)

    # ---- Optics fields
    betx = interp_on_grid(df["BETX"].to_numpy(dtype=float))
    bety = interp_on_grid(df["BETY"].to_numpy(dtype=float))

    if have_alpha:
        alfx = interp_on_grid(df["ALFX"].to_numpy(dtype=float))
        alfy = interp_on_grid(df["ALFY"].to_numpy(dtype=float))
        gamx = (1.0 + alfx**2) / betx
        gamy = (1.0 + alfy**2) / bety
    else:
        # fallback to provided gammas
        alfx = None
        alfy = None
        gamx = interp_on_grid(df["GAMMAX"].to_numpy(dtype=float))
        gamy = interp_on_grid(df["GAMMAY"].to_numpy(dtype=float))

    # Optional phase advances (not required, but useful sometimes)
    mux = (
        _unwrap_phase(df["MUX"].to_numpy(dtype=float)) if "MUX" in df.columns else None
    )
    muy = (
        _unwrap_phase(df["MUY"].to_numpy(dtype=float)) if "MUY" in df.columns else None
    )
    if mux is not None:
        mux = interp_on_grid(mux)
    if muy is not None:
        muy = interp_on_grid(muy)

    # ---- Dispersion
    if include_dispersion:
        Dx = (
            interp_on_grid(df["DX"].to_numpy(dtype=float))
            if "DX" in df.columns
            else np.zeros_like(s_grid_m)
        )
        Dpx = (
            interp_on_grid(df["DPX"].to_numpy(dtype=float))
            if "DPX" in df.columns
            else np.zeros_like(s_grid_m)
        )
    else:
        Dx = np.zeros_like(s_grid_m)
        Dpx = np.zeros_like(s_grid_m)

    # ---- Beam envelopes (NEW PHYSICS semantics from your create_smoothed_lattice)
    eps = float(emittance_RMS)
    sigx = np.sqrt(np.maximum(0.0, eps * betx))
    sigxp = np.sqrt(np.maximum(0.0, eps * gamx))
    sigy = np.sqrt(np.maximum(0.0, eps * bety))
    sigyp = np.sqrt(np.maximum(0.0, eps * gamy))

    # Correlations: cov(x,x') = -eps*alpha
    if have_alpha:
        cov_x_xp = -eps * alfx
        cov_y_yp = -eps * alfy
    else:
        cov_x_xp = np.zeros_like(s_grid_m)
        cov_y_yp = np.zeros_like(s_grid_m)

    # Include dispersion contribution if sigma_delta is provided (position+angle RMS)
    if include_dispersion and (sigma_delta is not None):
        sd = float(sigma_delta)
        sigx_tot = np.sqrt(sigx**2 + (Dx * sd) ** 2)
        sigxp_tot = np.sqrt(sigxp**2 + (Dpx * sd) ** 2)
    else:
        sigx_tot, sigxp_tot = sigx, sigxp

    # ---- Geometry (x,y, angle_of_central_p)
    # If your df already contains x,y and either angle or px,py, we interpolate those.
    # If not present, we still return callables, but they will be zero (so downstream doesn’t crash).
    if have_xy:
        x_m = interp_on_grid(df[x_col].to_numpy(dtype=float))
        y_m = interp_on_grid(df[y_col].to_numpy(dtype=float))
    else:
        x_m = np.zeros_like(s_grid_m)
        y_m = np.zeros_like(s_grid_m)

    if have_angle:
        angle = interp_on_grid(df[angle_col].to_numpy(dtype=float))
    elif have_p:
        px = interp_on_grid(df[px_col].to_numpy(dtype=float))
        py = interp_on_grid(df[py_col].to_numpy(dtype=float))
        angle = np.arctan2(py, px)
    else:
        angle = np.zeros_like(s_grid_m)

    # Rotation/closing: keep flags but don’t invent geometry unless you explicitly do that elsewhere.
    # (If you want the earlier geometric rotate/close behavior, we can merge it in cleanly.)
    # if rotated:
    #     # Rotate around middle of the ring in the sampled arrays
    #     mid = len(s_grid_m) // 2
    #     x0, y0 = x_m[mid], y_m[mid]
    #     a0 = angle[mid]
    #     ca, sa = np.cos(-a0), np.sin(-a0)
    #     xr = (x_m - x0) * ca - (y_m - y0) * sa
    #     yr = (x_m - x0) * sa + (y_m - y0) * ca
    #     x_m, y_m = xr, yr
    #     angle = angle - a0
    if rotated:
        # Rotate around middle of the ring in the sampled arrays
        mid = len(s_grid_m) // 2
        x0, y0 = x_m[mid], y_m[mid]

        # --- compute the local trajectory direction from geometry (robust)
        # use a window around the midpoint to estimate tangent
        k = 200  # points; tune 50-500 depending on your grid density
        i0 = max(0, mid - k)
        i1 = min(len(x_m) - 1, mid + k)

        dx = x_m[i1] - x_m[i0]
        dy = y_m[i1] - y_m[i0]
        theta_tan = np.arctan2(dy, dx)  # direction of trajectory near the center

        # Rotate by -theta_tan around (x0,y0)
        ca, sa = np.cos(-theta_tan), np.sin(-theta_tan)
        xr = (x_m - x0) * ca - (y_m - y0) * sa
        yr = (x_m - x0) * sa + (y_m - y0) * ca
        x_m, y_m = xr, yr

        # If you still want an "angle_of_central_p" returned, make it consistent:
        # Option 1 (recommended): define it as the geometric tangent angle everywhere
        # (so it's guaranteed consistent with x,y)
        dxf = np.gradient(x_m, s_grid_m)
        dyf = np.gradient(y_m, s_grid_m)
        angle = np.arctan2(dyf, dxf)

        # Optional: force the direction at the center to point along +x (not -x)
        if np.mean(np.diff(x_m[mid - 200 : mid + 200])) < 0:
            x_m *= -1
            angle = np.arctan2(np.gradient(y_m, s_grid_m), np.gradient(x_m, s_grid_m))

    # if_sublattice: the “close by straight lines” logic is geometry-specific.
    # If you still need that, say so and I’ll splice the earlier closure logic into this version
    # while keeping all optics periodic. For now: we keep the physics outputs and interface.

    # ---- Convert to OLD units convention
    s_cm = s_grid_m * const.m_to_cm
    x_cm = x_m * const.m_to_cm
    y_cm = y_m * const.m_to_cm

    beamsize_x_cm = sigx_tot * const.m_to_cm
    beamsize_y_cm = sigy * const.m_to_cm

    # IMPORTANT: beamdiv in your old interface was an angle in rad.
    # Your new-physics version uses sigma_x' directly (dimensionless ~ rad for small angles).
    # If you want the "angle" version, keep arctan. For small angles, arctan(sigma) ~ sigma.
    beamdiv_x = np.arctan(sigxp_tot)
    beamdiv_y = np.arctan(sigyp)

    # ---- Build interpolation objects exactly as before
    lattice_dict = {}
    u = np.linspace(0.0, 1.0, len(s_cm))

    lattice_dict["x"] = interp1d(u, x_cm, bounds_error=False, fill_value=None)
    lattice_dict["y"] = interp1d(u, y_cm, bounds_error=False, fill_value=None)
    lattice_dict["s"] = interp1d(u, s_cm, bounds_error=False, fill_value=None)

    lattice_dict["angle_of_central_p"] = interp1d(
        u, angle, bounds_error=False, fill_value=None
    )

    lattice_dict["beamsize_x"] = interp1d(
        u, beamsize_x_cm, bounds_error=False, fill_value=None
    )
    lattice_dict["beamsize_y"] = interp1d(
        u, beamsize_y_cm, bounds_error=False, fill_value=None
    )

    lattice_dict["beamdiv_x"] = interp1d(
        u, beamdiv_x, bounds_error=False, fill_value=None
    )
    lattice_dict["beamdiv_y"] = interp1d(
        u, beamdiv_y, bounds_error=False, fill_value=None
    )

    lattice_dict["dispersion_Dx"] = interp1d(u, Dx, bounds_error=False, fill_value=None)
    lattice_dict["dispersion_Dpx"] = interp1d(
        u, Dpx, bounds_error=False, fill_value=None
    )

    lattice_dict["inv_s"] = interp1d(s_cm, u, bounds_error=False, fill_value=None)

    # ---- Beam momentum (same definition as you asked)
    lattice_dict["beam_p0"] = np.sqrt(float(df.attrs["ENERGY"]) ** 2 - const.m_mu**2)

    # ---- Extras I strongly recommend for MC beam sampling (also interpolants)
    # Twiss / gamma
    lattice_dict["betx"] = interp1d(u, betx, bounds_error=False, fill_value=None)
    lattice_dict["bety"] = interp1d(u, bety, bounds_error=False, fill_value=None)
    lattice_dict["gamx"] = interp1d(u, gamx, bounds_error=False, fill_value=None)
    lattice_dict["gamy"] = interp1d(u, gamy, bounds_error=False, fill_value=None)
    if have_alpha:
        lattice_dict["alfx"] = interp1d(u, alfx, bounds_error=False, fill_value=None)
        lattice_dict["alfy"] = interp1d(u, alfy, bounds_error=False, fill_value=None)

    # Covariances (for correlated Gaussian sampling in (x,x') and (y,y'))
    lattice_dict["cov_x_xp"] = interp1d(
        u, cov_x_xp, bounds_error=False, fill_value=None
    )
    lattice_dict["cov_y_yp"] = interp1d(
        u, cov_y_yp, bounds_error=False, fill_value=None
    )

    # If you want the *raw* sigma_x' (not arctan), expose it too:
    lattice_dict["sigma_xp"] = interp1d(
        u, sigxp_tot, bounds_error=False, fill_value=None
    )
    lattice_dict["sigma_yp"] = interp1d(u, sigyp, bounds_error=False, fill_value=None)

    # Helpful meta for sampling
    lattice_dict["length_cm"] = float(C * const.m_to_cm)
    lattice_dict["length_m"] = float(C)
    lattice_dict["emittance_RMS"] = float(eps)
    lattice_dict["midpoint"] = bool(midpoint)
    lattice_dict["include_dispersion"] = bool(include_dispersion)
    lattice_dict["sigma_delta"] = None if sigma_delta is None else float(sigma_delta)

    # Optional phases if you ever need them
    if mux is not None:
        lattice_dict["mux"] = interp1d(u, mux, bounds_error=False, fill_value=None)
    if muy is not None:
        lattice_dict["muy"] = interp1d(u, muy, bounds_error=False, fill_value=None)

    # user extras
    lattice_dict.update(kwargs)

    # ---- Contract check (fail fast if something regresses)
    REQUIRED = {
        "x",
        "y",
        "s",
        "angle_of_central_p",
        "beamsize_x",
        "beamsize_y",
        "beamdiv_x",
        "beamdiv_y",
        "dispersion_Dx",
        "dispersion_Dpx",
        "inv_s",
        "beam_p0",
    }
    missing = REQUIRED - set(lattice_dict.keys())
    if missing:
        raise RuntimeError(f"Missing required lattice keys: {sorted(missing)}")

    return lattice_dict
