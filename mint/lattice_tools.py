import pandas as pd
import numpy as np
import matplotlib.patches as patches
from scipy.interpolate import interp1d

from mint import const
from mint import plot_tools as pt


class Lattice:
    """
    Class to represent a particle accelerator lattice.
    """

    def __init__(self, **kwargs):

        lattice_dict = kwargs.copy()

        # World-coordinate convention (both lab frame and beam-comoving frame):
        #   x -> toward the center of the ring (+x points to the center)
        #   y -> up, out of the ring plane (0 for a planar ring)
        #   z -> tangential, along the beam direction of travel
        # lattice.x/y/z return world coordinates directly, and lattice.tangent(u)
        # returns the world-frame unit tangent [tx, ty, tz] of the central orbit.
        for key in [
            "beam_p0",
            "x",
            "y",
            "s",
            "angle_of_central_p",
            "inv_s",
        ]:
            if key in lattice_dict:
                setattr(self, key, lattice_dict.pop(key))
            else:
                raise ValueError(f"Lattice dictionary must contain {key} key.")

        # World longitudinal coordinate z (tangential). Defaults to 0 (planar ring
        # drawn purely in the horizontal x-z plane) if a producer does not supply it.
        if "z" in lattice_dict:
            self.z = lattice_dict.pop("z")
        else:
            self.z = lambda u: np.zeros_like(np.asarray(u, dtype=float))

        # World unit tangent [tx, ty, tz] of the central orbit. If not supplied,
        # derive it from angle_of_central_p, the tangent angle measured in the
        # horizontal z-x plane from +z toward +x: t = (sin a, 0, cos a).
        if "tangent" in lattice_dict:
            self.tangent = lattice_dict.pop("tangent")
        else:
            self.tangent = lambda uu: np.vstack(
                [
                    np.sin(self.angle_of_central_p(uu)),
                    np.zeros_like(np.asarray(uu, dtype=float)),
                    np.cos(self.angle_of_central_p(uu)),
                ]
            )

        # Transverse beam divergence
        if "beamdiv_x" in lattice_dict and "beamdiv_y" in lattice_dict:
            self.beamdiv_x = lattice_dict.pop("beamdiv_x")
            self.beamdiv_y = lattice_dict.pop("beamdiv_y")
        elif "beamdiv" in lattice_dict:
            self.beamdiv_x = lattice_dict.pop("beamdiv")
            self.beamdiv_y = self.beamdiv_x
        elif (
            ("beamdiv_x" in lattice_dict and "beamdiv_y" not in lattice_dict)
            or ("beamdiv_x" not in lattice_dict and "beamdiv_y" in lattice_dict)
            or ("beamdiv_x" in lattice_dict and "beam_div" in lattice_dict)
            or ("beamdiv_y" in lattice_dict and "beam_div" in lattice_dict)
        ):
            raise ValueError("Inconsistent beam divergence specifications.")
        else:
            self.beamdiv_x = lambda x: 0
            self.beamdiv_y = lambda x: 0

        # Longitudinal beam divergence
        if "beamdiv_z" in lattice_dict:
            self.beamdiv_z = lattice_dict.pop("beamdiv_z")
        else:
            self.beamdiv_z = lambda x: 0

        # Adding the beam transverse size
        if "beamsize_x" in lattice_dict and "beamsize_y" in lattice_dict:
            self.beamsize_x = lattice_dict.pop("beamsize_x")
            self.beamsize_y = lattice_dict.pop("beamsize_y")
        elif "beamsize" in lattice_dict:
            self.beamsize_x = lattice_dict.pop("beamsize")
            self.beamsize_y = self.beamsize_x
        elif (
            ("beamsize_x" in lattice_dict and "beamsize_y" not in lattice_dict)
            or ("beamsize_x" not in lattice_dict and "beamsize_y" in lattice_dict)
            or ("beamsize_x" in lattice_dict and "beamsize" in lattice_dict)
            or ("beamsize_y" in lattice_dict and "beamsize" in lattice_dict)
        ):
            raise ValueError("Inconsistent beam size specifications.")
        else:
            self.beamsize_x = lambda x: 0
            self.beamsize_y = lambda x: 0

        # Longitudinal beam size
        if "beamsize_z" in lattice_dict:
            self.beamsize_z = lattice_dict.pop("beamsize_z")
        else:
            self.beamsize_z = lambda x: 0

        # dP/dx
        if "dpdx" in lattice_dict:
            self.dpdx = lattice_dict.pop("dpdx")
        else:
            self.dpdx = lambda x: 0

        # idenfity which pass
        if "which_pass" in lattice_dict:
            self.which_pass = lattice_dict.pop("which_pass")
        else:
            self.which_pass = lambda x: 1

        # Making sure everyone is callable
        for attr in [
            "beam_p0",
            "beamdiv_x",
            "beamdiv_y",
            "beamdiv_z",
            "beamsize_x",
            "beamsize_y",
            "beamsize_z",
        ]:
            val = getattr(self, attr)
            if not callable(val):
                if isinstance(val, (int, float)):
                    setattr(self, attr, lambda _, v=val: v)
                else:
                    print(
                        "Warning! Attribute",
                        attr,
                        "is not callable and not a number. Setting it to zero.",
                    )
                    setattr(self, attr, lambda _, v=val: 0)

        self.Nmu_per_bunch = lattice_dict.pop("Nmu_per_bunch", 1)
        self.duty_factor = lattice_dict.pop("duty_factor", 1)
        self.bunch_multiplicity = lattice_dict.pop("bunch_multiplicity", 1)
        self.finj = lattice_dict.pop("finj", 1)

        # lattice designs
        self.name = lattice_dict.pop("name", "unnamed")
        self.short_name = lattice_dict.pop("short_name", self.name)
        self.n_elements = lattice_dict.pop("n_elements", None)

        # Any remaining entries (e.g. Twiss interpolants from
        # create_smoothed_lattice) are stored as attributes.
        for key, value in lattice_dict.items():
            setattr(self, key, value)




def create_straight_lattice(
    total_length=100e2, n_elements=10_000, p0_injected=0.225, p0_ejected=1.25, **kwargs
):

    n_points = 300

    # Straight section: longitudinal runs along the beam; transverse = 0.
    u_vals = np.linspace(0, 1, n_points)
    longitudinal = u_vals * total_length - total_length / 2

    # WORLD coordinates: z = longitudinal (tangential), x = transverse (0 here),
    # y = 0 (planar).
    z_world = longitudinal
    x_world = np.zeros_like(longitudinal)
    y_world = np.zeros_like(longitudinal)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_world, y_world, z_world), n_elements=n_elements
    )
    # Any additional user-input
    kwargs["beam_p0"] = interp1d(
        u_vals, u_vals * (p0_ejected - p0_injected) + p0_injected
    )

    kwargs["dpdx"] = interp1d(
        u_vals, np.full_like(u_vals, (p0_ejected - p0_injected) / total_length)
    )

    kwargs["n_elements"] = n_elements

    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    # vertices in world (x, y, z) order
    lattice.vertices = (x_world, y_world, z_world)

    return lattice


def get_s_element(x, y, z=None):
    """Return segment lengths between consecutive points.

    If z is provided, compute 3D lengths; otherwise 2D.
    """
    if z is None:
        return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    else:
        return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)


def create_RLA_lattice(
    straight_length=70e2,
    n_elements=10_000,
    n_points=300,
    p0_injection=1.25,
    dp_dx_LA=0.1,
    half=True,
    beam_dyn=True,
    **kwargs,
):
    """
    Design taken from
        https://indico.fnal.gov/event/8903/contributions/110580/attachments/71967/86347/Acceleration_NF_MC.pdf


        Structure looks like:

            (straight section + drop section + back through straight section + ...)
        where
            drop section = (3 drift sections + 1 bending section + 3 drift sections)

        Repeat for each pass with larger and larger drop sections

        Acceleration only happens in the straight section.

    """
    n_points_per_element = n_points

    s_length = np.array([0])
    dpdx = np.array([])
    which_pass = np.array([])

    # first pass
    markers_1 = np.array([(0.161586 * straight_length) + (straight_length / 2)])
    markers_1 = np.append(markers_1, markers_1[0] + (0.109756 * straight_length))
    markers_1 = np.append(markers_1, markers_1[1] + (0.246951 * straight_length))

    if half:
        x_1 = np.linspace(0, straight_length / 2, n_points_per_element)
        y_1 = np.full_like(x_1, 0)
        s_length = np.append(s_length, get_s_element(x_1, y_1))
        dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))
    else:
        x_1 = np.linspace(
            -straight_length / 2, straight_length / 2, n_points_per_element
        )
        y_1 = np.full_like(x_1, 0)
        s_length = np.append(s_length, get_s_element(x_1, y_1))
        dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))

    # droplet drift sections
    x_1_1 = np.linspace(straight_length / 2, markers_1[0], n_points_per_element)
    y_1_1 = 0.18529 * x_1_1 - (0.0926448 * straight_length)
    s_length = np.append(s_length, get_s_element(x_1_1, y_1_1))

    x_1_2 = np.linspace(markers_1[0], markers_1[1], n_points_per_element)
    y_1_2 = 0.436461 * x_1_2 - (0.258817 * straight_length)
    s_length = np.append(s_length, get_s_element(x_1_2, y_1_2))

    x_1_3 = np.linspace(markers_1[1], markers_1[2], n_points_per_element)
    y_1_3 = 0.654692 * x_1_3 - (0.427147 * straight_length)
    s_length = np.append(s_length, get_s_element(x_1_3, y_1_3))

    # droplet arc
    theta_1 = np.linspace(1.04327, (2 * np.pi) - 1.04327, n_points_per_element)
    radius_1 = 0.284043 * straight_length
    center_1 = 0.645828 * straight_length + (straight_length / 2)

    x_curve_1 = -radius_1 * np.cos(theta_1) + center_1
    y_curve_1 = radius_1 * np.sin(theta_1)
    s_length = np.append(s_length, get_s_element(x_curve_1, y_curve_1))

    # droplet return drift sections
    x_1_4 = x_1_3[::-1]
    y_1_4 = -y_1_3[::-1]
    s_length = np.append(s_length, get_s_element(x_1_4, y_1_4))

    x_1_5 = x_1_2[::-1]
    y_1_5 = -y_1_2[::-1]
    s_length = np.append(s_length, get_s_element(x_1_5, y_1_5))

    x_1_6 = x_1_1[::-1]
    y_1_6 = -y_1_1[::-1]
    s_length = np.append(s_length, get_s_element(x_1_6, y_1_6))

    dpdx = np.append(dpdx, np.full(7 * n_points_per_element, 0.0))
    which_pass = np.full_like(dpdx, 1)

    # second pass

    markers_2 = np.array([(-0.170732 * straight_length) - (straight_length / 2)])
    markers_2 = np.append(markers_2, markers_2[0] - (0.128049 * straight_length))
    markers_2 = np.append(markers_2, markers_2[1] - (0.286585 * straight_length))

    x_2 = np.linspace(straight_length / 2, -straight_length / 2, n_points_per_element)
    y_2 = np.full_like(x_2, 0)
    s_length = np.append(s_length, get_s_element(x_2, y_2))
    dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))
    which_pass = np.append(which_pass, np.full(n_points_per_element, 2))

    x_2_1 = np.linspace(-straight_length / 2, markers_2[0], n_points_per_element)
    y_2_1 = -0.210437 * x_2_1 - (0.105218 * straight_length)
    s_length = np.append(s_length, get_s_element(x_2_1, y_2_1))

    x_2_2 = np.linspace(markers_2[0], markers_2[1], n_points_per_element)
    y_2_2 = -0.444256 * x_2_2 - (0.262048 * straight_length)
    s_length = np.append(s_length, get_s_element(x_2_2, y_2_2))

    x_2_3 = np.linspace(markers_2[1], markers_2[2], n_points_per_element)
    y_2_3 = -0.658172 * x_2_3 - (0.43292 * straight_length)
    s_length = np.append(s_length, get_s_element(x_2_3, y_2_3))

    theta_2 = np.linspace(1.0748, (2 * np.pi) - 1.0748, n_points_per_element)
    radius_2 = 0.329418 * straight_length
    center_2 = -0.737655 * straight_length - (straight_length / 2)

    x_curve_2 = radius_2 * np.cos(theta_2) + center_2
    y_curve_2 = radius_2 * np.sin(theta_2)
    s_length = np.append(s_length, get_s_element(x_curve_2, y_curve_2))

    x_2_4 = x_2_3[::-1]
    y_2_4 = -y_2_3[::-1]
    s_length = np.append(s_length, get_s_element(x_2_4, y_2_4))

    x_2_5 = x_2_2[::-1]
    y_2_5 = -y_2_2[::-1]
    s_length = np.append(s_length, get_s_element(x_2_5, y_2_5))

    x_2_6 = x_2_1[::-1]
    y_2_6 = -y_2_1[::-1]
    s_length = np.append(s_length, get_s_element(x_2_6, y_2_6))

    dpdx = np.append(dpdx, np.full(7 * n_points_per_element, 0.0))
    which_pass = np.append(which_pass, np.full(7 * n_points_per_element, 2))

    # third pass

    markers_3 = np.array([(0.216463 * straight_length) + (straight_length / 2)])
    markers_3 = np.append(markers_3, markers_3[0] + (0.155488 * straight_length))
    markers_3 = np.append(markers_3, markers_3[1] + (0.344512 * straight_length))

    x_3 = np.linspace(-straight_length / 2, straight_length / 2, n_points_per_element)
    y_3 = np.full_like(x_3, 0)
    s_length = np.append(s_length, get_s_element(x_3, y_3))
    dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))
    which_pass = np.append(which_pass, np.full(n_points_per_element, 3))

    x_3_1 = np.linspace(straight_length / 2, markers_3[0], n_points_per_element)
    y_3_1 = 0.0968204 * x_3_1 - (0.0484102 * straight_length)
    s_length = np.append(s_length, get_s_element(x_3_1, y_3_1))

    x_3_2 = np.linspace(markers_3[0], markers_3[1], n_points_per_element)
    y_3_2 = 0.365857 * x_3_2 - (0.241165 * straight_length)
    s_length = np.append(s_length, get_s_element(x_3_2, y_3_2))

    x_3_3 = np.linspace(markers_3[1], markers_3[2], n_points_per_element)
    y_3_3 = 0.651794 * x_3_3 - (0.490488 * straight_length)
    s_length = np.append(s_length, get_s_element(x_3_3, y_3_3))

    theta_3 = np.linspace(1.06755, (2 * np.pi) - 1.06755, n_points_per_element)
    radius_3 = 0.354524 * straight_length
    center_3 = 0.871086 * straight_length + (straight_length / 2)

    x_curve_3 = -radius_3 * np.cos(theta_3) + center_3
    y_curve_3 = radius_3 * np.sin(theta_3)
    s_length = np.append(s_length, get_s_element(x_curve_3, y_curve_3))

    x_3_4 = x_3_3[::-1]
    y_3_4 = -y_3_3[::-1]
    s_length = np.append(s_length, get_s_element(x_3_4, y_3_4))

    x_3_5 = x_3_2[::-1]
    y_3_5 = -y_3_2[::-1]
    s_length = np.append(s_length, get_s_element(x_3_5, y_3_5))

    x_3_6 = x_3_1[::-1]
    y_3_6 = -y_3_1[::-1]
    s_length = np.append(s_length, get_s_element(x_3_6, y_3_6))

    dpdx = np.append(dpdx, np.full(7 * n_points_per_element, 0.0))
    which_pass = np.append(which_pass, np.full(7 * n_points_per_element, 3))

    # fourth pass

    markers_4 = np.array([(-0.246951 * straight_length) - (straight_length / 2)])
    markers_4 = np.append(markers_4, markers_4[0] - (0.17378 * straight_length))
    markers_4 = np.append(markers_4, markers_4[1] - (0.390244 * straight_length))

    x_4 = np.linspace(straight_length / 2, -straight_length / 2, n_points_per_element)
    y_4 = np.full_like(x_4, 0)
    s_length = np.append(s_length, get_s_element(x_4, y_4))
    dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))
    which_pass = np.append(which_pass, np.full(n_points_per_element, 4))

    x_4_1 = np.linspace(-straight_length / 2, markers_4[0], n_points_per_element)
    y_4_1 = -0.0848675 * x_4_1 - (0.0424337 * straight_length)
    s_length = np.append(s_length, get_s_element(x_4_1, y_4_1))

    x_4_2 = np.linspace(markers_4[0], markers_4[1], n_points_per_element)
    y_4_2 = -0.361803 * x_4_2 - (0.249291 * straight_length)
    s_length = np.append(s_length, get_s_element(x_4_2, y_4_2))

    x_4_3 = np.linspace(markers_4[1], markers_4[2], n_points_per_element)
    y_4_3 = -0.659804 * x_4_3 - (0.52367 * straight_length)
    s_length = np.append(s_length, get_s_element(x_4_3, y_4_3))

    theta_4 = np.linspace(1.06593, (2 * np.pi) - 1.06593, n_points_per_element)
    radius_4 = 0.392126 * straight_length
    center_4 = -0.977735 * straight_length - (straight_length / 2)

    x_curve_4 = radius_4 * np.cos(theta_4) + center_4
    y_curve_4 = radius_4 * np.sin(theta_4)
    s_length = np.append(s_length, get_s_element(x_curve_4, y_curve_4))

    x_4_4 = x_4_3[::-1]
    y_4_4 = -y_4_3[::-1]
    s_length = np.append(s_length, get_s_element(x_4_4, y_4_4))

    x_4_5 = x_4_2[::-1]
    y_4_5 = -y_4_2[::-1]
    s_length = np.append(s_length, get_s_element(x_4_5, y_4_5))

    x_4_6 = x_4_1[::-1]
    y_4_6 = -y_4_1[::-1]
    s_length = np.append(s_length, get_s_element(x_4_6, y_4_6))

    dpdx = np.append(dpdx, np.full(7 * n_points_per_element, 0.0))
    which_pass = np.append(which_pass, np.full(7 * n_points_per_element, 4))

    # Fifth and final pass
    x_5 = np.linspace(straight_length / 2, -straight_length / 2, n_points_per_element)
    y_5 = np.full_like(x_5, 0)
    s_length = np.append(s_length, get_s_element(x_5, y_5))
    dpdx = np.append(dpdx, np.full(n_points_per_element, dp_dx_LA))
    which_pass = np.append(which_pass, np.full(n_points_per_element, 5))

    # Concatenate all segments
    x_RLA = np.concatenate(
        [
            x_1,
            x_1_1,
            x_1_2,
            x_1_3,
            x_curve_1,
            x_1_4,
            x_1_5,
            x_1_6,
            x_2,
            x_2_1,
            x_2_2,
            x_2_3,
            x_curve_2,
            x_2_4,
            x_2_5,
            x_2_6,
            x_3,
            x_3_1,
            x_3_2,
            x_3_3,
            x_curve_3,
            x_3_4,
            x_3_5,
            x_3_6,
            x_4,
            x_4_1,
            x_4_2,
            x_4_3,
            x_curve_4,
            x_4_4,
            x_4_5,
            x_4_6,
            x_5,
        ]
    )
    y_RLA = np.concatenate(
        [
            y_1,
            y_1_1,
            y_1_2,
            y_1_3,
            y_curve_1,
            y_1_4,
            y_1_5,
            y_1_6,
            y_2,
            y_2_1,
            y_2_2,
            y_2_3,
            y_curve_2,
            y_2_4,
            y_2_5,
            y_2_6,
            y_3,
            y_3_1,
            y_3_2,
            y_3_3,
            y_curve_3,
            y_3_4,
            y_3_5,
            y_3_6,
            y_4,
            y_4_1,
            y_4_2,
            y_4_3,
            y_curve_4,
            y_4_4,
            y_4_5,
            y_4_6,
            y_5,
        ]
    )
    # Remove the last element (dpdx is defined in intervals, not points)
    dpdx = dpdx[:-1]

    # Normalize the arc-length
    ds_length = get_s_element(x_RLA, y_RLA)
    s = np.concatenate([[0], np.cumsum(ds_length)])
    u = s / s[-1]

    # define piecewise beta: example linear on straights, close to zero on arcs
    beta = np.full_like(s, 1e-3)

    if beam_dyn:

        # example user parameters for the straight segments (a,b) per straight pass
        # these are just examples — replace with your desired a,b values per straight
        straight_params = [
            [0.000000e00, 8.290000e04],  # seg 1  (s_start, s_end)
            [2.805025e05, 3.634025e05],  # seg 9
            [5.809080e05, 6.638080e05],  # seg 17
            [9.264227e05, 1.009323e06],  # seg 25
            [1.380230e06, s[-1]],  # seg 33 (final straight)
        ]

        for seg in straight_params:

            mask = (s >= seg[0]) & (s <= seg[1])
            beta[mask] = 0.0105498 * s[mask] + 21.5768e2
        print(beta)

        emittance = 25e-3  # [cm-rad]

        beamsize = np.sqrt(beta * emittance)
        # prevent divide-by-zero
        beamdiv = np.where(beta < 1.0, beta, np.sqrt(emittance / beta))

        # ---------------------------------------
        # Store as interpolation functions
        # ---------------------------------------
        # kwargs["beta"] = interp1d(u, beta, fill_value="extrapolate")
        kwargs["beamsize_x"] = interp1d(u, beamsize, fill_value="extrapolate")
        kwargs["beamdiv_x"] = interp1d(u, beamdiv, fill_value="extrapolate")

        # You can copy these to y if symmetric:
        kwargs["beamsize_y"] = kwargs["beamsize_x"]
        kwargs["beamdiv_y"] = kwargs["beamdiv_x"]

    # Map to WORLD coordinates: z = longitudinal (x_RLA, along the straights),
    # x = transverse (y_RLA, the drop/arc excursions), y = 0 (planar RLA).
    z_world = x_RLA
    x_world = y_RLA
    y_world = np.zeros_like(x_RLA)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_world, y_world, z_world), n_elements=n_elements
    )

    kwargs["beam_p0"] = interp1d(
        u, np.append([p0_injection], p0_injection + np.cumsum(dpdx * ds_length))
    )

    kwargs["dpdx"] = interp1d(u, np.append(dpdx, dpdx[-1]), bounds_error=True)

    kwargs["which_pass"] = interp1d(u, which_pass, bounds_error=True, kind="nearest")

    kwargs["n_elements"] = n_elements

    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    # vertices in world (x, y, z) order
    lattice.vertices = (x_world, y_world, z_world)

    fig, ax = pt.std_fig(figsize=(4, 3))
    ax.scatter(s, beta, s=1, color="blue")

    ax2 = ax.twinx()
    ax2.scatter(s, np.append(dpdx, dpdx[-1]), s=1, color="green")

    # ax.set_ylim(0,2e4)
    ax.set_xlabel("s [cm]")
    ax.set_ylabel("beta [cm]")
    ax2.set_ylabel("dpdx [GeV/cm]")
    ax.set_title("beta function along the RLA lattice")
    return lattice




#     lattice = Lattice(**lattice_dict)
#     lattice.vertices = (x_racetrack, y_racetrack)




def get_gyro_radius(E, B):
    return 3.3e2 * E / B  # cm (E in GeV, B in T)


def get_dtheta(s, R):
    return s / R


def advance_in_pos_and_momentum(x0, y0, px0, py0, dtheta, ds):
    # r = np.sqrt(x0**2 + y0**2)
    theta_p = np.arctan2(py0, px0)
    p = np.sqrt(px0**2 + py0**2)
    pxf = p * np.cos(theta_p - dtheta)
    pyf = p * np.sin(theta_p - dtheta)

    if dtheta == 0:
        return x0 + ds * np.cos(theta_p), y0 + ds * np.sin(theta_p), pxf, pyf
    else:
        R = ds / dtheta
        # coordinates centered around larmor circle
        x0_prime = R * np.cos(np.pi / 2 + theta_p)
        y0_prime = R * np.sin(np.pi / 2 + theta_p)

        xf_prime = R * np.cos(np.pi / 2 + theta_p - dtheta)
        yf_prime = R * np.sin(np.pi / 2 + theta_p - dtheta)

        dx = xf_prime - x0_prime
        dy = yf_prime - y0_prime

        return x0 + dx, y0 + dy, pxf, pyf




def create_lattice_dict_from_vertices(vertices, n_elements=None):
    """Build world-coordinate lattice interpolants from a curve of central-orbit
    vertices.

    Vertices are given in WORLD coordinates:
        x -> toward the center of the ring (+x to center),
        y -> up (0 for a planar ring),
        z -> tangential, along the beam direction of travel.
    Accept either (x, y) [planar, y=0] or (x, y, z) vertices.
    """
    if len(vertices) == 2:
        x_points, y_points = vertices
        z_points = np.zeros_like(x_points)
    elif len(vertices) == 3:
        x_points, y_points, z_points = vertices
    else:
        raise ValueError(
            "vertices must be a tuple/list of length 2 or 3: (x, y) or (x, y, z)"
        )

    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    z_points = np.asarray(z_points)

    if n_elements is None:
        n_elements = len(x_points)
    else:
        n_elements = min(max(n_elements, len(x_points)), int(1e6))

    # Compute arc-length (s) along the curve (support 3D)
    dx = np.diff(x_points)
    dy = np.diff(y_points)
    dz = np.diff(z_points)
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    s_vals = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = s_vals[-1]

    # Create interpolation functions
    fx = interp1d(s_vals, x_points, kind="linear")
    fy = interp1d(s_vals, y_points, kind="linear")
    fz = interp1d(s_vals, z_points, kind="linear")

    # New arc-length positions for smooth sampling
    s_dense = np.linspace(0, total_length, n_elements)

    x_dense = fx(s_dense)
    y_dense = fy(s_dense)
    z_dense = fz(s_dense)

    # Compute derivatives w.r.t. s (tangent vector components)
    dx_ds = np.gradient(x_dense, s_dense)
    dy_ds = np.gradient(y_dense, s_dense)
    dz_ds = np.gradient(z_dense, s_dense)

    # 3D unit tangent vector along the central orbit
    tangent_mag = np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)
    # avoid division by zero
    tangent_mag[tangent_mag == 0] = 1.0
    tx = dx_ds / tangent_mag
    ty = dy_ds / tangent_mag
    tz = dz_ds / tangent_mag

    # Tangent angle in the horizontal z-x plane, measured from +z toward +x.
    # (Used only for plotting / backward compatibility; the physics uses the full
    # 3D world tangent above.)
    angle_dense = np.arctan2(dx_ds, dz_ds)

    # Normalize to u ∈ [0, 1]
    u = np.linspace(0, 1, n_elements)

    lattice_dict = {
        "x": interp1d(u, x_dense, bounds_error=True),
        "y": interp1d(u, y_dense, bounds_error=True),
        "z": interp1d(u, z_dense, bounds_error=True),
        "s": interp1d(u, s_dense, bounds_error=True),
        "angle_of_central_p": interp1d(u, angle_dense, bounds_error=True),
        # tangent returns an (3, N) array when evaluated; keep as callable that returns stacked array
        "tangent": lambda uu: np.vstack(
            [
                interp1d(u, tx, bounds_error=True)(uu),
                interp1d(u, ty, bounds_error=True)(uu),
                interp1d(u, tz, bounds_error=True)(uu),
            ]
        ),
        "inv_s": interp1d(s_dense, u, bounds_error=True),
    }

    return lattice_dict




def get_lattice_dataframe_from_tfs(filename):
    # Initialize lists to store metadata and column data
    metadata = {}
    columns = []
    data = []

    # Open and read file
    with open(filename, "r") as file:
        for line in file:
            # Extract metadata lines
            if line.startswith("@"):
                parts = line.split()
                key = parts[1]
                value = " ".join(parts[3:]).strip('"')
                metadata[key] = value
            # Extract column names
            elif line.startswith("*"):
                columns = line.strip().split()[
                    1:
                ]  # Strip and split, and ignore the '*' character
            # Skip format line
            elif line.startswith("$"):
                continue
            # Extract data lines
            else:
                fields = [s.strip('"') for s in line.strip().split()]

                data.append(fields)

    # Validate and filter correct data rows
    correct_data = [row for row in data if len(row) == len(columns)]
    df = pd.DataFrame(correct_data, columns=columns)

    # Convert numerical columns to appropriate data types
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    n_elements = df.index.size

    new_cols = {
        "bending_magnet": df["ANGLE"] != 0,
        "GAMMAX": (1 + df["ALFX"] ** 2) / df["BETX"],
        "GAMMAY": (1 + df["ALFY"] ** 2) / df["BETY"],
        "x": np.zeros((n_elements,)),
        "y": np.zeros((n_elements,)),
        "px": np.zeros((n_elements,)),
        "py": np.zeros((n_elements,)),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Assign metadata to the DataFrame's attributes
    df.attrs = metadata

    # Propagate the survey geometry through the elements (vectorized version of
    # chaining advance_in_pos_and_momentum element by element).
    #
    # The direction angle entering element i is theta_in[i] = -sum of all
    # upstream bending kicks (the initial momentum points along +x). Zero-length
    # elements apply no kick and no displacement (matching the element-by-element
    # propagation, which skips L == 0 rows entirely). Each element then advances
    # the position by its exact chord:
    #     straight (kick = 0): (L cos(theta_in), L sin(theta_in))
    #     bend: R * (sin(theta_in) - sin(theta_in - kick),
    #                cos(theta_in - kick) - cos(theta_in)),  R = L / kick
    p0 = float(df.attrs["ENERGY"])
    L = df["L"].to_numpy(dtype=float)
    kick = np.where(L != 0.0, df["ANGLE"].to_numpy(dtype=float), 0.0)

    theta_in = np.concatenate([[0.0], -np.cumsum(kick)[:-1]])
    theta_out = theta_in - kick

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(kick != 0.0, L / np.where(kick != 0.0, kick, 1.0), 0.0)
    dx = np.where(
        kick == 0.0,
        L * np.cos(theta_in),
        R * (np.sin(theta_in) - np.sin(theta_out)),
    )
    dy = np.where(
        kick == 0.0,
        L * np.sin(theta_in),
        R * (np.cos(theta_out) - np.cos(theta_in)),
    )

    # Positions at the START of each element; momenta (|p| = p0 is conserved).
    df["x"] = np.concatenate([[0.0], np.cumsum(dx[:-1])])
    df["y"] = np.concatenate([[0.0], np.cumsum(dy[:-1])])
    df["px"] = p0 * np.cos(theta_in)
    df["py"] = p0 * np.sin(theta_in)

    return df


def plot_lattice(df):
    fig, ax = pt.std_fig(figsize=(10, 5))
    # ax.set_xlim(-220, 0)
    # ax.set_xlim(125, 150)
    # ax.set_ylim(-11, 11)
    # ax.scatter(df['x'][~df['bending_magnet']], df['y'][~df['bending_magnet']], marker='|', s=200, color='darkorange', zorder=2)
    # ax.scatter(df['x'][df['bending_magnet']], df['y'][df['bending_magnet']], marker='x', s=200, color='dodgerblue', zorder=2)
    ax.plot(df["x"], df["y"], linewidth=0.5, c="black")

    rect = patches.Rectangle(
        (-6, -6),
        12,
        12,
        linewidth=2,
        edgecolor="black",
        facecolor="None",
        hatch="///////",
    )
    ax.add_patch(rect)

    # Minimum size of linear step
    ds = 0.1
    # How tall is the magnet for x-y plane
    magnet_thickness = 1
    n_elements = df.index.size
    ds = 0.1
    # for i in list(range(1,100))+list(range(n_elements-100,n_elements)):
    for i in list(range(n_elements - 400, n_elements)):
        x, y, s = df["x"][i], df["y"][i], df["L"][i]
        px, py = df["px"][i], df["py"][i]
        dtheta = df["ANGLE"][i]
        # theta_p = np.arctan2(py, px)
        # r_arc = s / dtheta

        if df["L"][i] > 0:
            n_discrete_bend = max(int(s / ds), 30)
            x0, y0, px0, py0 = x, y, px, py
            for j in range(n_discrete_bend):
                xn, yn, pxn, pyn = advance_in_pos_and_momentum(
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
                        edgecolor="dodgerblue",
                        facecolor="dodgerblue",
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
                        edgecolor="orange",
                        facecolor="orange",
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

    ax.set_ylim(df["y"].min(), 10)
    ax.set_xlim(df["x"].min(), 0)

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    # if df['KEYWORD'][i] == 'DRIFT':
    # ax.plot([x, x+s*np.cos(theta_p)], [y, y+s*np.sin(theta_p)], color='black', linewidth=2)

    fig.savefig(
        f'plots/beam_optics/lattice_{df.attrs["ENERGY"]}_trajectory.pdf',
        dpi=500,
        bbox_inches="tight",
    )

# ==========================================================================
# Smoothed-lattice construction (moved here from the former
# mint/beam_optics.py, whose other contents were byte-identical copies of
# get_gyro_radius / get_dtheta / advance_in_pos_and_momentum above, a
# second plot_lattice, and a dead truncated variant).
# ==========================================================================

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
    Smoothed lattice (Twiss->gamma->beam envelopes + optional dispersion+sigma_delta),
    returning the legacy interface: interp1d callables in a lattice_dict.

    Key change vs the previous version:
      *Geometry (x,y,angle_of_central_p) is obtained by integrating through elements* by
      splitting each element's total bending kick into many tiny sub-kicks.

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

    # ---- Optional geometry columns
    have_xy = (x_col in df.columns) and (y_col in df.columns)
    have_p = (px_col in df.columns) and (py_col in df.columns)
    have_kick = angle_col in df.columns

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

    # Periodic interpolation knots helper (for optics fields)
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
    if n_elements <= 10:
        raise ValueError(
            "n_elements must be reasonably large (>10) for a smooth lattice."
        )

    s_grid_m = np.linspace(0.0, C, n_elements, endpoint=False)

    def interp_on_grid(arr: np.ndarray) -> np.ndarray:
        ss_ext, xx_ext = periodic_knots(arr)
        return np.interp(s_grid_m, ss_ext, xx_ext)

    # ---- Optics fields
    #
    # Interpolating the raw Twiss functions between sparse points is delicate near a
    # low-beta IP, where beta is parabolic, beta(s) = beta* + s^2/beta*, and plunges
    # by orders of magnitude between two tabulated points:
    #   * gamma is the phase-space INVARIANT -- constant in a field-free drift -- so it
    #     interpolates safely and is what we use for the DIVERGENCE, sqrt(eps*gamma).
    #     (Deriving gamma = (1+alpha^2)/beta from a linearly interpolated beta instead
    #     makes the divergence ~8x too low at the IP.)
    #   * beta (the BEAM SIZE) must respect its local parabola. We model beta inside
    #     each Twiss segment by its exact local form beta(d) = beta0 - 2 alpha0 d +
    #     gamma0 d^2 (minimum 1/gamma0 > 0, so always POSITIVE), and blend the left and
    #     right segment parabolas. This is exact in drifts, matches the Twiss table at
    #     every point, and -- unlike reconstructing beta = (1+alpha^2)/gamma from
    #     independently interpolated alpha, gamma -- does not produce spurious dips at
    #     alpha zero-crossings in the arcs.
    if have_alpha:
        # alpha interpolates cleanly (linear in drifts, smooth in arcs); used for the
        # x-x' covariance and for the beta segment-parabola blend.
        alfx = interp_on_grid(df["ALFX"].to_numpy(dtype=float))
        alfy = interp_on_grid(df["ALFY"].to_numpy(dtype=float))
    else:
        alfx = None
        alfy = None

    if have_gamma:
        gamx = interp_on_grid(df["GAMMAX"].to_numpy(dtype=float))
        gamy = interp_on_grid(df["GAMMAY"].to_numpy(dtype=float))
    else:
        # No gamma column: derive it (interpolating beta is imperfect in low-beta
        # drifts, but it is the best available without the invariant gamma).
        gamx = (1.0 + alfx**2) / interp_on_grid(df["BETX"].to_numpy(dtype=float))
        gamy = (1.0 + alfy**2) / interp_on_grid(df["BETY"].to_numpy(dtype=float))

    def beta_twiss_on_grid(beta_col: str, alpha_col: str) -> np.ndarray:
        """Interpolate beta(s) onto s_grid_m using the local Twiss parabola in each
        segment: beta(d) = beta0 - 2 alpha0 d + gamma0 d^2. Positive by construction,
        exact in drifts, and dip-free at alpha zero-crossings."""
        bt = df[beta_col].to_numpy(dtype=float)
        at = df[alpha_col].to_numpy(dtype=float)
        ss = np.mod(s, C)
        order = np.argsort(ss)
        ss, bt, at = ss[order], bt[order], at[order]
        keep = np.concatenate([[True], np.diff(ss) > 0])  # drop duplicate s
        ss, bt, at = ss[keep], bt[keep], at[keep]
        gt = (1.0 + at**2) / bt
        # periodic wrap so grid points near 0 and C are bracketed
        ss_ext = np.concatenate([[ss[-1] - C], ss, [ss[0] + C]])
        bt_ext = np.concatenate([[bt[-1]], bt, [bt[0]]])
        at_ext = np.concatenate([[at[-1]], at, [at[0]]])
        gt_ext = np.concatenate([[gt[-1]], gt, [gt[0]]])
        j = np.clip(np.searchsorted(ss_ext, s_grid_m) - 1, 0, len(ss_ext) - 2)
        sL, sR = ss_ext[j], ss_ext[j + 1]
        dL, dR = s_grid_m - sL, s_grid_m - sR
        beta_L = bt_ext[j] - 2.0 * at_ext[j] * dL + gt_ext[j] * dL**2
        beta_R = bt_ext[j + 1] - 2.0 * at_ext[j + 1] * dR + gt_ext[j + 1] * dR**2
        t = (s_grid_m - sL) / (sR - sL)
        return (1.0 - t) * beta_L + t * beta_R

    if have_alpha:
        betx = beta_twiss_on_grid("BETX", "ALFX")
        bety = beta_twiss_on_grid("BETY", "ALFY")
    else:
        betx = interp_on_grid(df["BETX"].to_numpy(dtype=float))
        bety = interp_on_grid(df["BETY"].to_numpy(dtype=float))

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

    # ---- Beam envelopes
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

    # ---- Geometry via element subdivision integration
    # This is the key change: we do NOT linearly interpolate sparse (x,y) points.

    def _integrate_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Integrate x(s), y(s), theta(s) by splitting each element's total kick into sub-kicks.

        Requires x,y and ANGLE (kick). Initial direction comes from px,py if present; otherwise
        it is estimated from the first two geometry points.
        """

        if not (have_xy and have_kick):
            return None

        # Initial position
        x0 = float(df.iloc[0][x_col])
        y0 = float(df.iloc[0][y_col])

        # Initial direction (unit vector)
        if have_p:
            px0 = float(df.iloc[0][px_col])
            py0 = float(df.iloc[0][py_col])
            pnorm = float(np.hypot(px0, py0))
            if np.isfinite(pnorm) and pnorm > 0.0:
                px0, py0 = px0 / pnorm, py0 / pnorm
            else:
                px0, py0 = 1.0, 0.0
        else:
            # estimate from next available geometry point
            if len(df) > 1:
                dx0 = float(df.iloc[1][x_col]) - x0
                dy0 = float(df.iloc[1][y_col]) - y0
                th0 = float(np.arctan2(dy0, dx0))
            else:
                th0 = 0.0
            px0, py0 = float(np.cos(th0)), float(np.sin(th0))

        # Target step size (meters)
        ds_target = C / n_elements

        xs = [x0]
        ys = [y0]
        ss = [0.0]
        ths = [float(np.arctan2(py0, px0))]

        x, y, px, py = x0, y0, px0, py0
        s_acc = 0.0

        for i in range(len(df)):
            ell = float(df.iloc[i]["L"])
            if (not np.isfinite(ell)) or ell <= 0.0:
                continue

            dtheta = float(df.iloc[i][angle_col])
            if not np.isfinite(dtheta):
                dtheta = 0.0

            n_sub = int(np.ceil(ell / max(ds_target, 1e-12)))
            n_sub = max(n_sub, 1)
            ds_sub = ell / n_sub
            dth_sub = dtheta / n_sub

            for _ in range(n_sub):
                x, y, px, py = advance_in_pos_and_momentum(
                    x, y, px, py, dth_sub, ds_sub
                )
                s_acc += ds_sub
                xs.append(x)
                ys.append(y)
                ss.append(s_acc)
                ths.append(float(np.arctan2(py, px)))

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        ss = np.asarray(ss, dtype=float)
        ths = np.asarray(ths, dtype=float)

        if len(ss) < 2 or ss[-1] <= 0.0:
            return None

        # Small mismatch between sum(L) and inferred C can happen; normalize to C.
        ss *= C / ss[-1]

        # Periodic endpoint to allow interpolation at/near C.
        # We force closure by repeating the first point at s=C.
        xs = np.concatenate([xs, xs[:1]])
        ys = np.concatenate([ys, ys[:1]])
        ss = np.concatenate([ss, np.array([C])])

        # Unwrap theta to avoid jumps during interpolation
        th_unw = np.unwrap(ths)
        # Append a consistent endpoint for theta; the value at C should match 0-turn closure
        th_unw = np.concatenate([th_unw, th_unw[:1] + (th_unw[-1] - th_unw[0])])

        x_m = np.interp(s_grid_m, ss, xs)
        y_m = np.interp(s_grid_m, ss, ys)
        theta_m = np.interp(s_grid_m, ss, th_unw)

        # Wrap back to [-pi, pi)
        theta_m = (theta_m + np.pi) % (2.0 * np.pi) - np.pi

        return x_m, y_m, theta_m

    geom = _integrate_geometry()

    if geom is None:
        # Fallback: interpolate sparse geometry if we cannot integrate
        if have_xy:
            x_m = interp_on_grid(df[x_col].to_numpy(dtype=float))
            y_m = interp_on_grid(df[y_col].to_numpy(dtype=float))
        else:
            x_m = np.zeros_like(s_grid_m)
            y_m = np.zeros_like(s_grid_m)

        if have_p:
            px = interp_on_grid(df[px_col].to_numpy(dtype=float))
            py = interp_on_grid(df[py_col].to_numpy(dtype=float))
            angle = np.arctan2(py, px)
        else:
            # last resort: tangent from geometry
            dxf = np.gradient(x_m, s_grid_m)
            dyf = np.gradient(y_m, s_grid_m)
            angle = np.arctan2(dyf, dxf)
    else:
        x_m, y_m, angle = geom

    # Rotation/centering if requested
    if rotated:
        mid = len(s_grid_m) // 2
        x0, y0 = x_m[mid], y_m[mid]

        # Use a wider window for robust tangent estimation
        k = 500
        i0 = max(0, mid - k)
        i1 = min(len(x_m) - 1, mid + k)

        dx = x_m[i1] - x_m[i0]
        dy = y_m[i1] - y_m[i0]
        theta_tan = float(np.arctan2(dy, dx))

        ca, sa = np.cos(-theta_tan), np.sin(-theta_tan)
        xr = (x_m - x0) * ca - (y_m - y0) * sa
        yr = (x_m - x0) * sa + (y_m - y0) * ca
        x_m, y_m = xr, yr

        # Keep angle consistent with rotated geometry
        angle = angle - theta_tan
        angle = (angle + np.pi) % (2.0 * np.pi) - np.pi

        # Optional: force the direction at the center to point along +x (not -x)
        if np.mean(np.diff(x_m[mid - 500 : mid + 500])) < 0:
            x_m *= -1
            angle = np.arctan2(np.gradient(y_m, s_grid_m), np.gradient(x_m, s_grid_m))

    # NOTE: if_sublattice closure is geometry-specific and is not implemented here.
    # Keep the flag for interface compatibility.
    _ = if_sublattice

    # ---- Map the integrated (drawing-plane) geometry to WORLD coordinates.
    # The MAD-X survey geometry lives in a horizontal plane: x_m is the
    # longitudinal coordinate (along the beam) and y_m the transverse one, with
    # `angle` the tangent angle measured from x_m toward y_m. In world axes:
    #   z (tangential) = x_m,  x (toward center) = s_transverse * y_m,  y (up) = 0.
    # Choose the transverse sign so the ring bulges toward +x (center at +x).
    s_transverse = 1.0 if float(np.mean(y_m)) >= 0.0 else -1.0

    z_world_m = x_m
    x_world_m = s_transverse * y_m
    # World tangent components (unit): tz = cos(angle), tx = s * sin(angle), ty = 0.
    tx_world = s_transverse * np.sin(angle)
    ty_world = np.zeros_like(angle)
    tz_world = np.cos(angle)
    # World tangent angle in the horizontal z-x plane, from +z toward +x.
    angle_world = np.arctan2(tx_world, tz_world)

    s_cm = s_grid_m * const.m_to_cm
    x_cm = x_world_m * const.m_to_cm
    y_cm = np.zeros_like(x_cm)
    z_cm = z_world_m * const.m_to_cm

    beamsize_x_cm = sigx_tot * const.m_to_cm
    beamsize_y_cm = sigy * const.m_to_cm

    # beamdiv in the old interface is an angle in rad.
    beamdiv_x = np.arctan(sigxp_tot)
    beamdiv_y = np.arctan(sigyp)

    # ---- Build interpolation objects
    lattice_dict: dict[str, object] = {}
    u = np.linspace(0.0, 1.0, len(s_cm))

    lattice_dict["x"] = interp1d(u, x_cm, bounds_error=False, fill_value=None)
    lattice_dict["y"] = interp1d(u, y_cm, bounds_error=False, fill_value=None)
    lattice_dict["z"] = interp1d(u, z_cm, bounds_error=False, fill_value=None)
    lattice_dict["s"] = interp1d(u, s_cm, bounds_error=False, fill_value=None)

    lattice_dict["angle_of_central_p"] = interp1d(
        u, angle_world, bounds_error=False, fill_value=None
    )

    # World unit tangent [tx, ty, tz] of the central orbit.
    _tx_i = interp1d(u, tx_world, bounds_error=False, fill_value=None)
    _ty_i = interp1d(u, ty_world, bounds_error=False, fill_value=None)
    _tz_i = interp1d(u, tz_world, bounds_error=False, fill_value=None)
    lattice_dict["tangent"] = lambda uu, _tx_i=_tx_i, _ty_i=_ty_i, _tz_i=_tz_i: np.vstack(
        [_tx_i(uu), _ty_i(uu), _tz_i(uu)]
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

    # ---- Beam momentum
    lattice_dict["beam_p0"] = np.sqrt(float(df.attrs["ENERGY"]) ** 2 - const.m_mu**2)

    # ---- Extras for MC beam sampling (also interpolants)
    lattice_dict["betx"] = interp1d(u, betx, bounds_error=False, fill_value=None)
    lattice_dict["bety"] = interp1d(u, bety, bounds_error=False, fill_value=None)
    lattice_dict["gamx"] = interp1d(u, gamx, bounds_error=False, fill_value=None)
    lattice_dict["gamy"] = interp1d(u, gamy, bounds_error=False, fill_value=None)

    if have_alpha:
        lattice_dict["alfx"] = interp1d(u, alfx, bounds_error=False, fill_value=None)
        lattice_dict["alfy"] = interp1d(u, alfy, bounds_error=False, fill_value=None)

    lattice_dict["cov_x_xp"] = interp1d(
        u, cov_x_xp, bounds_error=False, fill_value=None
    )
    lattice_dict["cov_y_yp"] = interp1d(
        u, cov_y_yp, bounds_error=False, fill_value=None
    )

    lattice_dict["sigma_xp"] = interp1d(
        u, sigxp_tot, bounds_error=False, fill_value=None
    )
    lattice_dict["sigma_yp"] = interp1d(u, sigyp, bounds_error=False, fill_value=None)

    lattice_dict["length_cm"] = float(C * const.m_to_cm)
    lattice_dict["length_m"] = float(C)
    lattice_dict["emittance_RMS"] = float(eps)
    lattice_dict["midpoint"] = bool(midpoint)
    lattice_dict["include_dispersion"] = bool(include_dispersion)
    lattice_dict["sigma_delta"] = None if sigma_delta is None else float(sigma_delta)

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
