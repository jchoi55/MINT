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
        self.name = lattice_dict.pop("name", 1)
        self.short_name = lattice_dict.pop("short_name", 1)
        self.n_elements = lattice_dict.pop("n_elements", 1)

        for key, value in lattice_dict.items():
            print("Setting additional", key, "to", value)
            self.__setattr__(key, value)
        if lattice_dict.keys():
            print(
                "Warning! The following keys were not recognized and will be ignored:",
                lattice_dict.keys(),
            )


def create_racetrack_lattice(
    straight_length=100e2, total_length=300e2, n_elements=10_000, **kwargs
):

    racetrack_radius = (
        (total_length - 2 * straight_length) / 2 / np.pi
    )  # radius of the semicircles
    n_points = 300
    # Semicircle on the left
    theta_left = np.linspace(3 * np.pi / 2, np.pi / 2, n_points)
    x_left = -straight_length / 2 + racetrack_radius * np.cos(theta_left)
    y_left = racetrack_radius * np.sin(theta_left)

    # Semicircle on the right
    theta_right = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    x_right = straight_length / 2 - racetrack_radius * np.cos(theta_right)
    y_right = racetrack_radius * np.sin(theta_right)

    # Top straight
    x_top = np.linspace(-straight_length / 2, straight_length / 2, n_points)
    y_top = np.full_like(x_top, racetrack_radius)

    # Bottom straight
    x_bottom = np.linspace(straight_length / 2, -straight_length / 2, n_points)
    y_bottom = np.full_like(x_bottom, -racetrack_radius)

    # Concatenate all segments. The curve is built in a 2D plane where the first
    # array is longitudinal (along the straights) and the second is transverse.
    longitudinal = np.concatenate([x_left, x_top, x_right, x_bottom])
    transverse = np.concatenate([y_left, y_top, y_right, y_bottom])
    # Shift so the top straight (the IP straight) sits at transverse = 0; the ring
    # then extends toward negative transverse.
    transverse -= np.max(transverse)

    # Map to WORLD coordinates: z = longitudinal (tangential), x = toward the ring
    # center (+x points to the center, so flip the transverse sign), y = 0 (planar).
    z_world = longitudinal
    x_world = -transverse
    y_world = np.zeros_like(longitudinal)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_world, y_world, z_world), n_elements=n_elements
    )
    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    # vertices in world (x, y, z) order
    lattice.vertices = (x_world, y_world, z_world)

    return lattice


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


def append_lattices(
    lattice1,
    lattice2,
    vert_shift,
    hor_shift,
    RLA=False,  # determine if we are adding an RLA lattice (if so, we need to handle which_pass differently)
    p0_injection=0.255,
    Nmu_per_bunch_inj=2e12,
    **kwargs,
):
    # Append lattices so they go from one after another in space and time

    # Extract vertex arrays in WORLD (x, y, z) order:
    #   xw -> transverse (toward center), yw -> vertical (up), zw -> longitudinal.
    xw = lattice1.vertices[0]
    yw = lattice1.vertices[1]
    zw = lattice1.vertices[2] if len(lattice1.vertices) > 2 else np.zeros_like(xw)

    xw2 = lattice2.vertices[0]
    yw2 = lattice2.vertices[1]
    zw2 = lattice2.vertices[2] if len(lattice2.vertices) > 2 else np.zeros_like(xw2)

    ds_length1 = get_s_element(xw, yw, zw)
    s1 = np.concatenate([[0], np.cumsum(ds_length1)])
    u1 = s1 / s1[-1]

    ds_length2 = get_s_element(xw2, yw2, zw2)
    s2 = np.concatenate([[0], np.cumsum(ds_length2)])
    u2 = s2 / s2[-1]

    # Shift second lattice so it follows the first in space. The lattices are
    # chained along the longitudinal (world z) direction; hor_shift adds an extra
    # longitudinal gap and vert_shift an extra vertical (world y) offset.
    xw2_shifted = xw2 + xw[-1]
    yw2_shifted = yw2 + yw[-1] + vert_shift
    zw2_shifted = zw2 + zw[-1] + hor_shift

    # Create a single straight transition connecting the end of lattice1 to the
    # start of the shifted lattice2, interpolated in all three world coordinates.
    Ntrans = 10_000
    trans_x = np.linspace(xw[-1], xw2_shifted[0], Ntrans)
    trans_y = np.linspace(yw[-1], yw2_shifted[0], Ntrans)
    trans_z = np.linspace(zw[-1], zw2_shifted[0], Ntrans)

    xw = np.concatenate((xw, trans_x, xw2_shifted))
    yw = np.concatenate((yw, trans_y, yw2_shifted))
    zw = np.concatenate((zw, trans_z, zw2_shifted))

    dpdx1 = lattice1.dpdx(u1)
    dpdx2 = lattice2.dpdx(u2)  # still local
    dpdxtrans = np.full_like(trans_x, 0.0)
    dpdx = np.concatenate((dpdx1, dpdxtrans, dpdx2))

    # compute ds for transition using 3D segments
    ds_trans = get_s_element(trans_x, trans_y, trans_z)
    strans = np.concatenate([[0], np.cumsum(ds_trans)])

    total_length = s1[-1] + strans[-1] + s2[-1]
    s_total = np.concatenate((s1, strans + s1[-1], s2 + strans[-1] + s1[-1]))
    u_total = s_total / total_length
    ds_total = np.concatenate(
        (ds_length1, np.array([0]), ds_trans, np.array([0]), ds_length2)
    )

    kwargs["beam_p0"] = interp1d(
        u_total,
        np.append([p0_injection], (p0_injection + np.cumsum(dpdx[:-1] * ds_total))),
    )

    kwargs["dpdx"] = interp1d(u_total, dpdx)

    if RLA == True:
        # Handle which_pass: offset lattice2's passes so they follow lattice1
        which_pass_1 = lattice1.which_pass(u1)
        which_pass_2 = lattice2.which_pass(u2)
        max_pass_1 = np.max(which_pass_1)
        which_pass_2_offset = which_pass_2 + max_pass_1
        which_pass_trans = np.full_like(trans_x, which_pass_1[-1])
        which_pass_combined = np.concatenate(
            (which_pass_1, which_pass_trans, which_pass_2_offset)
        )
        kwargs["which_pass"] = interp1d(
            u_total,
            which_pass_combined,
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",
        )
    else:
        # if we are combining a linac and RLA, the pass is 0 for the linac and then RLA's pass number
        which_pass_2 = lattice2.which_pass(u2)
        linac_pass = np.full_like(u1, 0)
        which_pass_trans = np.full_like(trans_x, 0)
        which_pass_combined = np.concatenate(
            (linac_pass, which_pass_trans, which_pass_2)
        )
        kwargs["which_pass"] = interp1d(
            u_total,
            which_pass_combined,
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",
        )

    lattice_dict = create_lattice_dict_from_vertices(
        (xw, yw, zw), n_elements=len(xw)
    )

    kwargs["n_elements"] = lattice1.n_elements + lattice2.n_elements

    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    # vertices in world (x, y, z) order: x = transverse (toward center),
    # y = vertical, z = longitudinal. Vertical shifts (world y) are preserved.
    lattice.vertices = (xw, yw, zw)

    # combine lattice names
    lattice.name = lattice1.name + "+" + lattice2.name
    lattice.short_name = lattice1.short_name + "+" + lattice2.short_name

    # use lattice design parameters from the first lattice
    lattice.duty_factor = lattice1.duty_factor
    lattice.bunch_multiplicity = lattice1.bunch_multiplicity
    lattice.finj = lattice1.finj

    if Nmu_per_bunch_inj == lattice1.Nmu_per_bunch:
        lattice.Nmu_per_bunch = lattice1.Nmu_per_bunch
        print("is same; lattice nmu per bunch is:", lattice.Nmu_per_bunch)
    else:
        lattice.Nmu_per_bunch = Nmu_per_bunch_inj
        print("is different; lattice nmu per bunch is:", lattice.Nmu_per_bunch)

    return lattice


#     lattice = Lattice(**lattice_dict)
#     lattice.vertices = (x_racetrack, y_racetrack)


def create_elliptical_lattice(
    length_minor, length_major, center=(0, 0), n_elements=10_000, **kwargs
):
    """Create an elliptical lattice.

    Args:
        length_minor (_type_): _description_
        length_major (_type_): _description_
        n_elements (_type_, optional): _description_. Defaults to 10_000.
    """
    theta = np.linspace(0, 2 * np.pi, n_elements)
    # Ellipse drawn in the horizontal plane: major axis -> longitudinal (world z),
    # minor axis -> transverse (world x), world y = 0 (planar).
    z_world = center[0] + length_major * np.cos(theta)
    x_world = center[1] + length_minor * np.sin(theta)
    y_world = np.zeros_like(z_world)

    lattice_dict = create_lattice_dict_from_vertices(
        (x_world, y_world, z_world), n_elements=n_elements
    )

    # Any additional user-input
    lattice_dict.update(kwargs)

    lattice = Lattice(**lattice_dict)
    # vertices in world (x, y, z) order
    lattice.vertices = (x_world, y_world, z_world)

    return lattice


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


def advance_in_pos(x0, y0, theta_0, dtheta, ds):
    theta_mid = theta_0 + dtheta / 2
    x_new = x0 + ds * np.cos(theta_mid)
    y_new = y0 + ds * np.sin(theta_mid)
    return x_new, y_new


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


def concatenate_lattice_dfs(df1, df2, x_shift=None, y_shift=None):
    """Concatenate two lattice dataframes by shifting the x values of the second dataframe.


    Parameters
    ----------
    df1 : pd.DataFrame
        The first lattice dataframe.
    df2 : pd.DataFrame
        The second lattice dataframe to be shifted and concatenated.
    x_shift : float, optional
        The amount to shift the x values of df2. If None, uses the max x value from df1.
    y_shift : float, optional
        The amount to shift the y values of df2. If None, uses the max y value from df1.

    NOTE: the attrs of df1 are preserved; attrs of df2 are discarded.

    Returns
    -------
    pd.DataFrame
        The concatenated lattice dataframe.
    """

    # Concatenate the two dataframes with shifted x values
    df1 = df1.copy()
    df2 = df2.copy()

    # Get the max x value from the straight lattice to use for shifting
    if x_shift is None:
        x_shift = df1["x"].max()
    if y_shift is None:
        y_shift = df1["y"].max()

    # Shift x, y, and S values
    df2["x"] = df2["x"] + x_shift
    df2["y"] = df2["y"] + y_shift
    # Shift S coordinate to continue from where df1 ended
    s_shift = df1["S"].max()
    df2["S"] = df2["S"] + s_shift

    # Concatenate the dataframes
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined.attrs = df1.attrs

    del df1, df2
    return df_combined


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

    # Put the initial conditions for the particle momentum
    df.loc[0, "px"] = float(df.attrs["ENERGY"])

    # for i in range(0, n_elements):
    for i in range(0, n_elements - 1):
        if df["L"][i] == 0:
            (
                df.loc[i + 1, "x"],
                df.loc[i + 1, "y"],
                df.loc[i + 1, "px"],
                df.loc[i + 1, "py"],
            ) = (df["x"][i], df["y"][i], df["px"][i], df["py"][i])
            continue
        else:
            (
                df.loc[i + 1, "x"],
                df.loc[i + 1, "y"],
                df.loc[i + 1, "px"],
                df.loc[i + 1, "py"],
            ) = advance_in_pos_and_momentum(
                df["x"][i],
                df["y"][i],
                df["px"][i],
                df["py"][i],
                dtheta=df["ANGLE"][i],
                ds=df["L"][i],
            )

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
