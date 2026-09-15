import shutil

from cycler import cycler
import numpy as np

from math import log10, floor

import matplotlib
import colorsys
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.spatial.distance import pdist, squareform


###########################
fsize = 11
fsize_annotate = 10

std_figsize = (1.2 * 3.7, 1.3 * 2.3617)
std_axes_form = [0.18, 0.16, 0.79, 0.76]

rcparams = {
    "axes.labelsize": fsize,
    "xtick.labelsize": fsize,
    "ytick.labelsize": fsize,
    "figure.figsize": std_figsize,
    "legend.frameon": False,
    "legend.loc": "best",
}
# Use LaTeX text rendering when a LaTeX installation is available; otherwise
# fall back to matplotlib's mathtext so plots still render everywhere.
if shutil.which("latex"):
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    rc("text", usetex=True)
if shutil.which("latex"):
    # When LaTeX is available, prefer Computer Modern but include fallbacks.
    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"]})
else:
    # Avoid Matplotlib searching for Computer Modern when LaTeX is not available;
    # prefer commonly installed serif fonts to prevent findfont warnings/errors.
    rc("font", **{"family": "serif", "serif": ["DejaVu Serif", "Times New Roman", "serif"]})
matplotlib.rcParams["hatch.linewidth"] = 0.3

rcParams.update(rcparams)

cblind_safe_wheel = [
    "#3f90da",
    "#bd1f01",
    "#ffa90e",
    "#94a4a2",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

plt.rcParams["axes.prop_cycle"] = cycler(color=cblind_safe_wheel)


##########################
#
def std_fig(ax_form=std_axes_form, figsize=std_figsize, rasterized=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_form, rasterized=rasterized)
    ax.patch.set_alpha(0.0)
    return fig, ax










# Function to find the path that connects points in order of closest proximity


def get_ordered_closed_region(points, logx=False, logy=False):
    xraw, yraw = points

    # check for nans
    if np.isnan(points).sum() > 0:
        raise ValueError("NaN's were found in input data. Cannot order the contour.")

    # check for repeated x-entries -- remove them
    # x, mask_diff = np.unique(x, return_index=True)
    # y = y[mask_diff]

    if logy:
        if (yraw == 0).any():
            raise ValueError("y values cannot contain any zeros in log mode.")
        yraw = np.log10(yraw)
    if logx:
        if (xraw == 0).any():
            raise ValueError("x values cannot contain any zeros in log mode.")
        xraw = np.log10(xraw)

    # Transform to unit square space:
    xmin, xmax = np.min(xraw), np.max(xraw)
    ymin, ymax = np.min(yraw), np.max(yraw)

    x = (xraw - xmin) / (xmax - xmin)
    y = (yraw - ymin) / (ymax - ymin)

    points = np.array([x, y]).T
    # points_s     = (points - points.mean(0))
    # angles       = np.angle((points_s[:,0] + 1j*points_s[:,1]))
    # points_sort  = points_s[angles.argsort()]
    # points_sort += points.mean(0)

    # if np.isnan(points_sort).sum()>0:
    #     raise ValueError("NaN's were found in sorted points. Cannot order the contour.")
    # # print(points.mean(0))
    # # return points_sort
    # tck, u = splprep(points_sort.T, u=None, s=0.0, per=0, k=1)
    # # u_new = np.linspace(u.min(), u.max(), len(points[:,0]))
    # x_new, y_new = splev(u, tck, der=0)
    # # x_new, y_new = splev(u_new, tck, der=0)
    dist_matrix = squareform(pdist(points))

    # Set diagonal to a large number to avoid self-loop
    np.fill_diagonal(dist_matrix, np.inf)

    # Start from the first point
    current_point = 0
    path = [current_point]

    # Find the nearest neighbor of each point
    while len(path) < len(points):
        # Find the nearest point that is not already in the path
        nearest = np.argmin(dist_matrix[current_point])
        # Add the nearest point to the path
        path.append(nearest)
        # Update the current point
        current_point = nearest
        # Mark the visited point so it's not revisited
        dist_matrix[:, current_point] = np.inf

    # Return the ordered path indices and the corresponding points
    x_new, y_new = points[path].T

    x_new = x_new * (xmax - xmin) + xmin
    y_new = y_new * (ymax - ymin) + ymin

    if logx:
        x_new = 10 ** (x_new)
    if logy:
        y_new = 10 ** (y_new)
    return x_new, y_new




def round_sig(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def sci_notation(
    num,
    sig_digits=1,
    precision=None,
    exponent=None,
    notex=False,
    optional_sci=False,
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num != 0:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), sig_digits)
        if coeff == 10:
            coeff = 1
            exponent += 1
        if precision is None:
            precision = sig_digits

        if optional_sci and np.abs(exponent) < optional_sci:
            string = rf"{round_sig(num, precision)}"
        else:
            string = r"{0:.{2}f}\times 10^{{{1:d}}}".format(coeff, exponent, precision)

        if notex:
            return string
        else:
            return f"${string}$"

    else:
        return r"0"


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


###########################


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
