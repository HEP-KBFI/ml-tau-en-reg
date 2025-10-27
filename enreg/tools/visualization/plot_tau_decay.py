import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math


class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """

    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0], y[0], " ", **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == " ":
                ##make this an invisible 'a':
                t = mtext.Text(0, 0, "a")
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0, 0, c, fontsize=13, **kwargs)

            # resetting unnecessary arguments
            t.set_ha("center")
            t.set_rotation(0)
            t.set_zorder(self.__zorder + 1)

            self.__Characters.append((c, t))
            axes.add_artist(t)

    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c, t in self.__Characters:
            t.set_zorder(self.__zorder + 1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        # preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w) / (figH * h)) * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

        # points of the curve in figure coordinates:
        x_fig, y_fig = (
            np.array(l)
            for l in zip(
                *self.axes.transData.transform(
                    [(i, j) for i, j in zip(self.__x, self.__y)]
                )
            )
        )

        # point distances in figure coordinates
        x_fig_dist = x_fig[1:] - x_fig[:-1]
        y_fig_dist = y_fig[1:] - y_fig[:-1]
        r_fig_dist = np.sqrt(x_fig_dist**2 + y_fig_dist**2)

        # arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)

        # angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        rel_pos = 10
        for c, t in self.__Characters:
            # finding the width of c:
            t.set_rotation(0)
            t.set_va("center")
            bbox1 = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            # ignore all letters that don't fit:
            if rel_pos + w / 2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != " ":
                t.set_alpha(1.0)

            # finding the two data points between which the horizontal
            # center point of the character will be situated
            # left and right indices:
            il = np.where(rel_pos + w / 2 >= l_fig)[0][-1]
            ir = np.where(rel_pos + w / 2 <= l_fig)[0][0]

            # if we exactly hit a data point:
            if ir == il:
                ir += 1

            # how much of the letter width was needed to find il:
            used = l_fig[il] - rel_pos
            rel_pos = l_fig[il]

            # relative distance between il and ir where the center
            # of the character will be
            fraction = (w / 2 - used) / r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il] + fraction * (self.__x[ir] - self.__x[il])
            y = self.__y[il] + fraction * (self.__y[ir] - self.__y[il])

            # getting the offset when setting correct vertical alignment
            # in data coordinates
            t.set_va(self.get_va())
            bbox2 = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0] - bbox1d[0])

            # the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array(
                [
                    [math.cos(rad), math.sin(rad) * aspect],
                    [-math.sin(rad) / aspect, math.cos(rad)],
                ]
            )

            ##computing the offset vector of the rotated character
            drp = np.dot(dr, rot_mat)

            # setting final position and rotation:
            t.set_position(np.array([x, y]) + drp)
            t.set_rotation(degs[il])

            t.set_va("center")
            t.set_ha("center")

            # updating rel_pos to right edge of character
            rel_pos += w - used


# def curveText(text, height, minTheta, maxTheta, ax):
#     interval = np.arange(minTheta, maxTheta, .022)
#     if( maxTheta <= np.pi):
#         progression = interval[::-1]
#         rotation = interval[::-1] - np.arctan(np.tan(np.pi/2))
#     else:
#         progression = interval
#         rotation = interval - np.arctan(np.tan(np.pi/2)) - np.pi

#     ## Render each letter individually
#     for i, rot, t in zip(progression, rotation, text):
#         ax.text(i, height, t, fontsize=11,rotation=np.degrees(rot), ha='center', va='center')


hadronic_decays = [1.463, 11.51, 25.93, 10.81, 9.80, 4.76 + 0.517]
hadronic_explode = [0.1] * len(hadronic_decays)
hadronic_labels = [
    "Rare",
    r"$h^{\pm} \nu_{\tau}$",  # h
    r"$h^{\pm} \pi_{0} \nu_{\tau}$",  # h + pi0
    r"$h^{\pm} \geq 2 \pi_{0} + \nu_{\tau}$",  # h + 2 pi0 -> Should replace with h + >= 2pi0
    r"$h^{\pm} h^{\mp} h^{\pm} \nu_{\tau}$",  # h h h
    r"$h^{\pm} h^{\mp} h^{\pm} \pi_{0} \nu_{\tau}$",  # hhh + pi0
    # r"$h^{\pm} h^{\mp} h^{\pm} \geq 2\pi_{0} \nu_{\tau}$",  # hhh + >= 2pi0
]
hadronic_colors = [
    plt.cm.Greens(0.5),
    plt.cm.Blues(0.4),
    plt.cm.Blues(0.5),
    plt.cm.Blues(0.6),
    plt.cm.Reds(0.4),
    plt.cm.Reds(0.6),
]


leptonic_decays = [17.82, 17.39]
leptonic_explode = [0.0] * len(leptonic_decays)
leptonic_labels = [
    r"$e^{-} \bar{\nu_{e}} \nu_{\tau}$",
    r"$\mu^{-} \bar{\nu_{\mu}} \nu_{\tau}$",
]
leptonic_colors = [plt.cm.Greys(0.5), plt.cm.Greys(0.6)]


subgroup_names = hadronic_labels + leptonic_labels
subgroup_size = hadronic_decays + leptonic_decays
subgroup_explode = hadronic_explode + leptonic_explode
subgroup_colors = hadronic_colors + leptonic_colors

group_names = ["Hadronic decays", "Leptonic decays"]
group_size = [sum(hadronic_decays), sum(leptonic_decays)]
group_explode = [0.0, 0.0]
group_colors = [plt.cm.Blues(0.8), plt.cm.Greys(0.8)]

fig, ax = plt.subplots()
# ax.axis('equal')

inner_circle = ax.pie(
    group_size,
    autopct="%1.1f%%",
    pctdistance=0.8,
    radius=1.0,
    # labels=group_names,
    explode=group_explode,
    labeldistance=0.7,
    colors=group_colors,
    textprops={"fontsize": 13, "color": "black"},
)[0]
plt.setp(inner_circle, width=0.4, edgecolor="black")

outer_circle = ax.pie(
    subgroup_size,
    autopct="%1.1f%%",
    pctdistance=0.87,
    radius=1.4,
    labels=subgroup_names,
    explode=subgroup_explode,
    colors=subgroup_colors,
    textprops={"fontsize": 13, "color": "black"},
)[0]

plt.setp(outer_circle, width=0.4, edgecolor="black")

plt.margins(5, 5)
N = 100

x_h = 0.4 * np.cos(np.linspace(1.2 * np.pi, 0, N))
y_h = 0.4 * np.sin(np.linspace(1.2 * np.pi, 0, N))


x_l = 0.5 * np.cos(np.linspace(1.33 * np.pi, 2 * np.pi, N))
y_l = 0.5 * np.sin(np.linspace(1.33 * np.pi, 2 * np.pi, N))


CurvedText(x_h, y_h, text="Hadronic decays", va="bottom", axes=ax, color="black")
CurvedText(x_l, y_l, text="Leptonic decays", va="bottom", axes=ax, color="black")
plt.text(-0.08, -0.08, r"$\tau$", fontsize=30)


plt.savefig("tau_decays.pdf", format="pdf", bbox_inches="tight", transparent=True)
