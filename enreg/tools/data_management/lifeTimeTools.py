"""
Helper functions tom get additional training inputs such as impact parameters
and calometric energy deposits (TODO). For the impact parameters a linear extrapolation
of the helix at the refference point is used. The definitions of track parameters
(d0, z0, phi, tanL, Omega) are taken from:
[1] https://flc.desy.de/lcnotes/notes/localfsExplorer_read?currentPath=/afs/desy.de/group/flc/lcnotes/LC-DET-2006-004.pdf
Math for finding the PCA from:
[2] https://www-h1.desy.de/psfiles/theses/h1th-134.pdf
[3] https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

Author: Torben Lange (KBFI), Laurits Tani (KBFI)
Date: 08.03.2022
"""

import numpy as np
import math
import awkward as ak

import vector

""" Control function to check if PCA calculation works,
    calculates the distance of any point on the track
    (full helix equation) to the PV.
    Compare to [1] and [2] for more details.

           x0' + 1/kappa*sin(phi+kappa*s)
    v(s) = y0' - 1/kappa*cos(phi+kappa*s)
           z0' + s*cot(theta)
    kappa = omega, z0' = z0+zr, phi = phi0, cot(theta)=tan(lambda), x0' = xr + cos(pi/2-phi0)*(d0 - 1/omega)
   y0'= yr - sin(pi/2-phi0)*(d0 - 1/omega)
"""


def distHelix(trackP, vertex, s):
    x0_dash = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * (trackP["d0"] - 1.0 / trackP["omega"])
    y0_dash = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * (trackP["d0"] - 1.0 / trackP["omega"])
    z0_dash = trackP["z0"] + trackP["zr"]
    dx = x0_dash + 1.0 / trackP["omega"] * math.sin(trackP["phi0"] + trackP["omega"] * s) - vertex[0]
    dy = y0_dash - 1.0 / trackP["omega"] * math.cos(trackP["phi0"] + trackP["omega"] * s) - vertex[1]
    dz = z0_dash + s * trackP["tanL"] - vertex[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


""" Control function to check if PCA calculation works,
    calculates the distance of any point on the track
    (linear approx ) to the PV.
    Compare to [1] and [2] for more details.

           s*cos(phi0) + x0
   v(s) =  s*sin(phi0) + y0
           s*tan(lambda) + z0'

  x0 = xr + cos(pi/2-phi0)*d0, y0 = yr - sin(pi/2 -phi0)*d0, z0' = z0+zr
"""


def distApprox(trackP, vertex, s):
    x0 = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    y0 = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    z0_dash = trackP["z0"] + trackP["zr"]
    dx = s * math.cos(trackP["phi0"]) + x0 - vertex[0]
    dy = s * math.sin(trackP["phi0"]) + y0 - vertex[1]
    dz = s * trackP["tanL"] + z0_dash - vertex[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


"""
Helper to calculate the position of the PCA to the PV according to [3]
"""


def calcPCA(trackP, vertex, f2D=False):
    s = calcSPCA(trackP, vertex, f2D)
    x0 = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    y0 = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    z0_dash = trackP["z0"] + trackP["zr"]
    x = s * math.cos(trackP["phi0"]) + x0
    y = s * math.sin(trackP["phi0"]) + y0
    z = s * trackP["tanL"] + z0_dash
    return [x, y, z]


"""
Helper to calculate the uncertainty on the PCA calculated
according to [3]
"""


def calcPCA_error(trackP, vertex, f2D=False):
    s = calcSPCA(trackP, vertex, f2D)
    s_error = calcSPCA_Error(trackP, vertex, f2D)
    # x0 = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    # y0 = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    # z0_dash = trackP["z0"] + trackP["zr"]
    x0_error = math.sqrt(
        math.cos(math.pi / 2.0 - trackP["phi0"])
        * math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0_error"]
        * trackP["d0_error"]
        + math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * trackP["phi0_error"]
        * trackP["phi0_error"]
    )  # noqa
    y0_error = math.sqrt(
        math.sin(math.pi / 2.0 - trackP["phi0"])
        * math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0_error"]
        * trackP["d0_error"]
        + math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * trackP["phi0_error"]
        * trackP["phi0_error"]
    )  # noqa
    z0_dash_error = math.sqrt(trackP["z0_error"] * trackP["z0_error"])

    # x = s * math.cos(trackP["phi0"]) + x0
    # y = s * math.sin(trackP["phi0"]) + y0
    # z = s * trackP["tanL"] + z0_dash

    x_error = math.sqrt(
        math.cos(trackP["phi0"]) * math.cos(trackP["phi0"]) * s_error * s_error
        + x0_error * x0_error
        + s * math.sin(trackP["phi0"]) * s * math.sin(trackP["phi0"]) * trackP["phi0_error"] * trackP["phi0_error"]
    )  # noqa
    y_error = math.sqrt(
        math.sin(trackP["phi0"]) * math.sin(trackP["phi0"]) * s_error * s_error
        + y0_error * y0_error
        + s * math.cos(trackP["phi0"]) * s * math.cos(trackP["phi0"]) * trackP["phi0_error"] * trackP["phi0_error"]
    )  # noqa
    z_error = math.sqrt(
        trackP["tanL"] * trackP["tanL"] * s_error * s_error
        + s * s * trackP["tanL_error"] * trackP["tanL_error"]
        + z0_dash_error * z0_dash_error
    )
    return [x_error, y_error, z_error]


"""
Helper to calculate the xy travel distance s at the PCA
to the PV. For this bring linear approximation of helix
equation to the form needed in [3].
            ax + (bx-ax)*t
i.e. v(t) = ay + (by-ay)*t
            az + (bz-az)*t

t==s
for approx eq see: distApprox
"""


def calcSPCA(trackP, vertex, f2D=False):
    x0 = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    y0 = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    z0_dash = trackP["z0"] + trackP["zr"]
    ax, ay, az = x0, y0, z0_dash
    bx = math.cos(trackP["phi0"]) + x0
    by = math.sin(trackP["phi0"]) + y0
    bz = trackP["tanL"] + z0_dash
    num = -((ax - vertex[0]) * (bx - ax) + (ay - vertex[1]) * (by - ay) + (az - vertex[2]) * (bz - az))
    den = abs((bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az))
    if f2D:
        num = -((ax - vertex[0]) * (bx - ax) + (ay - vertex[1]) * (by - ay))
        den = abs((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
    return num / den


"""
Helper to calculate the error for the xy travel distance s at the PCA
to the PV. For this bring linear approximation of helix
equation to the form needed in [3].
            ax + (bx-ax)*t
i.e. v(t) = ay + (by-ay)*t
            az + (bz-az)*t

t==s
for approx eq see: distApprox
"""


def calcSPCA_Error(trackP, vertex, f2D=False):
    x0 = trackP["xr"] + math.cos(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    x0_error = math.sqrt(
        math.cos(math.pi / 2.0 - trackP["phi0"])
        * math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0_error"]
        * trackP["d0_error"]
        + math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * trackP["phi0_error"]
        * trackP["phi0_error"]
    )  # noqa
    y0 = trackP["yr"] - math.sin(math.pi / 2.0 - trackP["phi0"]) * trackP["d0"]
    y0_error = math.sqrt(
        math.sin(math.pi / 2.0 - trackP["phi0"])
        * math.sin(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0_error"]
        * trackP["d0_error"]
        + math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * math.cos(math.pi / 2.0 - trackP["phi0"])
        * trackP["d0"]
        * trackP["phi0_error"]
        * trackP["phi0_error"]
    )  # noqa
    z0_dash = trackP["z0"] + trackP["zr"]
    z0_dash_error = math.sqrt(trackP["z0_error"] * trackP["z0_error"])
    ax, ay, az = x0, y0, z0_dash
    ax_error, ay_error, az_error = x0_error, y0_error, z0_dash_error
    bx = math.cos(trackP["phi0"]) + x0
    by = math.sin(trackP["phi0"]) + y0
    bz = trackP["tanL"] + z0_dash
    bx_error = math.sqrt(
        math.sin(trackP["phi0"]) * math.sin(trackP["phi0"]) * trackP["phi0_error"] * trackP["phi0_error"]
        + x0_error * x0_error
    )
    by_error = math.sqrt(
        math.cos(trackP["phi0"]) * math.cos(trackP["phi0"]) * trackP["phi0_error"] * trackP["phi0_error"]
        + y0_error * y0_error
    )
    bz_error = math.sqrt(trackP["tanL_error"] * trackP["tanL_error"] + z0_dash_error * z0_dash_error)
    num = -((ax - vertex[0]) * (bx - ax) + (ay - vertex[1]) * (by - ay) + (az - vertex[2]) * (bz - az))
    num_error = math.sqrt(
        ax_error * ax_error * (2 * ax - bx - vertex[0]) * (2 * ax - bx - vertex[0])
        + bx_error * bx_error * (vertex[0] - bx) * (vertex[0] - bx)
        + ay_error * ay_error * (2 * ay - by - vertex[0]) * (2 * ay - by - vertex[0])
        + by_error * by_error * (vertex[0] - by) * (vertex[0] - by)
        + az_error * az_error * (2 * az - bz - vertex[0]) * (2 * az - bz - vertex[0])
        + bz_error * bz_error * (vertex[0] - bz) * (vertex[0] - bz)
    )
    den = (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az)
    den_error = math.sqrt(
        bx_error * bx_error * (2 * bx - 2 * ax) * (2 * bx - 2 * ax)
        + ax_error * ax_error * (2 * ax - 2 * bx) * (2 * ax - 2 * bx)
        + by_error * by_error * (2 * by - 2 * ay) * (2 * by - 2 * ay)
        + ay_error * ay_error * (2 * ay - 2 * by) * (2 * ay - 2 * by)
        + bz_error * bz_error * (2 * bz - 2 * az) * (2 * bz - 2 * az)
        + az_error * az_error * (2 * az - 2 * bz) * (2 * az - 2 * bz)
    )
    if f2D:
        num = -((ax - vertex[0]) * (bx - ax) + (ay - vertex[1]) * (by - ay) + (az - vertex[2]) * (bz - az))
        num_error = math.sqrt(
            ax_error * ax_error * (2 * ax - bx - vertex[0]) * (2 * ax - bx - vertex[0])
            + bx_error * bx_error * (vertex[0] - bx) * (vertex[0] - bx)
            + ay_error * ay_error * (2 * ay - by - vertex[0]) * (2 * ay - by - vertex[0])
            + by_error * by_error * (vertex[0] - by) * (vertex[0] - by)
        )
        den = (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az)
        den_error = math.sqrt(
            bx_error * bx_error * (2 * bx - 2 * ax) * (2 * bx - 2 * ax)
            + ax_error * ax_error * (2 * ax - 2 * bx) * (2 * ax - 2 * bx)
            + by_error * by_error * (2 * by - 2 * ay) * (2 * by - 2 * ay)
            + ay_error * ay_error * (2 * ay - 2 * by) * (2 * ay - 2 * by)
        )
    return math.sqrt(
        (1 / den) * (1 / den) * num_error * num_error + (num / (den * den)) * (num / (den * den)) * den_error * den_error
    )


"""
Helper to quickly cross check omega
i.e. the track to reco particle association
"""


def ptToOmega(pT, B):
    r = pT / (0.3 * B)
    rInMM = r * 1000
    return 1.0 / rInMM


"""
Wraps the above helper functions to find the PCA and lifetime variables.
"""


def findTrackPCAs(
    frame,
    ev,
    recoParticleCollection="MergedRecoParticles",
    trackCollection="SiTracks_Refitted_1",
    vertexCollection="PrimaryVertices",
    debug=-1,
):
    vertex = [
        frame[vertexCollection][ev][vertexCollection + ".position.x"][0],
        frame[vertexCollection][ev][vertexCollection + ".position.y"][0],
        frame[vertexCollection][ev][vertexCollection + ".position.z"][0],
    ]
    particles_ = frame[recoParticleCollection]
    particles = ak.Record({k.replace(f"{recoParticleCollection}.", ""): particles_[k] for k in particles_.fields})
    reco_particle_mask = particles["type"] != 0
    trkIndexMap = frame["idx_track"][ev]
    partTickleTrackLink_b = particles["tracks_begin"][reco_particle_mask][ev]
    partTickleTrackLink_e = particles["tracks_end"][reco_particle_mask][ev]
    partTickleTrackLink = []
    for ili, part_trkidx in enumerate(partTickleTrackLink_b):
        if part_trkidx == partTickleTrackLink_e[ili]:
            partTickleTrackLink.append(-1)  # no track / neutral
        else:
            partTickleTrackLink.append(trkIndexMap[part_trkidx])
    impacts_pcas_pvs = -1000 * np.ones((len(partTickleTrackLink), 17))
    impacts_pcas_errors = -1000 * np.ones((len(partTickleTrackLink), 11))

    # debug for track particle assocsiation
    if debug > 2:
        P4 = vector.awk(
            ak.zip(
                {
                    "px": particles["momentum.x"][ev],
                    "py": particles["momentum.y"][ev],
                    "pz": particles["momentum.z"][ev],
                    "energy": particles["energy"][ev],
                }
            )
        )
        omegas_theo = [ptToOmega(p, 4) for p in P4.pt]
        charge = particles["charge"][ev]
        if ev < 10:
            print("########################################")
            print("OMEGA debug event", ev)
            for ili, part_trkidx in enumerate(partTickleTrackLink):
                print(part_trkidx)
                print("particle:", ili)
                if charge[ili] == 0:
                    continue
                print("charge:", charge[ili])
                print("pt:", P4.pt[ili])
                print("predicted omega:", omegas_theo[ili] * charge[ili])
                print("omega:", frame[trackCollection][ev][trackCollection + ".omega"][part_trkidx * 4])
            print("########################################")
    for ili, part_trkidx in enumerate(partTickleTrackLink):
        # each track exists 4 times, go to copy for trackstate at IP as interpolation works best here
        # i.e track 0 is present at idx 0-3 for different track states, 1 at 4-7, and so on -> multiply by 4
        si_trkidx = part_trkidx * 4
        if part_trkidx < 0:
            if debug >= 0:
                print("Found no SiTrack for particle! Maybe neutral?")
        elif si_trkidx < 0 or si_trkidx >= len(frame[trackCollection][ev][trackCollection + ".location"]):
            print("Warning, invalid track indx, please check!")
        else:
            if debug >= 0:
                print("Found SiTrack for particle!")
            trackP = {}
            pr = [
                frame[trackCollection][ev][trackCollection + ".referencePoint.x"][si_trkidx],
                frame[trackCollection][ev][trackCollection + ".referencePoint.y"][si_trkidx],
                frame[trackCollection][ev][trackCollection + ".referencePoint.z"][si_trkidx],
            ]
            d0 = frame[trackCollection][ev][trackCollection + ".D0"][si_trkidx]
            z0 = frame[trackCollection][ev][trackCollection + ".Z0"][si_trkidx]
            phi0 = frame[trackCollection][ev][trackCollection + ".phi"][si_trkidx]
            tanL = frame[trackCollection][ev][trackCollection + ".tanLambda"][si_trkidx]
            omega = frame[trackCollection][ev][trackCollection + ".omega"][si_trkidx]
            # declared with 21 entries but only has 15 as expected
            cov = frame[trackCollection][ev][trackCollection + ".covMatrix[21]"][si_trkidx]
            """ following:
                https://github.com/iLCSoft/ILDPerformance/blob/master/tracking/src/DDDiagnostics.cc#L777-L803
                We only use the sign of omega, error not needed
            """
            d0_error = cov[0]
            z0_error = cov[9]
            phi0_error = cov[2]
            tanL_error = cov[14]
            omega_error = cov[5]
            extras = [phi0, tanL, omega]
            trackP = {
                "omega": omega,
                "omega_error": omega_error,
                "tanL": tanL,
                "tanL_error": tanL_error,
                "phi0": phi0,
                "phi0_error": phi0_error,
                "d0": d0,
                "d0_error": d0_error,
                "z0": z0,
                "z0_error": z0_error,
                "xr": pr[0],
                "yr": pr[1],
                "zr": pr[2],
            }
            pca = calcPCA(trackP, vertex)
            pca_error = calcPCA_error(trackP, vertex)
            dz = abs(pca[2] - vertex[2])
            dz_error = math.sqrt(pca_error[2] * pca_error[2])
            dxy = math.sqrt(sum([(vertex[i] - pca[i]) * (vertex[i] - pca[i]) for i in range(2)]))
            dxy_error = math.sqrt(
                sum([4 * ((vertex[i] - pca[i]) * (vertex[i] - pca[i]) * pca_error[i] * pca_error[i]) for i in range(2)])
            )
            d3 = math.sqrt(sum([(vertex[i] - pca[i]) * (vertex[i] - pca[i]) for i in range(3)]))
            d3_error = math.sqrt(
                sum([4 * ((vertex[i] - pca[i]) * (vertex[i] - pca[i]) * pca_error[i] * pca_error[i]) for i in range(3)])
            )
            pca_f2D = calcPCA(trackP, vertex, f2D=True)
            pca_f2D_error = calcPCA_error(trackP, vertex, f2D=True)
            dz_f2D = pca_f2D[2] - vertex[2]
            dz_f2D_error = math.sqrt(pca_f2D_error[2] * pca_f2D_error[2])
            dxy_f2D = math.sqrt(sum([(vertex[i] - pca_f2D[i]) * (vertex[i] - pca_f2D[i]) for i in range(2)]))
            dxy_f2D_error = math.sqrt(
                sum(
                    [
                        4 * ((vertex[i] - pca_f2D[i]) * (vertex[i] - pca_f2D[i]) * pca_f2D_error[i] * pca_f2D_error[i])
                        for i in range(2)
                    ]
                )
            )
            d3_f2D = math.sqrt(sum([(vertex[i] - pca_f2D[i]) * (vertex[i] - pca_f2D[i]) for i in range(3)]))
            d3_f2D_error = math.sqrt(
                sum(
                    [
                        4 * ((vertex[i] - pca_f2D[i]) * (vertex[i] - pca_f2D[i]) * pca_f2D_error[i] * pca_f2D_error[i])
                        for i in range(3)
                    ]
                )
            )
            # xy impact parameter, z impact parameter , 3d impact parameter, d0/z0 track parameters, PCA and PV:
            impacts_pcas_pvs[ili] = [dxy, dz, d3, d0, z0, dxy_f2D, dz_f2D, d3_f2D] + pca + vertex + extras
            # corresponding uncertainties:
            impacts_pcas_errors[ili] = [
                dxy_error,
                dz_error,
                d3_error,
                d0_error,
                z0_error,
                dxy_f2D_error,
                dz_f2D_error,
                d3_f2D_error,
            ] + pca_error
    return [impacts_pcas_pvs, impacts_pcas_errors]


"""
Helper to project the impact parameters of the track on the direction jet axis.
(currently diabled as sign seems to be random?)
(Following BTV-11-002)
"""


def calculateImpactParameterSigns(ips, pca, pv, jetp4):
    jet_direction = [jetp4["x"], jetp4["y"], jetp4["z"]]
    jet_norm = math.sqrt(sum([jet_direction[i] * jet_direction[i] for i in range(3)]))
    jet_direction = [(1.0 / jet_norm) * jet_direction[i] for i in range(len(jet_direction))]
    newips = []
    for iip, ip in enumerate(ips):
        if ip == -1000.0:
            newips.append(-1000.0)
            continue
        pca_direction = np.array([pca[i][iip] - pv[i][iip] for i in range(3)])
        pca_norm = math.sqrt(sum([pca_direction[i] * pca_direction[i] for i in range(3)]))
        if pca_norm > 0:
            pca_direction = 1 / pca_norm * pca_direction
        else:
            pca_direction = jet_direction
        sign = np.sign(sum([pca_direction[i] * jet_direction[i] for i in range(3)]))
        newips.append(sign * abs(ip))
    return newips
