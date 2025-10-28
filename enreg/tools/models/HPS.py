import os
import math
import json
import copy
import vector
import numpy as np
import awkward as ak
from functools import cmp_to_key
from omegaconf import OmegaConf
from omegaconf import DictConfig
from enreg.tools import general as g
from enreg.tools.data_management import features as f


class hpsParticleBase:
    def __init__(self, p4, barcode=-1):
        self.p4 = p4
        self.updatePtEtaPhiMass()
        self.barcode = barcode

    def updatePtEtaPhiMass(self):
        self.energy = self.p4.energy
        self.p = self.p4.p
        self.pt = self.p4.pt
        self.theta = self.p4.theta
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

        self.u_x = math.cos(self.phi) * math.sin(self.theta)
        self.u_y = math.sin(self.phi) * math.sin(self.theta)
        self.u_z = math.cos(self.theta)


class Cand(hpsParticleBase):
    def __init__(self, p4, pdgId, q, d0, d0err, dz, dzerr, barcode=-1):
        super().__init__(p4=p4, barcode=barcode)
        self.pdgId = pdgId
        self.abs_pdgId = abs(pdgId)
        self.q = q
        self.d0 = d0
        self.d0err = d0err
        self.dz = math.fabs(dz)
        self.dzerr = dzerr

    def print(self):
        output = (
            "cand #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, pdgId = %i, charge = %1.1f"
            % (
                self.barcode,
                self.energy,
                self.pt,
                self.eta,
                self.phi,
                self.mass,
                self.pdgId,
                self.q,
            )
        )
        if abs(self.q) > 0.5:
            output += ", d0 = %1.3f +/- %1.3f, dz = %1.3f +/- %1.3f" % (
                self.d0 * 1.0e4,
                self.d0err * 1.0e4,
                self.dz * 1.0e4,
                self.dzerr * 1.0e4,
            )
        print(output)


def buildCands(
    cand_p4s, cand_pdgIds, cand_qs, cand_d0s, cand_d0errs, cand_dzs, cand_dzerrs
):
    if not (
        len(cand_p4s) == len(cand_pdgIds)
        and len(cand_pdgIds) == len(cand_qs)
        and len(cand_qs) == len(cand_d0s)
        and len(cand_d0s) == len(cand_d0errs)
        and len(cand_d0errs) == len(cand_dzs)
        and len(cand_dzs) == len(cand_dzerrs)
    ):
        raise ValueError(
            "Length of arrays for candidate for p4 and other features don't match !!"
        )
    cands = []
    num_cands = len(cand_p4s)
    for idx in range(num_cands):
        cand = Cand(
            cand_p4s[idx],
            cand_pdgIds[idx],
            cand_qs[idx],
            cand_d0s[idx],
            cand_d0errs[idx],
            cand_dzs[idx],
            cand_dzerrs[idx],
            barcode=idx,
        )
        cands.append(cand)
    return cands


def readCands(data):
    event_cand_p4s = g.reinitialize_p4(data["event_reco_cand_p4s"])
    event_cand_pdgIds = data["event_reco_cand_pdg"]
    event_cand_qs = data["event_reco_cand_charge"]
    event_cand_d0s = data["event_reco_cand_dxy"]
    event_cand_d0errs = data["event_reco_cand_dxy_err"]
    event_cand_dzs = data["event_reco_cand_dz"]
    event_cand_dzerrs = data["event_reco_cand_dz_err"]
    if not (
        len(event_cand_p4s) == len(event_cand_pdgIds)
        and len(event_cand_pdgIds) == len(event_cand_qs)
        and len(event_cand_qs) == len(event_cand_d0s)
        and len(event_cand_d0s) == len(event_cand_d0errs)
        and len(event_cand_d0errs) == len(event_cand_dzs)
        and len(event_cand_dzs) == len(event_cand_dzerrs)
    ):
        raise ValueError(
            "Length of arrays for candidate p4 and other features don't match !!"
        )
    event_cands = []
    num_jets = len(event_cand_p4s)
    for idx in range(num_jets):
        jet_cands = buildCands(
            event_cand_p4s[idx],
            event_cand_pdgIds[idx],
            event_cand_qs[idx],
            event_cand_d0s[idx],
            event_cand_d0errs[idx],
            event_cand_dzs[idx],
            event_cand_dzerrs[idx],
        )
        event_cands.append(jet_cands)
    return event_cands


def comp_p_sum(cands):
    p_sum = 0.0
    for cand in cands:
        p_sum += cand.p
    return p_sum


def comp_pt_sum(cands):
    pt_sum = 0.0
    for cand in cands:
        pt_sum += cand.pt
    return pt_sum


def selectCandsByDeltaR(cands, ref, dRmax):
    selectedCands = []
    for cand in cands:
        dR = f.deltaR_thetaPhi(cand.theta, cand.phi, ref.theta, ref.phi)
        if dR < dRmax:
            selectedCands.append(cand)
    return selectedCands


def selectCandsByPdgId(cands, pdgIds=[]):
    selectedCands = []
    for cand in cands:
        if cand.abs_pdgId in pdgIds:
            selectedCands.append(cand)
    return selectedCands


class Jet(hpsParticleBase):
    def __init__(
        self,
        jet_p4,
        jet_constituents_p4,
        jet_constituents_pdgId,
        jet_constituents_q,
        jet_constituent_d0,
        jet_constituent_d0err,
        jet_constituent_dz,
        jet_constituent_dzerr,
        barcode=-1,
    ):
        super().__init__(p4=jet_p4, barcode=barcode)
        self.constituents = buildCands(
            jet_constituents_p4,
            jet_constituents_pdgId,
            jet_constituents_q,
            jet_constituent_d0,
            jet_constituent_d0err,
            jet_constituent_dz,
            jet_constituent_dzerr,
        )
        # CV: reverse=True argument needed in order to sort jet constituents in order of decreasing (and NOT increasing) pT
        self.constituents.sort(key=lambda cand: cand.pt, reverse=True)
        self.num_constituents = len(self.constituents)
        self.q = 0.0
        for cand in self.constituents:
            self.q += cand.q

    def print(self):
        print(
            "jet #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.2f, #constituents = %i"
            % (
                self.barcode,
                self.energy,
                self.pt,
                self.eta,
                self.phi,
                self.mass,
                len(self.constituents),
            )
        )
        print("constituents:")
        for cand in self.constituents:
            cand.print()


def buildJets(
    jet_p4s,
    jet_constituent_p4s,
    jet_constituent_pdgIds,
    jet_constituent_qs,
    jet_constituent_d0s,
    jet_constituent_d0errs,
    jet_constituent_dzs,
    jet_constituent_dzerrs,
):
    if not (
        len(jet_p4s) == len(jet_constituent_p4s)
        and len(jet_constituent_p4s) == len(jet_constituent_pdgIds)
        and len(jet_constituent_pdgIds) == len(jet_constituent_qs)
        and len(jet_constituent_qs) == len(jet_constituent_d0s)
        and len(jet_constituent_d0s) == len(jet_constituent_d0errs)
        and len(jet_constituent_d0errs) == len(jet_constituent_dzs)
        and len(jet_constituent_dzs) == len(jet_constituent_dzerrs)
    ):
        raise ValueError(
            "Length of arrays for jet p4, constituent p4, and other constituent features don't match !!"
        )
    jets = []
    num_jets = len(jet_p4s)
    for idx in range(num_jets):
        jet = Jet(
            jet_p4s[idx],
            jet_constituent_p4s[idx],
            jet_constituent_pdgIds[idx],
            jet_constituent_qs[idx],
            jet_constituent_d0s[idx],
            jet_constituent_d0errs[idx],
            jet_constituent_dzs[idx],
            jet_constituent_dzerrs[idx],
            idx,
        )
        jets.append(jet)
    return jets


def readJets(data):
    jet_p4s = g.reinitialize_p4(data["reco_jet_p4s"])
    jet_constituent_p4s = g.reinitialize_p4(data["reco_cand_p4s"])
    jet_constituent_pdgIds = data["reco_cand_pdg"]
    jet_constituent_qs = data["reco_cand_charge"]
    jet_constituent_d0s = data["reco_cand_dxy"]
    jet_constituent_d0errs = data["reco_cand_dxy_err"]
    jet_constituent_dzs = data["reco_cand_dz"]
    jet_constituent_dzerrs = data["reco_cand_dz_err"]
    jets = buildJets(
        jet_p4s,
        jet_constituent_p4s,
        jet_constituent_pdgIds,
        jet_constituent_qs,
        jet_constituent_d0s,
        jet_constituent_d0errs,
        jet_constituent_dzs,
        jet_constituent_dzerrs,
    )
    return jets


m_pi0 = 0.135


class Strip(hpsParticleBase):
    def __init__(self, cands=[], barcode=-1):
        cand_p4s = [cand.p4 for cand in cands]
        cand_p4s = np.array([[p4.px, p4.py, p4.pz, p4.E] for p4 in cand_p4s])
        sum_p4 = None
        if len(cand_p4s) == 0:
            sum_p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(cand_p4s, axis=0)
            sum_p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])
        strip_px = sum_p4.px
        strip_py = sum_p4.py
        strip_pz = sum_p4.pz
        strip_E = math.sqrt(
            strip_px * strip_px
            + strip_py * strip_py
            + strip_pz * strip_pz
            + m_pi0 * m_pi0
        )
        strip_p4 = vector.obj(px=strip_px, py=strip_py, pz=strip_pz, E=strip_E)
        super().__init__(p4=strip_p4, barcode=barcode)
        self.cands = set(cands)

    def print(self):
        print(
            "strip #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f"
            % (self.barcode, self.energy, self.pt, self.eta, self.phi, self.mass)
        )
        for cand in self.cands:
            cand.print()


def getParameter(cfg, name, default_value):
    value = None
    if name in cfg.keys():
        value = cfg[name]
    else:
        value = default_value
    return value


class StripAlgo:
    def __init__(self, cfg, verbosity=0):
        self.cfg = cfg
        if verbosity >= 1:
            print("<StripAlgo::StripAlgo>:")
        if verbosity >= 1:
            print(" useGammas = %s" % self.cfg.useGammas)
            print(" minGammaPtSeed = %1.2f" % self.cfg.minGammaPtSeed)
            print(" minGammaPtAdd = %1.2f" % self.cfg.minGammaPtAdd)
            print(" useElectrons = %s" % self.cfg.useElectrons)
            print(" minElectronPtSeed = %1.2f" % self.cfg.minElectronPtSeed)
            print(" minElectronPtAdd = %1.2f" % self.cfg.minElectronPtAdd)
            print(" minStripPt = %1.2f" % self.cfg.minStripPt)
            print(" updateStripAfterEachCand = %s" % self.cfg.updateStripAfterEachCand)
            print(" maxStripBuildIterations = %i" % self.cfg.maxStripBuildIterations)
            print(" maxStripSizeEta = %1.2f" % self.cfg.maxStripSizeEta)
            print(" maxStripSizePhi = %1.2f" % self.cfg.maxStripSizePhi)

        self.verbosity = verbosity

    def updateStripP4(self, strip):
        strip.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for cand in strip.cands:
            strip.p4 = strip.p4 + cand.p4
        strip.updatePtEtaPhiMass()

    def addCandsToStrip(
        self, strip, cands, candBarcodesPreviousStrips, candBarcodesCurrentStrip
    ):
        isCandAdded = False
        for cand in cands:
            if not (
                cand.barcode in candBarcodesPreviousStrips
                or cand.barcode in candBarcodesCurrentStrip
            ):
                dEta = f.deltaTheta(cand.theta, strip.theta)
                dPhi = f.deltaPhi(cand.phi, strip.phi)
                if dEta < self.cfg.maxStripSizeEta and dPhi < self.cfg.maxStripSizePhi:
                    strip.cands.add(cand)
                    if self.cfg.updateStripAfterEachCand:
                        self.updateStripP4(strip)
                    candBarcodesCurrentStrip.add(cand.barcode)
                    isCandAdded = True
        return isCandAdded

    def markCandsInStrip(self, candBarcodesPreviousStrips, candBarcodesCurrentStrip):
        candBarcodesPreviousStrips.update(candBarcodesCurrentStrip)

    def buildStrips(self, cands):
        if self.verbosity >= 3:
            print("<StripAlgo::buildStrips>:")
        seedCands = []
        addCands = []
        for cand in cands:
            if (cand.abs_pdgId == 22 and self.cfg.useGammas) or (
                cand.abs_pdgId == 11 and self.cfg.useElectrons
            ):
                minPtSeed = None
                minPtAdd = None
                if cand.abs_pdgId == 22:
                    minPtSeed = self.cfg.minGammaPtSeed
                    minPtAdd = self.cfg.minGammaPtAdd
                elif cand.abs_pdgId == 11:
                    minPtSeed = self.cfg.minElectronPtSeed
                    minPtAdd = self.cfg.minElectronPtAdd
                else:
                    assert 0
                if cand.pt > minPtSeed:
                    seedCands.append(cand)
                elif cand.pt > minPtAdd:
                    addCands.append(cand)
        if self.verbosity >= 3:
            print("seedCands:")
            for cand in seedCands:
                cand.print()
            print("#seedCands = %i" % len(seedCands))
            print("addCands:")
            for cand in addCands:
                cand.print()
            print("#addCands = %i" % len(addCands))

        output_strips = []

        seedCandBarcodesPreviousStrips = set()
        addCandBarcodesPreviousStrips = set()

        idxStrip = 0
        for seedCand in seedCands:
            if self.verbosity >= 4:
                print("Processing seedCand #%i" % seedCand.barcode)
            if seedCand.barcode not in seedCandBarcodesPreviousStrips:
                currentStrip = Strip([seedCand], idxStrip)

                seedCandBarcodesCurrentStrip = set([seedCand.barcode])
                addCandBarcodesCurrentStrip = set()

                stripBuildIterations = 0
                while (
                    stripBuildIterations < self.cfg.maxStripBuildIterations
                    or self.cfg.maxStripBuildIterations == -1
                ):
                    isCandAdded = False
                    isCandAdded |= self.addCandsToStrip(
                        currentStrip,
                        seedCands,
                        seedCandBarcodesPreviousStrips,
                        seedCandBarcodesCurrentStrip,
                    )
                    isCandAdded |= self.addCandsToStrip(
                        currentStrip,
                        addCands,
                        addCandBarcodesPreviousStrips,
                        addCandBarcodesCurrentStrip,
                    )
                    if not self.cfg.updateStripAfterEachCand:
                        self.updateStripP4(currentStrip)
                    if not isCandAdded:
                        break
                    ++stripBuildIterations

                if self.verbosity >= 4:
                    print("currentStrip:")
                    currentStrip.print()

                if currentStrip.pt > self.cfg.minStripPt:
                    currentStrip.barcode = idxStrip
                    ++idxStrip
                    output_strips.append(currentStrip)
                    self.markCandsInStrip(
                        seedCandBarcodesPreviousStrips, seedCandBarcodesCurrentStrip
                    )
                    self.markCandsInStrip(
                        addCandBarcodesPreviousStrips, addCandBarcodesCurrentStrip
                    )

        if self.verbosity >= 4:
            print("output_strips:")
            for strip in output_strips:
                strip.print()

        return output_strips


class Tau(hpsParticleBase):
    def __init__(self, chargedCands=[], strips=[], barcode=-1):
        chargedCand_and_strip_p4s = [chargedCand.p4 for chargedCand in chargedCands] + [
            strip.p4 for strip in strips
        ]
        chargedCand_and_strip_p4s = np.array(
            [[p4.px, p4.py, p4.pz, p4.E] for p4 in chargedCand_and_strip_p4s]
        )
        sum_p4 = None
        if len(chargedCand_and_strip_p4s) == 0:
            sum_p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(chargedCand_and_strip_p4s, axis=0)
            sum_p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])
        super().__init__(p4=sum_p4, barcode=barcode)
        self.q = 0.0
        self.leadChargedCand_pt = 0.0
        for chargedCand in chargedCands:
            if chargedCand.q > 0.0:
                self.q += 1.0
            elif chargedCand.q < 0.0:
                self.q -= 1.0
            else:
                assert 0
            if chargedCand.pt > self.leadChargedCand_pt:
                self.leadChargedCand_pt = chargedCand.pt
        self.signal_chargedCands = chargedCands
        self.signal_strips = strips
        self.updateSignalCands()
        self.idDiscr = -1.0
        self.decayMode = "undefined"
        self.iso_cands = set()
        self.chargedIso_dR0p5 = -1.0
        self.gammaIso_dR0p5 = -1.0
        self.neutralHadronIso_dR0p5 = -1.0
        self.combinedIso_dR0p5 = -1.0

    def updateSignalCands(self):
        self.num_signal_chargedCands = len(self.signal_chargedCands)
        self.num_signal_strips = len(self.signal_strips)
        self.signal_cands = set()
        self.signal_cands.update(self.signal_chargedCands)
        for strip in self.signal_strips:
            for cand in strip.cands:
                self.signal_cands.add(cand)
        self.signal_gammaCands = selectCandsByPdgId(self.signal_cands, [22])

    def print(self):
        print(
            "tau #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, idDiscr = %1.3f, decayMode = %s"
            % (
                self.barcode,
                self.energy,
                self.pt,
                self.eta,
                self.phi,
                self.mass,
                self.idDiscr,
                self.decayMode,
            )
        )
        # print("signal_chargedCands:")
        # for cand in self.signal_chargedCands:
        #    cand.print()
        # print("#signal_gammaCands = %i" % len(self.signal_gammaCands))
        # print("#signal_chargedCands = %i" % len(self.signal_chargedCands))
        # print("leadChargedCand: pT = %1.1f" % self.leadChargedCand_pt)
        # print("signal_strips:")
        # for strip in self.signal_strips:
        #    strip.print()
        print("signal_cands:")
        for cand in self.signal_cands:
            cand.print()
        print("iso_cands:")
        for cand in self.iso_cands:
            cand.print()
        print(
            " isolation: charged = %1.2f, gamma = %1.2f, neutralHadron = %1.2f, combined = %1.2f"
            % (
                self.chargedIso_dR0p5,
                self.gammaIso_dR0p5,
                self.neutralHadronIso_dR0p5,
                self.combinedIso_dR0p5,
            )
        )


def write_tau_p4s(taus):
    retVal = vector.awk(
        ak.zip(
            {
                "px": [tau.p4.px for tau in taus],
                "py": [tau.p4.py for tau in taus],
                "pz": [tau.p4.pz for tau in taus],
                "mass": [tau.p4.mass for tau in taus],
            }
        )
    )
    return retVal


def build_dummy_array(dtype=float):
    num = 0
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            ak.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def write_tau_cand_p4s(taus, collection):
    # TODO: There is a smarter way to do this
    retVal = ak.Array(
        [
            (
                vector.awk(
                    ak.zip(
                        {
                            "px": [cand.p4.px for cand in getattr(tau, collection)],
                            "py": [cand.p4.py for cand in getattr(tau, collection)],
                            "pz": [cand.p4.pz for cand in getattr(tau, collection)],
                            "mass": [cand.p4.mass for cand in getattr(tau, collection)],
                        }
                    )
                )
                if len(getattr(tau, collection)) >= 1
                else build_dummy_array()
            )
            for tau in taus
        ]
    )
    return retVal


def write_tau_cand_attrs(taus, collection, attr, dtype):
    # TODO: There is a smarter way to do this
    retVal = ak.Array(
        [
            (
                ak.Array([getattr(cand, attr) for cand in getattr(tau, collection)])
                if len(getattr(tau, collection)) >= 1
                else build_dummy_array(dtype)
            )
            for tau in taus
        ]
    )
    return retVal


def get_decayMode(tau):
    # TODO: How to handle undefined
    retVal = None
    if tau.decayMode == "undefined":
        retVal = -1
    elif tau.decayMode == "1Prong0Pi0":
        retVal = 0
    elif tau.decayMode == "1Prong1Pi0":
        retVal = 1
    elif tau.decayMode == "1Prong2Pi0":
        retVal = 2
    elif tau.decayMode == "3Prong0Pi0":
        retVal = 10
    elif tau.decayMode == "3Prong1Pi0":
        retVal = 11
    else:
        raise RuntimeError("Invalid decayMode = '%s'" % tau.decayMode)
    return retVal


def comp_photonPtSumOutsideSignalCone(tau):
    retVal = 0.0
    for cand in tau.signal_gammaCands:
        dR = f.deltaR_thetaPhi(tau.theta, tau.phi, cand.theta, cand.phi)
        if dR > tau.signalConeSize:
            retVal += cand.pt
    return retVal


def comp_pt_weighted_dX(tau, cands, metric):
    pt_weighted_dX_sum = 0.0
    pt_sum = 0.0
    if metric is not None:
        for cand in cands:
            dX = abs(metric(tau, cand))
            pt_weighted_dX_sum += cand.pt * dX
            pt_sum += cand.pt
    if pt_sum > 0.0:
        return pt_weighted_dX_sum / pt_sum
    else:
        return 0.0


def writeTaus(taus):
    retVal = {
        "tau_p4s": write_tau_p4s(taus),
        "tauSigCand_p4s": write_tau_cand_p4s(taus, "signal_cands"),
        "tauSigCand_pdgIds": write_tau_cand_attrs(taus, "signal_cands", "pdgId", int),
        "tauSigCand_q": write_tau_cand_attrs(taus, "signal_cands", "q", float),
        "tauSigCand_d0": write_tau_cand_attrs(taus, "signal_cands", "d0", float),
        "tauSigCand_d0err": write_tau_cand_attrs(taus, "signal_cands", "d0err", float),
        "tauSigCand_dz": write_tau_cand_attrs(taus, "signal_cands", "dz", float),
        "tauSigCand_dzerr": write_tau_cand_attrs(taus, "signal_cands", "dzerr", float),
        "tauIsoCand_p4s": write_tau_cand_p4s(taus, "iso_cands"),
        "tauIsoCand_pdgIds": write_tau_cand_attrs(taus, "iso_cands", "pdgId", int),
        "tauIsoCand_q": write_tau_cand_attrs(taus, "iso_cands", "q", float),
        "tauIsoCand_d0": write_tau_cand_attrs(taus, "iso_cands", "d0", float),
        "tauIsoCand_d0err": write_tau_cand_attrs(taus, "iso_cands", "d0err", float),
        "tauIsoCand_dz": write_tau_cand_attrs(taus, "iso_cands", "dz", float),
        "tauIsoCand_dzerr": write_tau_cand_attrs(taus, "iso_cands", "dzerr", float),
        # "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),  # As we do not want to overwrite the PT value for this.
        "tauChargedIso_dR0p5": ak.Array([tau.chargedIso_dR0p5 for tau in taus]),
        "tauGammaIso_dR0p5": ak.Array([tau.gammaIso_dR0p5 for tau in taus]),
        "tauNeutralHadronIso_dR0p5": ak.Array(
            [tau.neutralHadronIso_dR0p5 for tau in taus]
        ),
        "tauChargedIso_dR0p3": ak.Array(
            [
                comp_pt_sum(selectCandsByDeltaR(tau.iso_chargedCands, tau, 0.3))
                for tau in taus
            ]
        ),
        "tauGammaIso_dR0p3": ak.Array(
            [
                comp_pt_sum(selectCandsByDeltaR(tau.iso_gammaCands, tau, 0.3))
                for tau in taus
            ]
        ),
        "tauNeutralHadronIso_dR0p3": ak.Array(
            [
                comp_pt_sum(selectCandsByDeltaR(tau.iso_neutralHadronCands, tau, 0.3))
                for tau in taus
            ]
        ),
        "tauPhotonPtSumOutsideSignalCone": ak.Array(
            [comp_photonPtSumOutsideSignalCone(tau) for tau in taus]
        ),
        "tau_charge": ak.Array([tau.q for tau in taus]),
        "tau_decaymode": ak.Array([get_decayMode(tau) for tau in taus]),
        "tau_nGammas": ak.Array([len(tau.signal_gammaCands) for tau in taus]),
        "tau_nCharged": ak.Array([len(tau.signal_chargedCands) for tau in taus]),
        "tau_leadChargedCand_pt": ak.Array([tau.leadChargedCand_pt for tau in taus]),
        "tau_emEnergyFrac": ak.Array(
            [
                (comp_pt_sum(tau.signal_gammaCands) / tau.pt) if tau.pt > 0.0 else 0.0
                for tau in taus
            ]
        ),
        "tau_dEta_strip": ak.Array(
            [
                comp_pt_weighted_dX(tau, tau.signal_gammaCands, comp_deltaTheta)
                for tau in taus
            ]
        ),
        "tau_dPhi_strip": ak.Array(
            [
                comp_pt_weighted_dX(tau, tau.signal_gammaCands, comp_deltaPhi)
                for tau in taus
            ]
        ),
        "tau_dR_signal": ak.Array(
            [
                comp_pt_weighted_dX(tau, tau.signal_gammaCands, comp_deltaR_thetaPhi)
                for tau in taus
            ]
        ),
        "tau_dR_iso": ak.Array(
            [
                comp_pt_weighted_dX(tau, tau.iso_gammaCands, comp_deltaR_thetaPhi)
                for tau in taus
            ]
        ),
    }
    return retVal


def comp_deltaR_thetaPhi(obj_1, obj_2):
    return f.deltaR_thetaPhi(obj_1.theta, obj_1.phi, obj_2.theta, obj_2.phi)


def comp_deltaPhi(obj_1, obj_2):
    return f.deltaPhi(obj_1.phi, obj_2.phi)


def comp_deltaTheta(obj_1, obj_2):
    return f.deltaTheta(obj_1.theta, obj_2.theta)


class CombinatoricsGenerator:
    def __init__(self, verbosity):
        self.verbosity = verbosity

    def factorial(self, k):
        assert k >= 0
        if k <= 1:
            return 1
        else:
            return k * self.factorial(k - 1)

    def generate(self, k, n):
        if self.verbosity >= 3:
            print("<CombinatoricsGenerator::generate>:")
            print(" k=%i & n=%i" % (k, n))

        if k <= 0 or k > n:
            if self.verbosity >= 3:
                print("combinations = []")
                print("#combinations = 0 (expected = 0)")
            return []

        retVal = []

        digits = [idx for idx in range(k)]

        current_digit = k - 1
        while True:
            assert (
                len("".join("%i" % digits[idx] for idx in range(k)))
                <= len("%i" % n) * k
            )
            retVal.append(copy.deepcopy(digits))
            if digits[current_digit] < (n - (k - current_digit)):
                digits[current_digit] = digits[current_digit] + 1
            else:
                while current_digit >= 0 and digits[current_digit] >= (
                    n - (k - current_digit)
                ):
                    current_digit -= 1
                if current_digit >= 0:
                    digits[current_digit] = digits[current_digit] + 1
                    for idx in range(current_digit + 1, k):
                        digits[idx] = digits[current_digit] + (idx - current_digit)
                    current_digit = k - 1
                else:
                    break

        if self.verbosity >= 3:
            print("combinations = %s" % retVal)
            print(
                "#combinations = %i (expected = %i)"
                % (
                    len(retVal),
                    self.factorial(n) / (self.factorial(k) * self.factorial(n - k)),
                )
            )

        return retVal


def rank_tau_candidates(tau1, tau2):
    if tau1.num_signal_chargedCands > tau2.num_signal_chargedCands:
        return +1
    if tau1.num_signal_chargedCands < tau2.num_signal_chargedCands:
        return -1
    if tau1.pt > tau2.pt:
        return +1
    if tau1.pt < tau2.pt:
        return -1
    if tau1.num_signal_strips > tau2.num_signal_strips:
        return +1
    if tau1.num_signal_strips < tau2.num_signal_strips:
        return -1
    if tau1.combinedIso_dR0p5 < tau2.combinedIso_dR0p5:
        return +1
    if tau1.combinedIso_dR0p5 > tau2.combinedIso_dR0p5:
        return -1
    return 0


class HPSAlgo:
    def __init__(self, cfg, verbosity=0):
        self.cfg = cfg
        self.verbosity = verbosity
        if verbosity >= 1:
            print("<HPSAlgo::HPSAlgo>:")
        #  TODO
        #  Here input is cfg.builder
        if verbosity >= 1:
            print("signalCands:")
            print(
                " minChargedHadronPt = %1.2f" % self.cfg.signalCands.minChargedHadronPt
            )
            print(" minElectronPt = %1.2f" % self.cfg.signalCands.minElectronPt)
            print(" minMuonPt = %1.2f" % self.cfg.signalCands.minMuonPt)

        if verbosity >= 1:
            print("isolationCands:")
            print(
                " minChargedHadronPt = %1.2f"
                % self.cfg.isolationCands.minChargedHadronPt
            )
            print(" minElectronPt = %1.2f" % self.cfg.isolationCands.minElectronPt)
            print(" minGammaPt = %1.2f" % self.cfg.isolationCands.minGammaPt)
            print(" minMuonPt = %1.2f" % self.cfg.isolationCands.minMuonPt)
            print(
                " minNeutralHadronPt = %1.2f"
                % self.cfg.isolationCands.minNeutralHadronPt
            )

        if verbosity >= 1:
            print("matchingConeSize = %1.2f" % self.cfg.matchingConeSize)
            print("isolationConeSize = %1.2f" % self.cfg.isolationConeSize)
            print("metric = '%s'" % self.cfg.metric)

        self.stripAlgo = StripAlgo(cfg.StripAlgo, verbosity)

        if verbosity >= 1:
            print("targetedDecayModes:")
            for decayMode in [
                "1Prong0Pi0",
                "1Prong1Pi0",
                "1Prong2Pi0",
                "3Prong0Pi0",
                "3Prong1Pi0",
            ]:
                if decayMode in self.cfg.decayModes.keys():
                    print(" %s:" % decayMode)
                    targetedDecayMode = self.cfg.decayModes[decayMode]
                    print(
                        "  numChargedCands = %i" % targetedDecayMode["numChargedCands"]
                    )
                    print("  numStrips = %i" % targetedDecayMode["numStrips"])
                    print(
                        "  tauMass: min = %1.2f, max = %1.2f"
                        % (
                            targetedDecayMode["minTauMass"],
                            targetedDecayMode["maxTauMass"],
                        )
                    )
                    if targetedDecayMode["numStrips"] > 0:
                        print(
                            "  stripMass: min = %1.2f, max = %1.2f"
                            % (
                                targetedDecayMode["minStripMass"],
                                targetedDecayMode["maxStripMass"],
                            )
                        )
                    print(
                        "  maxChargedCands = %i" % targetedDecayMode["maxChargedCands"]
                    )
                    print("  maxStrips = %i" % targetedDecayMode["maxStrips"])

        self.combinatorics = CombinatoricsGenerator(verbosity)
        self.verbosity = verbosity

    def selectSignalChargedCands(self, cands):
        signalCands = []
        for cand in cands:
            if (
                (cand.abs_pdgId == 11 and cand.pt > self.cfg.signalCands.minElectronPt)
                or (cand.abs_pdgId == 13 and cand.pt > self.cfg.signalCands.minMuonPt)
                or (
                    cand.abs_pdgId == 211
                    and cand.pt > self.cfg.signalCands.minChargedHadronPt
                )
            ):
                signalCands.append(cand)
        return signalCands

    def selectIsolationCands(self, cands):
        isolationCands = []
        for cand in cands:
            if (
                (
                    cand.abs_pdgId == 11
                    and cand.pt > self.cfg.isolationCands.minElectronPt
                )
                or (
                    cand.abs_pdgId == 13 and cand.pt > self.cfg.isolationCands.minMuonPt
                )
                or (
                    cand.abs_pdgId == 22
                    and cand.pt > self.cfg.isolationCands.minGammaPt
                )
                or (
                    cand.abs_pdgId == 211
                    and cand.pt > self.cfg.isolationCands.minChargedHadronPt
                )
                or (
                    cand.abs_pdgId == 130
                    and cand.pt > self.cfg.isolationCands.minNeutralHadronPt
                )
            ):
                isolationCands.append(cand)
        return isolationCands

    def cleanCands(self, cands_to_clean, cands, dRmatch=1.0e-3):
        cleanedCands = []
        for cand_to_clean in cands_to_clean:
            isOverlap = False
            for cand in cands:
                if cand.pdgId == cand_to_clean.pdgId and cand.q == cand_to_clean.q:
                    dR = f.deltaR_thetaPhi(
                        cand.theta, cand.phi, cand_to_clean.theta, cand_to_clean.phi
                    )  # TODO
                    if dR < dRmatch:
                        isOverlap = True
                        break
            if not isOverlap:
                cleanedCands.append(cand_to_clean)
        return cleanedCands

    def cleanStrips(self, strips, chargedCands):
        cleanedStrips = []
        for strip in strips:
            cleanedCands = self.cleanCands(strip.cands, chargedCands)
            cleanedStrip = Strip(cleanedCands, strip.barcode)
            if (
                len(cleanedCands) > 0
                and cleanedStrip.pt > self.cfg.StripAlgo.minStripPt
            ):
                cleanedStrips.append(cleanedStrip)
        return cleanedStrips

    def buildTau(self, jet, event_iso_cands):
        if self.verbosity >= 2:
            print("<hpsAlgo::buildTau>:")

        signal_chargedCands = self.selectSignalChargedCands(jet.constituents)
        if self.verbosity >= 2:
            print("#signal_chargedCands = %i" % len(signal_chargedCands))
            if self.verbosity >= 3:
                print("signal_chargedCands")
                for cand in signal_chargedCands:
                    cand.print()

        signal_strips = self.stripAlgo.buildStrips(jet.constituents)
        # CV: reverse=True argument needed in order to sort strips in order of decreasing (and NOT increasing) pT
        signal_strips.sort(key=lambda strip: strip.pt, reverse=True)
        if self.verbosity >= 2:
            print("#signal_strips = %i" % len(signal_strips))
            if self.verbosity >= 3:
                print("signal_strips")
                for strip in signal_strips:
                    strip.print()

        jet_iso_cands = self.selectIsolationCands(jet.constituents)
        if self.verbosity >= 2:
            print("#jet_iso_cands = %i" % len(jet_iso_cands))
            if self.verbosity >= 3:
                print("jet_iso_cands:")
                for cand in jet_iso_cands:
                    cand.print()

        event_iso_cands = self.selectIsolationCands(event_iso_cands)
        event_iso_cands = selectCandsByDeltaR(
            event_iso_cands, jet, self.cfg.isolationConeSize + self.cfg.matchingConeSize
        )
        event_iso_cands = self.cleanCands(event_iso_cands, jet.constituents)
        if self.verbosity >= 2:
            print("#event_iso_cands = %i" % len(event_iso_cands))
            print("event_iso_cands:")
            for cand in event_iso_cands:
                cand.print()

        tau_candidates = []
        barcode = 0
        for decayMode, cfgDecayMode in self.cfg.decayModes.items():
            if self.verbosity >= 4:
                print(
                    "decayMode = %s: numChargedCands = %i, numStrips = %i"
                    % (
                        decayMode,
                        cfgDecayMode["numChargedCands"],
                        cfgDecayMode["numStrips"],
                    )
                )

            decayMode_numChargedCands = cfgDecayMode["numChargedCands"]
            if len(signal_chargedCands) < decayMode_numChargedCands:
                continue

            decayMode_numStrips = cfgDecayMode["numStrips"]
            selectedStrips = []
            if decayMode_numStrips > 0 and len(signal_strips) > 0:
                minStripMass = cfgDecayMode["minStripMass"]
                maxStripMass = cfgDecayMode["maxStripMass"]
                for strip in signal_strips:
                    if strip.mass > minStripMass and strip.mass < maxStripMass:
                        selectedStrips.append(strip)
            if self.verbosity >= 4:
                print("selectedStrips = %i" % len(selectedStrips))
            if len(selectedStrips) < decayMode_numStrips:
                continue

            chargedCandCombos = self.combinatorics.generate(
                decayMode_numChargedCands,
                min(len(signal_chargedCands), cfgDecayMode["maxChargedCands"]),
            )
            if self.verbosity >= 4:
                print("chargedCandCombos = %s" % chargedCandCombos)
            stripCombos = self.combinatorics.generate(
                decayMode_numStrips, min(len(selectedStrips), cfgDecayMode["maxStrips"])
            )
            if self.verbosity >= 4:
                print("stripCombos = %s" % stripCombos)

            numChargedCandCombos = len(chargedCandCombos)
            for idxChargedCandCombo in range(numChargedCandCombos):
                chargedCandCombo = chargedCandCombos[idxChargedCandCombo]
                assert len(chargedCandCombo) == decayMode_numChargedCands
                chargedCands = [
                    signal_chargedCands[chargedCandCombo[idx]]
                    for idx in range(decayMode_numChargedCands)
                ]

                numStripCombos = len(stripCombos)
                for idxStripCombo in range(max(1, numStripCombos)):
                    stripCombo = []
                    strips = []
                    if idxStripCombo < numStripCombos:
                        stripCombo = stripCombos[idxStripCombo]
                        assert len(stripCombo) == decayMode_numStrips
                        strips = [
                            selectedStrips[stripCombo[idx]]
                            for idx in range(decayMode_numStrips)
                        ]
                    if self.verbosity >= 4:
                        print(
                            "Processing combination of chargedCands = %s & strips = %s"
                            % (chargedCandCombo, stripCombo)
                        )

                    cleanedStrips = self.cleanStrips(strips, chargedCands)
                    if self.verbosity >= 4:
                        print("#cleanedStrips = %i" % len(cleanedStrips))
                    if len(cleanedStrips) < decayMode_numStrips:
                        continue
                    # CV: reverse=True argument needed in order to sort strips in order of decreasing (and NOT increasing) pT
                    cleanedStrips.sort(key=lambda strip: strip.pt, reverse=True)

                    tau_candidate = Tau(chargedCands, cleanedStrips, barcode)
                    tau_candidate.jet = jet
                    tau_candidate.decayMode = decayMode
                    tau_candidate.signalConeSize = max(
                        min(0.10, 3.0 / tau_candidate.pt), 0.05
                    )
                    passesSignalCone = True
                    for cand in tau_candidate.signal_chargedCands:
                        if (
                            f.deltaR_thetaPhi(
                                tau_candidate.theta,
                                tau_candidate.phi,
                                cand.theta,
                                cand.phi,
                            )
                            > tau_candidate.signalConeSize
                        ):
                            passesSignalCone = False
                            break
                    for strip in tau_candidate.signal_strips:
                        if (
                            f.deltaR_thetaPhi(
                                tau_candidate.theta,
                                tau_candidate.phi,
                                strip.theta,
                                strip.phi,
                            )
                            > tau_candidate.signalConeSize
                        ):
                            passesSignalCone = False
                            break
                    if (
                        abs(round(tau_candidate.q)) == 1
                        and f.deltaR_thetaPhi(
                            tau_candidate.theta,
                            tau_candidate.phi,
                            tau_candidate.jet.theta,
                            tau_candidate.jet.phi,
                        )
                        < self.cfg.matchingConeSize
                        and passesSignalCone
                        and tau_candidate.mass
                        > self.cfg.decayModes[decayMode]["minTauMass"]
                        and tau_candidate.mass
                        < self.cfg.decayModes[decayMode]["maxTauMass"]
                    ):
                        tau_iso_cands = selectCandsByDeltaR(
                            jet_iso_cands, tau_candidate, self.cfg.isolationConeSize
                        )
                        tau_iso_cands = self.cleanCands(
                            tau_iso_cands, tau_candidate.signal_cands
                        )
                        tau_iso_cands.extend(event_iso_cands)

                        tau_candidate.iso_cands = tau_iso_cands
                        tau_candidate.iso_chargedCands = selectCandsByPdgId(
                            tau_iso_cands, [11, 13, 211]
                        )
                        tau_candidate.iso_gammaCands = selectCandsByPdgId(
                            tau_iso_cands, [22]
                        )
                        tau_candidate.iso_neutralHadronCands = selectCandsByPdgId(
                            tau_iso_cands, [130]
                        )
                        # if self.metric == "eta-phi" or self.metric == "theta-phi":
                        #     tau_candidate.chargedIso_dR0p5 = comp_pt_sum(tau_candidate.iso_chargedCands)
                        #     tau_candidate.gammaIso_dR0p5 = comp_pt_sum(tau_candidate.iso_gammaCands)
                        #     tau_candidate.neutralHadronIso_dR0p5 = comp_pt_sum(tau_candidate.iso_neutralHadronCands)
                        tau_candidate.chargedIso_dR0p5 = comp_pt_sum(
                            tau_candidate.iso_chargedCands
                        )
                        tau_candidate.gammaIso_dR0p5 = comp_pt_sum(
                            tau_candidate.iso_gammaCands
                        )
                        tau_candidate.neutralHadronIso_dR0p5 = comp_pt_sum(
                            tau_candidate.iso_neutralHadronCands
                        )
                        # elif self.metric == "angle3d":
                        #     tau_candidate.chargedIso_dR0p5 = comp_p_sum(tau_candidate.iso_chargedCands)
                        #     tau_candidate.gammaIso_dR0p5 = comp_p_sum(tau_candidate.iso_gammaCands)
                        #     tau_candidate.neutralHadronIso_dR0p5 = comp_p_sum(tau_candidate.iso_neutralHadronCands)
                        # else:
                        #     raise RuntimeError("Invalid configuration parameter 'metric' = '%s' !!" % self.metric)
                        # CV: don't use neutral hadrons when computing the isolation of the tau
                        tau_candidate.combinedIso_dR0p5 = (
                            tau_candidate.chargedIso_dR0p5
                            + tau_candidate.gammaIso_dR0p5
                        )

                        # CV: constant alpha choosen such that idDiscr varies smoothly between 0 and 1
                        #     for typical values of the combined isolation pT-sum
                        alpha = 0.2
                        tau_candidate.idDiscr = math.exp(
                            -alpha * tau_candidate.combinedIso_dR0p5
                        )

                        if self.verbosity >= 3:
                            tau_candidate.print()
                        tau_candidates.append(tau_candidate)
                        barcode += 1
                    else:
                        if self.verbosity >= 4:
                            print("fails preselection:")
                            print(" q = %i" % round(tau_candidate.q))
                            print(
                                " dR(tau,jet) = %1.2f"
                                % f.deltaR_thetaPhi(
                                    tau_candidate.theta,
                                    tau_candidate.phi,
                                    tau_candidate.jet.theta,
                                    tau_candidate.jet.phi,
                                )
                            )
                            print(
                                " signalConeSize = %1.2f" % tau_candidate.signalConeSize
                            )
                            for idx, cand in enumerate(
                                tau_candidate.signal_chargedCands
                            ):
                                print(
                                    " dR(tau,signal_chargedCand #%i) = %1.2f"
                                    % (
                                        idx,
                                        f.deltaR_thetaPhi(
                                            tau_candidate.theta,
                                            tau_candidate.phi,
                                            cand.theta,
                                            cand.phi,
                                        ),
                                    )
                                )
                            for idx, strip in enumerate(
                                tau_candidate.signal_chargedCands
                            ):
                                print(
                                    " dR(tau,signal_strip #%i) = %1.2f"
                                    % (
                                        idx,
                                        f.deltaR_thetaPhi(
                                            tau_candidate.theta,
                                            tau_candidate.phi,
                                            strip.theta,
                                            strip.phi,
                                        ),
                                    )
                                )
                            print(" mass = %1.2f" % tau_candidate.mass)

        # CV: sort tau candidates by multiplicity of charged signal candidates,
        #     pT, multiplicity of strips, and combined isolation (in that order);
        #     reverse=True argument needed in order to sort tau candidates in order of decreasing (and NOT increasing) rank
        tau_candidates.sort(key=cmp_to_key(rank_tau_candidates), reverse=True)
        if self.verbosity >= 2:
            print("#tau_candidates = %i" % len(tau_candidates))

        tau = None
        if len(tau_candidates) > 0:
            tau = tau_candidates[0]
        return tau


class HPSTauBuilder:
    def __init__(self, cfg, verbosity=0):
        self.cfg = cfg
        self.verbosity = verbosity
        self.hpsAlgo = HPSAlgo(self.cfg.builder, self.verbosity)

    def print_config(self):
        primitive_cfg = OmegaConf.to_container(self.cfg)
        print(json.dumps(primitive_cfg, indent=4))

    def process_jets(self, data):
        if self.verbosity >= 3:
            print("data:")
            print(data.fields)

        jets = readJets(data)
        event_cands = readCands(data)

        taus = []
        for idxJet, jet in enumerate(jets):
            if self.verbosity >= 2:
                print("Processing entry %i" % idxJet)
                jet.print()
            elif idxJet > 0 and (idxJet % 100) == 0:
                print("Processing entry %i" % idxJet)
            # CV: enable the following two lines for faster turn-around time when testing
            # if idxJet > 10:
            #    continue

            event_iso_cands = event_cands[idxJet]
            # CV: reverse=True argument needed in order to sort candidates in order of decreasing (and NOT increasing) pT)
            event_iso_cands.sort(key=lambda cand: cand.pt, reverse=True)
            if self.verbosity >= 4:
                print("event_iso_cands:")
                for cand in event_iso_cands:
                    cand.print()

            tau = self.hpsAlgo.buildTau(jet, event_iso_cands)
            if tau is None:
                if self.verbosity >= 2:
                    print("Failed to find tau associated to jet:")
                    jet.print()
                    print(" -> building dummy tau")
                # CV: build "dummy" tau to maintain 1-to-1 correspondence between taus and jets
                tau = Tau()
                tau.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
                tau.updatePtEtaPhiMass()
                tau.signal_cands = set()
                tau.signal_gammaCands = set()
                tau.iso_cands = set()
                tau.iso_chargedCands = set()
                tau.iso_gammaCands = set()
                tau.iso_neutralHadronCands = set()
                tau.metric_dR_or_angle = None
                tau.metric_dEta_or_dTheta = None
                tau.idDiscr = -1.0
                tau.q = 0.0
                tau.decayMode = "undefined"
                tau.barcode = -1
            if self.verbosity >= 2:
                tau.print()
            if self.verbosity >= 4 and idxJet > 100:
                raise ValueError("STOP.")
            taus.append(tau)

        retVal = writeTaus(taus)
        return retVal
