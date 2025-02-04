import numba
import uproot
import vector
import fastjet
import numpy as np
import awkward as ak
from particle import pdgid
from enreg.tools import general as g
from enreg.tools.data_management import lifeTimeTools as lt


def load_single_file_contents(
        path: str,
        tree_path: str = "events",
        branches: list = [
            "MCParticles",
            "MergedRecoParticles",
            "SiTracks_Refitted_1",
            "PrimaryVertices",
            "MCParticles#1.index"
        ],
) -> ak.Array:
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        print(f"ROOT file has {tree.num_entries} entries")
        arrays = tree.arrays(branches)
        idx0 = "RecoMCTruthLink#0/RecoMCTruthLink#0.index"
        idx1 = "RecoMCTruthLink#1/RecoMCTruthLink#1.index"
        idx_recoparticle = tree.arrays(idx0)[idx0]
        idx_mc_particlesarticle = tree.arrays(idx1)[idx1]
        # index in the MergedRecoParticles collection
        arrays["idx_reco"] = idx_recoparticle
        # index in the MCParticles collection
        arrays["idx_mc"] = idx_mc_particlesarticle
        # index the track collection
        idx3 = "MergedRecoParticles#1/MergedRecoParticles#1.index"
        idx_recoparticle_track = tree.arrays(idx3)[idx3]
        arrays["idx_track"] = idx_recoparticle_track
    return arrays


def calculate_p4(p_type: str, arrays: ak.Array):
    particles = ak.Record({k.replace(f"{p_type}.", ""): arrays[k] for k in arrays.fields if p_type in k})
    particle_p4 = vector.awk(
        ak.zip(
            {
                "mass": particles["mass"],
                "px": particles["momentum.x"],
                "py": particles["momentum.y"],
                "pz": particles["momentum.z"],
            }
        )
    )
    return particles, particle_p4


def cluster_jets(particles_p4, min_pt=5.0):
    jetdef = fastjet.JetDefinition2Param(fastjet.ee_genkt_algorithm, 0.4, -1)
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=min_pt))
    jets = vector.awk(ak.zip({"energy": jets["t"], "x": jets["x"], "y": jets["y"], "z": jets["z"]}))
    constituent_index = ak.Array(cluster.constituent_index(min_pt=min_pt))
    njets = np.sum(ak.num(jets))
    print(f"clustered {njets} jets")
    return jets, constituent_index


###############################################################################
###############################################################################
###############           MATCH TAUS WITH GEN JETS               ##############
###############################################################################
###############################################################################


def get_all_tau_best_combinations(vis_tau_p4s, gen_jets):
    vis_tau_p4s = ak.zip(
        {
            "pt": vis_tau_p4s.pt,
            "eta": vis_tau_p4s.eta,
            "phi": vis_tau_p4s.phi,
            "energy": vis_tau_p4s.energy,
        }
    )
    gen_jets_p4 = ak.zip(
        {
            "pt": gen_jets.pt,
            "eta": gen_jets.eta,
            "phi": gen_jets.phi,
            "energy": gen_jets.energy,
        }
    )
    tau_indices, gen_indices = match_jets(vis_tau_p4s, gen_jets_p4, 0.4)
    pairs = []
    for tau_idx, gen_idx in zip(tau_indices, gen_indices):
        pair = []
        for i in range(len(tau_idx)):
            pair.append([tau_idx[i], gen_idx[i]])
        pairs.append(pair)
    return ak.Array(pairs)


###############################################################################
###############################################################################
###############################################################################
###############################################################################


@numba.njit
def deltar(eta1, phi1, eta2, phi2):
    deta = np.abs(eta1 - eta2)
    dphi = deltaphi(phi1, phi2)
    return np.sqrt(deta ** 2 + dphi ** 2)


@numba.njit
def deltaphi(phi1, phi2):
    diff = phi1 - phi2
    return np.arctan2(np.sin(diff), np.cos(diff))


@numba.njit
def match_jets(jets1, jets2, deltaR_cut):
    iev = len(jets1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = jets1[ev]
        j2 = jets2[ev]

        jet_inds_1 = []
        jet_inds_2 = []
        for ij1 in range(len(j1)):
            if j1[ij1].energy == 0:
                continue
            drs = np.zeros(len(j2), dtype=np.float64)
            for ij2 in range(len(j2)):
                if j2[ij2].energy == 0:
                    continue
                eta1 = j1.eta[ij1]
                eta2 = j2.eta[ij2]
                phi1 = j1.phi[ij1]
                phi2 = j2.phi[ij2]

                # Workaround for https://github.com/scikit-hep/vector/issues/303
                # dr = j1[ij1].deltaR(j2[ij2])
                dr = deltar(eta1, phi1, eta2, phi2)
                drs[ij2] = dr
            if len(drs) > 0:
                min_idx_dr = np.argmin(drs)
                if drs[min_idx_dr] < deltaR_cut:
                    jet_inds_1.append(ij1)
                    jet_inds_2.append(min_idx_dr)
        jet_inds_1_ev.append(jet_inds_1)
        jet_inds_2_ev.append(jet_inds_2)
    return jet_inds_1_ev, jet_inds_2_ev


def get_jet_constituent_p4s(reco_p4, constituent_idx, num_ptcls_per_jet):
    reco_p4_flat = reco_p4[ak.flatten(constituent_idx, axis=-1)]
    ret = ak.from_iter(
        [ak.unflatten(reco_p4_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])
    return vector.awk(ak.zip({"x": ret.x, "y": ret.y, "z": ret.z, "mass": ret.tau}))


def get_jet_constituent_property(property_, constituent_idx, num_ptcls_per_jet):
    reco_property_flat = property_[ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))]
    )


def to_vector(jet):
    return vector.awk(
        ak.zip(
            {
                "pt": jet.pt,
                "eta": jet.eta,
                "phi": jet.phi,
                "energy": jet.energy,
            }
        )
    )


def map_pdgid_to_candid(pdg_id):
    if pdgid.is_hadron(pdg_id):
        if abs(pdgid.charge(pdg_id)) > 0:
            return 211  # charged hadron
        else:
            return 130  # neutral hadron
    else:
        return abs(pdg_id)


def get_matched_gen_jet_p4(reco_jets, gen_jets):
    reco_jets_ = to_vector(reco_jets)
    gen_jets_ = to_vector(gen_jets)
    reco_indices, gen_indices = match_jets(reco_jets_, gen_jets_, deltaR_cut=0.3)
    return reco_indices, gen_indices


def get_matched_gen_tau_property(gen_jets, best_combos, property_, dummy_value=-1):
    gen_jet_full_info_array = []
    for event_id in range(len(gen_jets)):
        mapping = {i[1]: i[0] for i in best_combos[event_id]}
        gen_jet_info_array = []
        for i, gen_jet in enumerate(gen_jets[event_id]):
            if len(best_combos[event_id]) > 0:
                if i in best_combos[event_id][:, 1]:
                    value = property_[event_id][mapping[i]]
                    gen_jet_info_array.append(value)
                else:
                    gen_jet_info_array.append(dummy_value)
            else:
                gen_jet_info_array.append(dummy_value)
        gen_jet_full_info_array.append(gen_jet_info_array)
    return ak.Array(gen_jet_full_info_array)


def retrieve_tau_jet_info(arrays: ak.Array, gen_jets):
    tau_info = {
        "tau_DV_x": [],
        "tau_DV_y": [],
        "tau_DV_z": [],
        "tau_full_p4s": [],
        "tau_decaymodes": [],
        "tau_vis_p4s": [],
        "tau_charges": [],
        "daughter_PDG": []
    }
    for event_idx, event in enumerate(arrays):
        idx_map = arrays['MCParticles#1.index'][event_idx]
        d_begin = arrays['MCParticles.daughters_begin'][event_idx]
        d_end = arrays['MCParticles.daughters_end'][event_idx]
        tau_mask = (np.abs(arrays['MCParticles.PDG'][event_idx]) == 15) * (
                arrays['MCParticles.generatorStatus'][event_idx] == 2)
        tau_indices = np.where(tau_mask)[0]
        tau_daughters = []
        for tau_idx in tau_indices:
            daughter_indices = list(range(d_begin[tau_idx], d_end[tau_idx]))
            new_indices = idx_map[daughter_indices]
            tau_daughters.append(new_indices)
        # TODO: Nüüd kontrolli lifetime muutujaid uuesti
        tau_info["tau_DV_x"].append([arrays['MCParticles.endpoint.x'][event_idx][tau_idx] for tau_idx in tau_indices])
        tau_info["tau_DV_y"].append([arrays['MCParticles.endpoint.y'][event_idx][tau_idx] for tau_idx in tau_indices])
        tau_info["tau_DV_z"].append([arrays['MCParticles.endpoint.x'][event_idx][tau_idx] for tau_idx in tau_indices])
        event_particle_p4s = vector.awk(ak.zip({
            "mass": arrays['MCParticles.mass'][event_idx],
            "x": arrays['MCParticles.momentum.x'][event_idx],
            "y": arrays['MCParticles.momentum.y'][event_idx],
            "z": arrays['MCParticles.momentum.z'][event_idx]})
        )
        tau_info["tau_full_p4s"].append([event_particle_p4s[tau_idx] for tau_idx in tau_indices])
        tau_info["tau_charges"].append(
            [pdgid.charge(arrays['MCParticles.PDG'][event_idx][tau_idx]) for tau_idx in tau_indices])
        info = retrieve_tau_info(tau_daughters, len(tau_indices), arrays[event_idx], event_particle_p4s)
        for key, value in info.items():
            tau_info[key].append(value)
    for key, value in tau_info.items():
        tau_info[key] = ak.Array(value)
    tau_vis_p4 = g.reinitialize_p4(tau_info['tau_vis_p4s'])
    best_combos = get_all_tau_best_combinations(tau_vis_p4, gen_jets)
    tau_gen_jet_p4s_fill_value = vector.awk(
        ak.zip(
            {
                "mass": [0.0],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
            }
        )
    )[0]
    gen_tau_jet_info = {
        "gen_jet_tau_vis_energy": get_matched_gen_tau_property(gen_jets, best_combos, tau_vis_p4.energy,
                                                               dummy_value=0),
        "gen_jet_tau_decaymode": get_matched_gen_tau_property(gen_jets, best_combos, tau_info['tau_decaymodes']),
        "tau_gen_jet_charge": get_matched_gen_tau_property(gen_jets, best_combos, tau_info['tau_charges'],
                                                           dummy_value=-999),
        "tau_gen_jet_p4s_full": get_matched_gen_tau_property(
            gen_jets, best_combos, tau_info['tau_full_p4s'], dummy_value=tau_gen_jet_p4s_fill_value),
        "tau_gen_jet_p4s": get_matched_gen_tau_property(
            gen_jets, best_combos, tau_vis_p4, dummy_value=tau_gen_jet_p4s_fill_value),
        "tau_gen_jet_DV_x": get_matched_gen_tau_property(gen_jets, best_combos, tau_info['tau_DV_x']),
        "tau_gen_jet_DV_y": get_matched_gen_tau_property(gen_jets, best_combos, tau_info['tau_DV_y']),
        "tau_gen_jet_DV_z": get_matched_gen_tau_property(gen_jets, best_combos, tau_info['tau_DV_z']),
    }
    return gen_tau_jet_info


def retrieve_tau_info(tau_daughters, n_taus, arrays, event_particle_p4s):
    tau_decay_modes = []
    tau_vis_p4s = []
    daughter_pdgs = []
    for tau_idx in range(n_taus):
        daughter_pdgs = [arrays['MCParticles.PDG'][d_idx] for d_idx in tau_daughters[tau_idx]]
        pdgs = [map_pdgid_to_candid(pdg_id) for pdg_id in daughter_pdgs]
        tau_vis_p4 = vector.awk(
            ak.zip(
                {
                    "mass": [0.0],
                    "x": [0.0],
                    "y": [0.0],
                    "z": [0.0],
                }
            )
        )[0]
        for tc in tau_daughters[tau_idx]:
            daughter_p4 = event_particle_p4s[tc]
            if abs(arrays['MCParticles.PDG'][tc]) not in [12, 14, 16]:
                tau_vis_p4 = tau_vis_p4 + daughter_p4
        tau_vis_p4s.append(tau_vis_p4)
        tau_decay_modes.append(g.get_decaymode(pdgs))
        daughter_pdgs.append(pdgs)
    tau_info = {
        "tau_decaymodes": tau_decay_modes,
        "tau_vis_p4s": tau_vis_p4s,
        "daughter_PDG": daughter_pdgs
    }
    return tau_info


def get_stable_mc_particles(mc_particles, mc_p4):
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (
            abs(mc_particles["PDG"]) != 16)
    particle_mask = stable_pythia_mask * neutrino_mask
    mc_particles = ak.Record({field: mc_particles[field][particle_mask] for field in mc_particles.fields})
    mc_p4 = g.reinitialize_p4(mc_p4[particle_mask])
    return mc_p4, mc_particles


def get_reco_particle_pdg(reco_particles):
    reco_particle_pdg = []
    for i in range(len(reco_particles.charge)):
        pdgs = ak.flatten(reco_particles["type"][i], axis=-1).to_numpy()
        mapped_pdgs = ak.from_iter([map_pdgid_to_candid(pdgs[j]) for j in range(len(pdgs))])
        reco_particle_pdg.append(mapped_pdgs)
    return ak.from_iter(reco_particle_pdg)


def clean_reco_particles(reco_particles, reco_p4):
    mask = reco_particles["type"] != 0
    reco_particles = ak.Record({field: reco_particles[field][mask] for field in reco_particles.fields})
    reco_p4 = g.reinitialize_p4(reco_p4[mask])
    return reco_particles, reco_p4


def filter_gen_jets(gen_jets, gen_jet_constituent_indices, stable_mc_particles):
    """Filter out all gen jets that have a lepton as one of their consituents (so in dR < 0.4)
    Currently see that also some jets with 6 hadrons and an electron are filtered out
    Roughly 90% of gen jets will be left after filtering
    """
    gen_num_ptcls_per_jet = ak.num(gen_jet_constituent_indices, axis=-1)
    gen_jet_pdgs = get_jet_constituent_property(stable_mc_particles.PDG, gen_jet_constituent_indices,
                                                gen_num_ptcls_per_jet)
    mask = []
    for gj_pdg in gen_jet_pdgs:
        sub_mask = []
        for gjp in gj_pdg:
            if (15 in np.abs(gjp)) or (13 in np.abs(gjp)):
                sub_mask.append(False)
            else:
                sub_mask.append(True)
        mask.append(sub_mask)
    mask = ak.Array(mask)
    return gen_jets[mask], gen_jet_constituent_indices[mask]


def get_genmatched_reco_particles_properties(reco_p4, mc_p4, reco_particles, mc_particles):
    v_mc_p4 = to_vector(mc_p4)
    v_reco_p4 = to_vector(reco_p4)
    reco_part_indices, gen_part_indices = match_jets(v_reco_p4, v_mc_p4, 0.01)
    matched_gen_PDG = []
    for ev, (ri, gi) in enumerate(zip(reco_part_indices, gen_part_indices)):
        matched_gen_PDG_ = np.ones(len(reco_particles.charge[ev]), dtype=int) * (-1)
        matched_gen_PDG_[ri] = mc_particles.PDG[ev][gi]
        matched_gen_PDG.append(matched_gen_PDG_)
    return ak.from_iter(matched_gen_PDG)


def match_Z_parton_to_reco_jet(mc_particles, mc_p4, reco_jets):
    all_daughter_PDGs = []
    all_daughter_p4s = []
    for ev in range(len(mc_particles.PDG)):
        mask = mc_particles.PDG[ev] == 23
        daughter_idx = range(mc_particles.daughters_begin[ev][mask][-1], mc_particles.daughters_end[ev][mask][-1])
        daughter_PDGs = mc_particles.PDG[ev][daughter_idx]
        daughter_p4s = mc_p4[ev][daughter_idx]
        all_daughter_PDGs.append(daughter_PDGs)
        all_daughter_p4s.append(daughter_p4s)
    all_daughter_PDGs = ak.from_iter(all_daughter_PDGs)
    all_daughter_p4s = ak.from_iter(all_daughter_p4s)
    all_daughter_p4s = g.reinitialize_p4(all_daughter_p4s)
    v_all_daughter_p4s = to_vector(all_daughter_p4s)
    reco_jets = to_vector(reco_jets)
    jet_indices, Z_daughter_indices = match_jets(reco_jets, v_all_daughter_p4s, 0.4)
    jet_parton_PDGs = []
    for ev in range(len(jet_indices)):
        ev_jet_parton_PDGs = np.zeros(len(reco_jets[ev]))
        if len(jet_indices[ev]) > 0:
            ev_jet_parton_PDGs[jet_indices[ev]] = all_daughter_PDGs[ev][Z_daughter_indices[ev]]
        jet_parton_PDGs.append(ev_jet_parton_PDGs)
    jet_parton_PDGs = ak.from_iter(jet_parton_PDGs)
    return jet_parton_PDGs


def no_tau_genjet_matching(gen_jets):
    filler = ak.zeros_like(gen_jets)
    gen_tau_jet_info = {
        "gen_jet_tau_vis_energy": ak.values_astype(ak.Array(ak.zeros_like(gen_jets) == ak.ones_like(gen_jets)), int),
        "gen_jet_tau_decaymode": ak.values_astype(ak.Array(ak.ones_like(gen_jets) == ak.ones_like(gen_jets)), int) * -1,
        "tau_gen_jet_charge": ak.values_astype(ak.Array(ak.ones_like(gen_jets) == ak.ones_like(gen_jets)), int) * -999,
        "tau_gen_jet_p4s_full": vector.awk(
            ak.zip({"mass": filler.mass, "px": filler.x, "py": filler.y, "pz": filler.z})
        ),
        "tau_gen_jet_p4s": vector.awk(
            ak.zip({"mass": filler.mass, "px": filler.x, "py": filler.y, "pz": filler.z})
        ),
        "tau_gen_jet_DV_x": ak.values_astype(ak.Array(ak.zeros_like(gen_jets) == ak.ones_like(gen_jets)), int),
        "tau_gen_jet_DV_y": ak.values_astype(ak.Array(ak.zeros_like(gen_jets) == ak.ones_like(gen_jets)), int),
        "tau_gen_jet_DV_z": ak.values_astype(ak.Array(ak.zeros_like(gen_jets) == ak.ones_like(gen_jets)), int),
    }
    return gen_tau_jet_info


def retrieve_stable_gen_particles(mc_particles, mc_p4):
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (
            abs(mc_particles["PDG"]) != 16)
    particle_mask = stable_pythia_mask * neutrino_mask
    stable_mc_particles = ak.Record({field: mc_particles[field][particle_mask] for field in mc_particles.fields})
    stable_mc_p4 = g.reinitialize_p4(mc_p4[particle_mask])
    return stable_mc_p4, stable_mc_particles


def process_input_file(input_path: str, tree_path: str, branches: list, remove_background: bool):
    arrays = load_single_file_contents(input_path, tree_path, branches)
    reco_particles, reco_p4 = calculate_p4(p_type="MergedRecoParticles", arrays=arrays)
    mc_particles, mc_p4 = calculate_p4(p_type="MCParticles", arrays=arrays)
    reco_particles, reco_p4 = clean_reco_particles(reco_particles=reco_particles, reco_p4=reco_p4)
    reco_jets, reco_jet_constituent_indices = cluster_jets(reco_p4, min_pt=0.0)
    stable_mc_p4, stable_mc_particles = retrieve_stable_gen_particles(mc_particles, mc_p4)
    gen_jets, gen_jet_constituent_indices = cluster_jets(stable_mc_p4, min_pt=0.0)
    gen_jets, gen_jet_constituent_indices = filter_gen_jets(gen_jets, gen_jet_constituent_indices, stable_mc_particles)
    reco_indices, gen_indices = get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jet_constituent_indices = ak.from_iter(
        [reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = g.reinitialize_p4(reco_jets)
    gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
    gen_jets = g.reinitialize_p4(gen_jets)
    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)
    if remove_background:
        gen_tau_jet_info = retrieve_tau_jet_info(arrays, gen_jets)
    else:
        gen_tau_jet_info = no_tau_genjet_matching(gen_jets)

    event_reco_cand_p4s = ak.from_iter([[reco_p4[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_lifetime_infos = ak.from_iter([lt.findTrackPCAs(arrays, i) for i in range(len(reco_p4))])
    event_lifetime_info = event_lifetime_infos[:, 0]
    event_lifetime_errs = event_lifetime_infos[:, 1]
    event_dxy = event_lifetime_info[:, :, 0]
    event_dz = event_lifetime_info[:, :, 1]
    event_d3 = event_lifetime_info[:, :, 2]
    event_d0 = event_lifetime_info[:, :, 3]
    event_z0 = event_lifetime_info[:, :, 4]

    event_dxy_f2D = event_lifetime_info[:, :, 5]
    event_dz_f2D = event_lifetime_info[:, :, 6]
    event_d3_f2D = event_lifetime_info[:, :, 7]
    event_PCA_x = event_lifetime_info[:, :, 8]
    event_PCA_y = event_lifetime_info[:, :, 9]
    event_PCA_z = event_lifetime_info[:, :, 10]
    event_PV_x = event_lifetime_info[:, :, 11]
    event_PV_y = event_lifetime_info[:, :, 12]
    event_PV_z = event_lifetime_info[:, :, 13]
    event_phi0 = event_lifetime_info[:, :, 14]
    event_tanL = event_lifetime_info[:, :, 15]
    event_omega = event_lifetime_info[:, :, 16]
    event_dxy_err = event_lifetime_errs[:, :, 0]
    event_dz_err = event_lifetime_errs[:, :, 1]
    event_d3_err = event_lifetime_errs[:, :, 2]
    event_d0_err = event_lifetime_errs[:, :, 3]
    event_z0_err = event_lifetime_errs[:, :, 4]
    event_dxy_f2D_err = event_lifetime_errs[:, :, 5]
    event_dz_f2D_err = event_lifetime_errs[:, :, 6]
    event_d3_f2D_err = event_lifetime_errs[:, :, 7]
    event_PCA_x_err = event_lifetime_errs[:, :, 8]
    event_PCA_y_err = event_lifetime_errs[:, :, 9]
    event_PCA_z_err = event_lifetime_errs[:, :, 10]
    event_reco_cand_dxy = ak.from_iter(
        [[event_dxy[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_dz = ak.from_iter([[event_dz[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_d3 = ak.from_iter([[event_d3[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_d0 = ak.from_iter([[event_d0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_z0 = ak.from_iter([[event_z0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_dxy_f2D = ak.from_iter(
        [[event_dxy_f2D[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_dz_f2D = ak.from_iter(
        [[event_dz_f2D[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_d3_f2D = ak.from_iter(
        [[event_d3_f2D[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_x = ak.from_iter(
        [[event_PCA_x[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PCA_y = ak.from_iter(
        [[event_PCA_y[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PCA_z = ak.from_iter(
        [[event_PCA_z[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_x = ak.from_iter(
        [[event_PV_x[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_y = ak.from_iter(
        [[event_PV_y[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_z = ak.from_iter(
        [[event_PV_z[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_phi0 = ak.from_iter(
        [[event_phi0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_tanL = ak.from_iter(
        [[event_tanL[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_omega = ak.from_iter(
        [[event_omega[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_dxy_err = ak.from_iter(
        [[event_dxy_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_dz_err = ak.from_iter(
        [[event_dz_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_d3_err = ak.from_iter(
        [[event_d3_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_d0_err = ak.from_iter(
        [[event_d0_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_z0_err = ak.from_iter(
        [[event_z0_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_dxy_f2D_err = ak.from_iter(
        [[event_dxy_f2D_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_dz_f2D_err = ak.from_iter(
        [[event_dz_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_d3_f2D_err = ak.from_iter(
        [[event_d3_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_x_err = ak.from_iter(
        [[event_PCA_x_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_y_err = ak.from_iter(
        [[event_PCA_y_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_z_err = ak.from_iter(
        [[event_PCA_z_err[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_signed_dxy = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_dxy[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_dz = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_dz[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_d3 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_d3[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_d0 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_d0[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_z0 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_z0[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_dxy_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_dxy_f2D[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_dz_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_dz_f2D[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    event_reco_cand_signed_d3_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    event_reco_cand_d3_f2D[j][i],
                    [event_reco_cand_PCA_x[j][i], event_reco_cand_PCA_y[j][i], event_reco_cand_PCA_z[j][i]],
                    [event_reco_cand_PV_x[j][i], event_reco_cand_PV_y[j][i], event_reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_dxy = get_jet_constituent_property(event_dxy, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dz = get_jet_constituent_property(event_dz, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d3 = get_jet_constituent_property(event_d3, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d0 = get_jet_constituent_property(event_d0, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_z0 = get_jet_constituent_property(event_z0, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dxy_f2D = get_jet_constituent_property(event_dxy_f2D, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dz_f2D = get_jet_constituent_property(event_dz_f2D, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d3_f2D = get_jet_constituent_property(event_d3_f2D, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_x = get_jet_constituent_property(event_PCA_x, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_y = get_jet_constituent_property(event_PCA_y, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_z = get_jet_constituent_property(event_PCA_z, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PV_x = get_jet_constituent_property(event_PV_x, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PV_y = get_jet_constituent_property(event_PV_y, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PV_z = get_jet_constituent_property(event_PV_z, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_phi0 = get_jet_constituent_property(event_phi0, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_tanL = get_jet_constituent_property(event_tanL, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_omega = get_jet_constituent_property(event_omega, reco_jet_constituent_indices, num_ptcls_per_jet)

    reco_cand_dxy_err = get_jet_constituent_property(event_dxy_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dz_err = get_jet_constituent_property(event_dz_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d3_err = get_jet_constituent_property(event_d3_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dxy_f2D_err = get_jet_constituent_property(event_dxy_f2D_err, reco_jet_constituent_indices,
                                                         num_ptcls_per_jet)
    reco_cand_dz_f2D_err = get_jet_constituent_property(event_dz_f2D_err, reco_jet_constituent_indices,
                                                        num_ptcls_per_jet)
    reco_cand_d3_f2D_err = get_jet_constituent_property(event_d3_f2D_err, reco_jet_constituent_indices,
                                                        num_ptcls_per_jet)
    reco_cand_signed_dxy = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_dxy[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_dz = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_dz[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_d3 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_d3[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_dxy_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_dxy_f2D[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_dz_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_dz_f2D[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_d3_f2D = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_d3_f2D[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_d0 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_d0[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_signed_z0 = ak.from_iter(
        [
            [
                lt.calculateImpactParameterSigns(
                    reco_cand_z0[j][i],
                    [reco_cand_PCA_x[j][i], reco_cand_PCA_y[j][i], reco_cand_PCA_z[j][i]],
                    [reco_cand_PV_x[j][i], reco_cand_PV_y[j][i], reco_cand_PV_z[j][i]],
                    reco_jets[j][i],
                )
                for i in range(len(reco_jets[j]))
            ]
            for j in range(len(reco_jets))
        ]
    )
    reco_cand_d0_err = get_jet_constituent_property(event_d0_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_z0_err = get_jet_constituent_property(event_z0_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_x_err = get_jet_constituent_property(event_PCA_x_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_y_err = get_jet_constituent_property(event_PCA_y_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_z_err = get_jet_constituent_property(event_PCA_z_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_particle_pdg = get_reco_particle_pdg(reco_particles)
    # IP variables documented below and more detailed in src/lifeTimeTools.py
    event_reco_cand_p4s = g.reinitialize_p4(event_reco_cand_p4s)
    event_cand_ordering_mask = ak.argsort(event_reco_cand_p4s.pt, axis=2, ascending=False)
    reco_cand_p4s = get_jet_constituent_p4s(reco_p4, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_ordering_mask = ak.argsort(reco_cand_p4s.pt, axis=2, ascending=False)
    data = {
        "event_reco_cand_p4s": event_reco_cand_p4s[event_cand_ordering_mask],
        "event_reco_cand_pdg": ak.from_iter(
            [[reco_particle_pdg[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        )[event_cand_ordering_mask],
        "event_reco_cand_charge": ak.from_iter(
            [[reco_particles["charge"][j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        )[event_cand_ordering_mask],
        "reco_cand_p4s": reco_cand_p4s[reco_cand_ordering_mask],
        "reco_cand_charge": get_jet_constituent_property(
            reco_particles["charge"], reco_jet_constituent_indices, num_ptcls_per_jet
        )[reco_cand_ordering_mask],
        "reco_cand_pdg":
            get_jet_constituent_property(reco_particle_pdg, reco_jet_constituent_indices, num_ptcls_per_jet)[
                reco_cand_ordering_mask],
        "reco_jet_p4s": vector.awk(
            ak.zip({"mass": reco_jets.mass, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z})
        ),
        # "reco_jet_Z_Dparton_pdg": jet_parton_PDGs,
        # "reco_cand_genPDG": get_jet_constituent_property(
        #     reco_particle_genPDG, reco_jet_constituent_indices, num_ptcls_per_jet
        # ),
        "event_reco_cand_dxy": event_reco_cand_dxy[event_cand_ordering_mask],
        # impact parameter in xy  for all pf in event
        "event_reco_cand_dz": event_reco_cand_dz[event_cand_ordering_mask],  # impact parameter in z for all pf in event
        "event_reco_cand_d3": event_reco_cand_d3[event_cand_ordering_mask],
        # impact parameter in 3d for all pf in event
        "event_reco_cand_dxy_f2D": event_reco_cand_dxy_f2D[event_cand_ordering_mask],
        # impact parameter in xy  for all pf in event (PCA found in 2DA)
        "event_reco_cand_dz_f2D": event_reco_cand_dz_f2D[event_cand_ordering_mask],
        # impact parameter in z for all pf in event (PCA found in 2DA)
        "event_reco_cand_d3_f2D": event_reco_cand_d3_f2D[event_cand_ordering_mask],
        # impact parameter in 3d for all pf in event (PCA found in 2DA)
        "event_reco_cand_dxy_err": event_reco_cand_dxy_err[event_cand_ordering_mask],
        # xy impact parameter error (all pf)
        "event_reco_cand_dz_err": event_reco_cand_dz_err[event_cand_ordering_mask],  # z impact parameter error (all pf)
        "event_reco_cand_d3_err": event_reco_cand_d3_err[event_cand_ordering_mask],
        # 3d impact parameter error (all pf)
        "event_reco_cand_dxy_f2D_err": event_reco_cand_dxy_f2D_err[event_cand_ordering_mask],
        # xy impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_dz_f2D_err": event_reco_cand_dz_f2D_err[event_cand_ordering_mask],
        # z impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_d3_f2D_err": event_reco_cand_d3_f2D_err[event_cand_ordering_mask],
        # 3d impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_signed_dxy": event_reco_cand_signed_dxy[event_cand_ordering_mask],
        # impact parameter in xy for all pf in event (jet sign)
        "event_reco_cand_signed_dz": event_reco_cand_signed_dz[event_cand_ordering_mask],
        # impact parameter in z for all pf in event (jet sign)
        "event_reco_cand_signed_d3": event_reco_cand_signed_d3[event_cand_ordering_mask],
        # impact parameter in 3d for all pf in event (jet sign)
        "event_reco_cand_signed_dxy_f2D": event_reco_cand_signed_dxy_f2D[event_cand_ordering_mask],
        # ip prm in xy for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_signed_dz_f2D": event_reco_cand_signed_dz_f2D[event_cand_ordering_mask],
        # ip prm in z for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_signed_d3_f2D": event_reco_cand_signed_d3_f2D[event_cand_ordering_mask],
        # ip prm in 3d for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_d0": event_reco_cand_d0[event_cand_ordering_mask],
        # track parameter, xy distance to reference point
        "event_reco_cand_z0": event_reco_cand_z0[event_cand_ordering_mask],
        # track parameter, z distance to reference point
        "event_reco_cand_d0_err": event_reco_cand_d0_err[event_cand_ordering_mask],  # track parameter error
        "event_reco_cand_z0_err": event_reco_cand_z0_err[event_cand_ordering_mask],  # track parameter error
        "event_reco_cand_signed_d0": event_reco_cand_signed_d0[event_cand_ordering_mask],
        # track prm, xy distance to reference point (jet sign)
        "event_reco_cand_signed_z0": event_reco_cand_signed_z0[event_cand_ordering_mask],
        # track prm, z distance to reference point (jet sign)
        "event_reco_cand_PCA_x": event_reco_cand_PCA_x[event_cand_ordering_mask],  # closest approach to PV (x-comp)
        "event_reco_cand_PCA_y": event_reco_cand_PCA_y[event_cand_ordering_mask],  # closest approach to PV (y-comp)
        "event_reco_cand_PCA_z": event_reco_cand_PCA_z[event_cand_ordering_mask],  # closest approach to PV (z-comp)
        "event_reco_cand_PCA_x_err": event_reco_cand_PCA_x_err[event_cand_ordering_mask],  # PCA error (x-comp)
        "event_reco_cand_PCA_y_err": event_reco_cand_PCA_y_err[event_cand_ordering_mask],  # PCA error (y-comp)
        "event_reco_cand_PCA_z_err": event_reco_cand_PCA_z_err[event_cand_ordering_mask],  # PCA error (z-comp)
        "event_reco_cand_PV_x": event_reco_cand_PV_x[event_cand_ordering_mask],  # primary vertex (PX) x-comp
        "event_reco_cand_PV_y": event_reco_cand_PV_y[event_cand_ordering_mask],  # primary vertex (PX) y-comp
        "event_reco_cand_PV_z": event_reco_cand_PV_z[event_cand_ordering_mask],  # primary vertex (PX) z-comp
        "event_reco_cand_phi0": event_reco_cand_phi0[event_cand_ordering_mask],
        "event_reco_cand_tanL": event_reco_cand_tanL[event_cand_ordering_mask],
        "event_reco_cand_omega": event_reco_cand_omega[event_cand_ordering_mask],
        "reco_cand_phi0": reco_cand_phi0[reco_cand_ordering_mask],
        "reco_cand_tanL": reco_cand_tanL[reco_cand_ordering_mask],
        "reco_cand_omega": reco_cand_omega[reco_cand_ordering_mask],
        "reco_cand_dxy": reco_cand_dxy[reco_cand_ordering_mask],  # impact parameter in xy
        "reco_cand_dz": reco_cand_dz[reco_cand_ordering_mask],  # impact parameter in z
        "reco_cand_d3": reco_cand_d3[reco_cand_ordering_mask],  # impact parameter in 3D
        "reco_cand_dxy_f2D": reco_cand_dxy_f2D[reco_cand_ordering_mask],  # impact parameter in xy (PCA found in 2DA)
        "reco_cand_dz_f2D": reco_cand_dz_f2D[reco_cand_ordering_mask],  # impact parameter in z (PCA found in 2DA)
        "reco_cand_d3_f2D": reco_cand_d3_f2D[reco_cand_ordering_mask],  # impact parameter in 3D (PCA found in 2DA)
        "reco_cand_signed_dxy": reco_cand_signed_dxy[reco_cand_ordering_mask],  # impact parameter in xy (jet sign)
        "reco_cand_signed_dz": reco_cand_signed_dz[reco_cand_ordering_mask],  # impact parameter in z (jet sign)
        "reco_cand_signed_d3": reco_cand_signed_d3[reco_cand_ordering_mask],  # impact parameter in 3d (jet sign)
        "reco_cand_signed_dxy_f2D": reco_cand_signed_dxy_f2D[reco_cand_ordering_mask],
        # impact parameter in xy (jet sign) (PCA found in 2DA)
        "reco_cand_signed_dz_f2D": reco_cand_signed_dz_f2D[reco_cand_ordering_mask],
        # impact parameter in z (jet sign) (PCA found in 2DA)
        "reco_cand_signed_d3_f2D": reco_cand_signed_d3_f2D[reco_cand_ordering_mask],
        # impact parameter in 3d (jet sign) (PCA found in 2DA)
        "reco_cand_dxy_err": reco_cand_dxy_err[reco_cand_ordering_mask],  # xy impact parameter error
        "reco_cand_dz_err": reco_cand_dz_err[reco_cand_ordering_mask],  # z impact parameter error
        "reco_cand_d3_err": reco_cand_d3_err[reco_cand_ordering_mask],  # 3d impact parameter error
        "reco_cand_dxy_f2D_err": reco_cand_dxy_f2D_err[reco_cand_ordering_mask],
        # xy impact parameter error (PCA found in 2DA)
        "reco_cand_dz_f2D_err": reco_cand_dz_f2D_err[reco_cand_ordering_mask],
        # z impact parameter error (PCA found in 2DA)
        "reco_cand_d3_f2D_err": reco_cand_d3_f2D_err[reco_cand_ordering_mask],
        # 3d impact parameter error (PCA found in 2DA)
        "reco_cand_d0": reco_cand_d0[reco_cand_ordering_mask],  # track parameter, xy distance to reference point
        "reco_cand_z0": reco_cand_z0[reco_cand_ordering_mask],  # track parameter, z distance to reference point
        "reco_cand_signed_d0": reco_cand_signed_d0[reco_cand_ordering_mask],
        # track parameter, xy distance to reference point (jet sign)
        "reco_cand_signed_z0": reco_cand_signed_z0[reco_cand_ordering_mask],
        # track parameter, z distance to reference point (jet sign)
        "reco_cand_d0_err": reco_cand_d0_err[reco_cand_ordering_mask],  # track parameter error
        "reco_cand_z0_err": reco_cand_z0_err[reco_cand_ordering_mask],  # track parameter error
        "reco_cand_PCA_x": reco_cand_PCA_x[reco_cand_ordering_mask],  # closest approach to PV (x-comp)
        "reco_cand_PCA_y": reco_cand_PCA_y[reco_cand_ordering_mask],  # closest approach to PV (y-comp)
        "reco_cand_PCA_z": reco_cand_PCA_z[reco_cand_ordering_mask],  # closest approach to PV (z-comp)
        "reco_cand_PCA_x_err": reco_cand_PCA_x_err[reco_cand_ordering_mask],  # PCA error (x-comp)
        "reco_cand_PCA_y_err": reco_cand_PCA_y_err[reco_cand_ordering_mask],  # PCA error (y-comp)
        "reco_cand_PCA_z_err": reco_cand_PCA_z_err[reco_cand_ordering_mask],  # PCA error (z-comp)
        "reco_cand_PV_x": reco_cand_PV_x[reco_cand_ordering_mask],  # primary vertex (PX) x-comp
        "reco_cand_PV_y": reco_cand_PV_y[reco_cand_ordering_mask],  # primary vertex (PX) y-comp
        "reco_cand_PV_z": reco_cand_PV_z[reco_cand_ordering_mask],  # primary vertex (PX) z-comp
        "gen_jet_p4s": vector.awk(
            ak.zip({"mass": gen_jets.mass, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z})),
        "gen_jet_tau_decaymode": gen_tau_jet_info["gen_jet_tau_decaymode"],
        "gen_jet_tau_vis_energy": gen_tau_jet_info["gen_jet_tau_vis_energy"],
        "gen_jet_tau_charge": gen_tau_jet_info["tau_gen_jet_charge"],
        "gen_jet_tau_p4s": gen_tau_jet_info["tau_gen_jet_p4s"],
        "gen_jet_full_tau_p4s": gen_tau_jet_info["tau_gen_jet_p4s_full"],
        "gen_jet_tau_decay_vertex_x": gen_tau_jet_info["tau_gen_jet_DV_x"],
        "gen_jet_tau_decay_vertex_y": gen_tau_jet_info["tau_gen_jet_DV_y"],
        "gen_jet_tau_decay_vertex_z": gen_tau_jet_info["tau_gen_jet_DV_z"],
    }
    data = ak.Record({key: ak.flatten(value, axis=1) for key, value in data.items()})

    ## remove backgrounds for signal samples
    removal_mask = data.gen_jet_tau_decaymode != 16
    if remove_background:
        removal_mask = (data.gen_jet_tau_decaymode != -1) * removal_mask
    print(f"{np.sum(removal_mask)} jets after masking")
    data = ak.Record({key: data[key][removal_mask] for key in data.fields})
    return data
