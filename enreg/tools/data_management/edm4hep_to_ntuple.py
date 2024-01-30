"""Script for ntupelizing the EDM4HEP dataformat to ML friendly format. To run the script in our singularity image:
           ./scripts/run-env.sh  src/edm4hep_to_ntuple.py [args]
Call with 'python3'
"""

import os
import time
import glob
import numba
import uproot
import hydra
import vector
import fastjet
import numpy as np
import general as g
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from lifeTimeTools import findTrackPCAs, calculateImpactParameterSigns


def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(ak.Record(data), output_path)


def load_single_file_contents(
    path: str,
    tree_path: str = "events",
    branches: list = ["MCParticles", "MergedRecoParticles", "SiTracks_Refitted_1", "PrimaryVertices"],
) -> ak.Array:
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
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


def calculate_p4(p_type: str, arrs: ak.Array):
    particles = arrs[p_type]
    particles = ak.Record({k.replace(f"{p_type}.", ""): particles[k] for k in particles.fields})
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


def cluster_jets(particles_p4):
    jetdef = fastjet.JetDefinition2Param(fastjet.ee_genkt_algorithm, 0.4, -1)
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=20.0))
    jets = vector.awk(ak.zip({"energy": jets["t"], "x": jets["x"], "y": jets["y"], "z": jets["z"]}))
    constituent_index = ak.Array(cluster.constituent_index(min_pt=20.0))
    return jets, constituent_index


###############################################################################
###############################################################################
#####                 TAU DECAYMODE CALCULATION                         #######
###############################################################################
###############################################################################


def get_all_tau_decaymodes(mc_particles, tau_mask, mask_addition):
    all_decaymodes = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        event_decaymodes = []
        if n_daughters == 0:
            event_decaymodes.append(-1)
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            daughter_pdgs = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            daughter_charges = np.abs(mc_particles.charge[e_idx][daughter_indices])
            PDGs = [map_pdgid_to_candid(pdgid, charge) for pdgid, charge in zip(daughter_pdgs, daughter_charges)]
            event_decaymodes.append(g.get_decaymode(PDGs, daughter_pdgs))
        all_decaymodes.append(event_decaymodes)
    return ak.from_iter(all_decaymodes)


###############################################################################
###############################################################################
#####                 TAU DECAY VERTEX FINDING                          #######
###############################################################################
###############################################################################


def get_all_tau_decayvertices(mc_particles, tau_mask, mask_addition):
    all_decay_vertices_x = []
    all_decay_vertices_y = []
    all_decay_vertices_z = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        event_decay_vertices_x = []
        event_decay_vertices_y = []
        event_decay_vertices_z = []
        if n_daughters == 0:
            event_decay_vertices_x.append(-1)
            event_decay_vertices_y.append(-1)
            event_decay_vertices_z.append(-1)
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            first_daughter = daughter_indices[0]
            tau_decay_vertex_x = mc_particles["vertex.x"][e_idx][first_daughter]
            tau_decay_vertex_y = mc_particles["vertex.y"][e_idx][first_daughter]
            tau_decay_vertex_z = mc_particles["vertex.z"][e_idx][first_daughter]
            event_decay_vertices_x.append(tau_decay_vertex_x)
            event_decay_vertices_y.append(tau_decay_vertex_y)
            event_decay_vertices_z.append(tau_decay_vertex_z)
        all_decay_vertices_x.append(event_decay_vertices_x)
        all_decay_vertices_y.append(event_decay_vertices_y)
        all_decay_vertices_z.append(event_decay_vertices_z)
    return ak.from_iter(all_decay_vertices_x), ak.from_iter(all_decay_vertices_y), ak.from_iter(all_decay_vertices_z)


###############################################################################
###############################################################################
###############              GET ALL TAU DAUGHTERS               ##############
###############################################################################
###############################################################################


def find_tau_daughters_all_generations(mc_particles, tau_mask, mask_addition):
    tau_daughters_all_events = []
    for event_idx in range(len(mc_particles.daughters_begin)):
        tau_daughter_indices = []
        for daughter_idx in range(len(mc_particles.daughters_begin[tau_mask][mask_addition][event_idx])):
            daughters = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][event_idx][daughter_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][event_idx][daughter_idx],
            )
            tau_daughter_indices.extend(daughters)
        event_tau_daughters_begin = mc_particles.daughters_begin[event_idx]
        event_tau_daughters_end = mc_particles.daughters_end[event_idx]
        all_tau_daughter_indices = get_event_tau_daughters(
            tau_daughter_indices, event_tau_daughters_begin, event_tau_daughters_end, mc_particles, event_idx
        )
        tau_daughters_all_events.append(all_tau_daughter_indices)
    return tau_daughters_all_events


def get_event_tau_daughters(
    tau_daughter_indices, event_tau_daughters_begin, event_tau_daughters_end, mc_particles, event_idx
):
    all_tau_daughter_indices = []
    all_tau_daughter_indices.extend(tau_daughter_indices)
    while len(tau_daughter_indices) != 0:
        new_tau_daughter_indices = []
        for tau_daughter_idx in tau_daughter_indices:
            daughter_mask = event_tau_daughters_end[tau_daughter_idx] < ak.num(
                mc_particles.daughters_begin[event_idx], axis=0
            )
            if len(mc_particles.daughters_begin[event_idx][tau_daughter_idx][daughter_mask]) > 0:
                daughters = list(
                    range(
                        event_tau_daughters_begin[tau_daughter_idx][daughter_mask][0],
                        event_tau_daughters_end[tau_daughter_idx][daughter_mask][0],
                    )
                )
                new_tau_daughter_indices.extend(daughters)
        tau_daughter_indices = new_tau_daughter_indices
        all_tau_daughter_indices.extend(new_tau_daughter_indices)
    return all_tau_daughter_indices


def get_jet_matched_constituent_gen_energy(
    arrays, reco_jet_constituent_indices, num_ptcls_per_jet, mc_p4, gen_tau_daughters
):
    maps = []
    for ridx, midx in zip(arrays["idx_reco"], arrays["idx_mc"]):
        maps.append(dict(zip(ridx, midx)))
    flat_indices = ak.flatten(reco_jet_constituent_indices, axis=-1)
    gen_energies = ak.from_iter(
        [
            ak.from_iter([mc_p4[ev_i][map_[i]].energy if i in gen_tau_daughters[ev_i] else 0 for i in ev])
            for ev_i, (ev, map_) in enumerate(zip(flat_indices, maps))
        ]
    )
    ret = ak.from_iter([ak.unflatten(gen_energies[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])
    return ret


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
    tau_indices, gen_indices = match_jets(vis_tau_p4s, gen_jets_p4, 0.1)
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
    return np.sqrt(deta**2 + dphi**2)


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
    ret = ak.from_iter([ak.unflatten(reco_p4_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])
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


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0
    # photon, electron, muon
    if abs(pdgid) in [22, 11, 13, 15, 16, 1, 2, 3, 4, 5, 21, 12, 14]:
        return abs(pdgid)
    # charged hadron
    if abs(charge) > 0:
        return 211
    # neutral hadron
    return 130


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


def get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4):
    all_events_tau_vis_p4s = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        tau_vis_p4s = []
        for d_idx in range(n_daughters):
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
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            for tau_daughter_p4 in mc_p4[e_idx][daughter_indices][vis_particle_map]:
                tau_vis_p4 = tau_vis_p4 + tau_daughter_p4
            tau_vis_p4s.append(tau_vis_p4)
        if len(tau_vis_p4s) > 0:
            all_events_tau_vis_p4s.append(tau_vis_p4s)
        else:
            all_events_tau_vis_p4s.append(
                vector.awk(
                    ak.zip(
                        {
                            "mass": [0.0],
                            "x": [0.0],
                            "y": [0.0],
                            "z": [0.0],
                        }
                    )
                )
            )
    all_events_tau_vis_p4s = g.reinitialize_p4(ak.from_iter(all_events_tau_vis_p4s))
    return all_events_tau_vis_p4s


def get_full_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4):
    all_events_tau_p4s = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        tau_p4s = []
        for d_idx in range(n_daughters):
            tau_p4s.append(mc_p4[tau_mask][mask_addition][e_idx][d_idx])
        if len(tau_p4s) > 0:
            all_events_tau_p4s.append(tau_p4s)
        else:
            all_events_tau_p4s.append(
                vector.awk(
                    ak.zip(
                        {
                            "mass": [0.0],
                            "x": [0.0],
                            "y": [0.0],
                            "z": [0.0],
                        }
                    )
                )
            )
    all_events_tau_p4s = g.reinitialize_p4(ak.from_iter(all_events_tau_p4s))
    return all_events_tau_p4s


def get_gen_tau_jet_info(gen_jets, tau_mask, mask_addition, mc_particles, mc_p4):
    vis_tau_p4s = get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4)
    full_tau_p4s = get_full_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4)
    best_combos = get_all_tau_best_combinations(vis_tau_p4s, gen_jets)
    tau_energies = vis_tau_p4s.energy
    tau_decaymodes = get_all_tau_decaymodes(mc_particles, tau_mask, mask_addition)
    tau_dv_x, tau_dv_y, tau_dv_z = get_all_tau_decayvertices(mc_particles, tau_mask, mask_addition)
    tau_charges = mc_particles.charge[tau_mask][mask_addition]
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
        "gen_jet_tau_vis_energy": get_matched_gen_tau_property(gen_jets, best_combos, tau_energies),
        "gen_jet_tau_decaymode": get_matched_gen_tau_property(gen_jets, best_combos, tau_decaymodes),
        "tau_gen_jet_charge": get_matched_gen_tau_property(gen_jets, best_combos, tau_charges),
        "tau_gen_jet_p4s_full": get_matched_gen_tau_property(
            gen_jets, best_combos, full_tau_p4s, dummy_value=tau_gen_jet_p4s_fill_value
        ),
        "tau_gen_jet_p4s": get_matched_gen_tau_property(
            gen_jets, best_combos, vis_tau_p4s, dummy_value=tau_gen_jet_p4s_fill_value
        ),
        "tau_gen_jet_DV_x": get_matched_gen_tau_property(gen_jets, best_combos, tau_dv_x),
        "tau_gen_jet_DV_y": get_matched_gen_tau_property(gen_jets, best_combos, tau_dv_y),
        "tau_gen_jet_DV_z": get_matched_gen_tau_property(gen_jets, best_combos, tau_dv_z),
    }
    return gen_tau_jet_info


def get_stable_mc_particles(mc_particles, mc_p4):
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (abs(mc_particles["PDG"]) != 16)
    particle_mask = stable_pythia_mask * neutrino_mask
    mc_particles = ak.Record({field: mc_particles[field][particle_mask] for field in mc_particles.fields})
    mc_p4 = g.reinitialize_p4(mc_p4[particle_mask])
    return mc_p4, mc_particles


def get_reco_particle_pdg(reco_particles):
    reco_particle_pdg = []
    for i in range(len(reco_particles.charge)):
        charges = ak.flatten(reco_particles["charge"][i], axis=-1).to_numpy()
        pdgs = ak.flatten(reco_particles["type"][i], axis=-1).to_numpy()
        mapped_pdgs = ak.from_iter([map_pdgid_to_candid(pdgs[j], charges[j]) for j in range(len(pdgs))])
        reco_particle_pdg.append(mapped_pdgs)
    return ak.from_iter(reco_particle_pdg)


def clean_reco_particles(reco_particles, reco_p4):
    mask = reco_particles["type"] != 0
    reco_particles = ak.Record({field: reco_particles[field][mask] for field in reco_particles.fields})
    reco_p4 = g.reinitialize_p4(reco_p4[mask])
    return reco_particles, reco_p4


def get_hadronically_decaying_hard_tau_masks(mc_particles):
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)
    mask_addition = []
    for i in range(len(mc_particles.PDG[tau_mask])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][i])
        daughter_mask = []
        for d in range(n_daughters):
            parent_idx = mc_particles.parents_begin[tau_mask][i][d]
            initial_tau = parent_idx < len(mc_particles.PDG[i])
            initial_tau_2 = mc_particles.daughters_end[tau_mask][i][d] < len(mc_particles.PDG[i])
            if initial_tau and initial_tau_2:
                parent_pdg = mc_particles.PDG[i][parent_idx]
                daughters_idx = range(
                    mc_particles.daughters_begin[tau_mask][i][d], mc_particles.daughters_end[tau_mask][i][d]
                )
                daughter_pdgs = np.abs(mc_particles.PDG[i][daughters_idx])
                daughter_charges = np.abs(mc_particles.charge[i][daughters_idx])
                PDGs = [map_pdgid_to_candid(pdgid, charge) for pdgid, charge in zip(daughter_pdgs, daughter_charges)]
                decaymode = g.get_decaymode(PDGs, daughter_pdgs)
                if decaymode != 16 and abs(parent_pdg) == 15:
                    daughter_mask.append(True)
                else:
                    daughter_mask.append(False)
            else:
                daughter_mask.append(False)
        mask_addition.append(daughter_mask)
    mask_addition = ak.Array(mask_addition)
    return tau_mask, mask_addition


def filter_gen_jets(gen_jets, gen_jet_constituent_indices, stable_mc_particles):
    """Filter out all gen jets that have a lepton as one of their consituents (so in dR < 0.4)
    Currently see that also some jets with 6 hadrons and an electron are filtered out
    Roughly 90% of gen jets will be left after filtering
    """
    gen_num_ptcls_per_jet = ak.num(gen_jet_constituent_indices, axis=-1)
    gen_jet_pdgs = get_jet_constituent_property(stable_mc_particles.PDG, gen_jet_constituent_indices, gen_num_ptcls_per_jet)
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
    return gen_jets[mask]


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
        #     print(daughter_PDGs, daughter_idx)
        daughter_p4s = mc_p4[ev][daughter_idx]
        all_daughter_PDGs.append(daughter_PDGs)
        all_daughter_p4s.append(daughter_p4s)
    all_daughter_PDGs = ak.from_iter(all_daughter_PDGs)
    all_daughter_p4s = ak.from_iter(all_daughter_p4s)
    all_daughter_p4s = g.reinitialize_p4(all_daughter_p4s)
    v_all_daughter_p4s = to_vector(all_daughter_p4s)
    reco_jets = to_vector(reco_jets)
    jet_indices, Z_daughter_indices = match_jets(reco_jets, v_all_daughter_p4s, 0.1)
    jet_parton_PDGs = []
    for ev in range(len(jet_indices)):
        ev_jet_parton_PDGs = np.zeros(len(reco_jets[ev]))
        if len(jet_indices[ev]) > 0:
            ev_jet_parton_PDGs[jet_indices[ev]] = all_daughter_PDGs[ev][Z_daughter_indices[ev]]
        jet_parton_PDGs.append(ev_jet_parton_PDGs)
    jet_parton_PDGs = ak.from_iter(jet_parton_PDGs)
    return jet_parton_PDGs


def process_input_file(arrays: ak.Array, remove_background):
    mc_particles, mc_p4 = calculate_p4(p_type="MCParticles", arrs=arrays)
    reco_particles, reco_p4 = calculate_p4(p_type="MergedRecoParticles", arrs=arrays)
    reco_particles, reco_p4 = clean_reco_particles(reco_particles=reco_particles, reco_p4=reco_p4)
    reco_jets, reco_jet_constituent_indices = cluster_jets(reco_p4)
    stable_mc_p4, stable_mc_particles = get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets, gen_jet_constituent_indices = cluster_jets(stable_mc_p4)
    gen_jets = filter_gen_jets(gen_jets, gen_jet_constituent_indices, stable_mc_particles)
    reco_indices, gen_indices = get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jet_constituent_indices = ak.from_iter([reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = g.reinitialize_p4(reco_jets)
    gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
    gen_jets = g.reinitialize_p4(gen_jets)
    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)
    tau_mask, mask_addition = get_hadronically_decaying_hard_tau_masks(mc_particles)
    gen_tau_jet_info = get_gen_tau_jet_info(gen_jets, tau_mask, mask_addition, mc_particles, mc_p4)

    reco_particle_genPDG = get_genmatched_reco_particles_properties(reco_p4, mc_p4, reco_particles, mc_particles)
    jet_parton_PDGs = match_Z_parton_to_reco_jet(mc_particles, mc_p4, reco_jets)

    gen_tau_daughters = find_tau_daughters_all_generations(mc_particles, tau_mask, mask_addition)
    event_reco_cand_p4s = ak.from_iter([[reco_p4[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_lifetime_infos = ak.from_iter([findTrackPCAs(arrays, i) for i in range(len(reco_p4))])
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
    event_reco_cand_dxy = ak.from_iter([[event_dxy[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
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
    event_reco_cand_PCA_x = ak.from_iter([[event_PCA_x[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PCA_y = ak.from_iter([[event_PCA_y[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PCA_z = ak.from_iter([[event_PCA_z[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_x = ak.from_iter([[event_PV_x[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_y = ak.from_iter([[event_PV_y[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PV_z = ak.from_iter([[event_PV_z[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_phi0 = ak.from_iter([[event_phi0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_tanL = ak.from_iter([[event_tanL[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_omega = ak.from_iter([[event_omega[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
    reco_cand_dxy_f2D_err = get_jet_constituent_property(event_dxy_f2D_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dz_f2D_err = get_jet_constituent_property(event_dz_f2D_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d3_f2D_err = get_jet_constituent_property(event_d3_f2D_err, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_signed_dxy = ak.from_iter(
        [
            [
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
                calculateImpactParameterSigns(
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
    data = {
        "event_reco_cand_p4s": g.reinitialize_p4(event_reco_cand_p4s),
        "event_reco_cand_pdg": ak.from_iter(
            [[reco_particle_pdg[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        ),
        "event_reco_cand_charge": ak.from_iter(
            [[reco_particles["charge"][j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        ),
        "reco_cand_p4s": get_jet_constituent_p4s(reco_p4, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_cand_charge": get_jet_constituent_property(
            reco_particles["charge"], reco_jet_constituent_indices, num_ptcls_per_jet
        ),
        "reco_cand_pdg": get_jet_constituent_property(reco_particle_pdg, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_jet_p4s": vector.awk(
            ak.zip({"mass": reco_jets.mass, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z})
        ),
        "reco_jet_Z_Dparton_pdg": jet_parton_PDGs,
        "reco_cand_genPDG": get_jet_constituent_property(
            reco_particle_genPDG, reco_jet_constituent_indices, num_ptcls_per_jet
        ),
        "event_reco_cand_dxy": event_reco_cand_dxy,  # impact parameter in xy  for all pf in event
        "event_reco_cand_dz": event_reco_cand_dz,  # impact parameter in z for all pf in event
        "event_reco_cand_d3": event_reco_cand_d3,  # impact parameter in 3d for all pf in event
        "event_reco_cand_dxy_f2D": event_reco_cand_dxy_f2D,  # impact parameter in xy  for all pf in event (PCA found in 2DA)
        "event_reco_cand_dz_f2D": event_reco_cand_dz_f2D,  # impact parameter in z for all pf in event (PCA found in 2DA)
        "event_reco_cand_d3_f2D": event_reco_cand_d3_f2D,  # impact parameter in 3d for all pf in event (PCA found in 2DA)
        "event_reco_cand_dxy_err": event_reco_cand_dxy_err,  # xy impact parameter error (all pf)
        "event_reco_cand_dz_err": event_reco_cand_dz_err,  # z impact parameter error (all pf)
        "event_reco_cand_d3_err": event_reco_cand_d3_err,  # 3d impact parameter error (all pf)
        "event_reco_cand_dxy_f2D_err": event_reco_cand_dxy_f2D_err,  # xy impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_dz_f2D_err": event_reco_cand_dz_f2D_err,  # z impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_d3_f2D_err": event_reco_cand_d3_f2D_err,  # 3d impact parameter error (all pf) (PCA found in 2DA)
        "event_reco_cand_signed_dxy": event_reco_cand_signed_dxy,  # impact parameter in xy for all pf in event (jet sign)
        "event_reco_cand_signed_dz": event_reco_cand_signed_dz,  # impact parameter in z for all pf in event (jet sign)
        "event_reco_cand_signed_d3": event_reco_cand_signed_d3,  # impact parameter in 3d for all pf in event (jet sign)
        "event_reco_cand_signed_dxy_f2D": event_reco_cand_signed_dxy_f2D,  # ip prm in xy for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_signed_dz_f2D": event_reco_cand_signed_dz_f2D,  # ip prm in z for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_signed_d3_f2D": event_reco_cand_signed_d3_f2D,  # ip prm in 3d for all pf in evt (j sign, 2D PCA)
        "event_reco_cand_d0": event_reco_cand_d0,  # track parameter, xy distance to referrence point
        "event_reco_cand_z0": event_reco_cand_z0,  # track parameter, z distance to referrence point
        "event_reco_cand_d0_err": event_reco_cand_d0_err,  # track parameter error
        "event_reco_cand_z0_err": event_reco_cand_z0_err,  # track parameter error
        "event_reco_cand_signed_d0": event_reco_cand_signed_d0,  # track prm, xy distance to referrence point (jet sign)
        "event_reco_cand_signed_z0": event_reco_cand_signed_z0,  # track prm, z distance to referrence point (jet sign)
        "event_reco_cand_PCA_x": event_reco_cand_PCA_x,  # closest approach to PV (x-comp)
        "event_reco_cand_PCA_y": event_reco_cand_PCA_y,  # closest approach to PV (y-comp)
        "event_reco_cand_PCA_z": event_reco_cand_PCA_z,  # closest approach to PV (z-comp)
        "event_reco_cand_PCA_x_err": event_reco_cand_PCA_x_err,  # PCA error (x-comp)
        "event_reco_cand_PCA_y_err": event_reco_cand_PCA_y_err,  # PCA error (y-comp)
        "event_reco_cand_PCA_z_err": event_reco_cand_PCA_z_err,  # PCA error (z-comp)
        "event_reco_cand_PV_x": event_reco_cand_PV_x,  # primary vertex (PX) x-comp
        "event_reco_cand_PV_y": event_reco_cand_PV_y,  # primary vertex (PX) y-comp
        "event_reco_cand_PV_z": event_reco_cand_PV_z,  # primary vertex (PX) z-comp
        "event_reco_cand_phi0": event_reco_cand_phi0,
        "event_reco_cand_tanL": event_reco_cand_tanL,
        "event_reco_cand_omega": event_reco_cand_omega,
        "reco_cand_phi0": reco_cand_phi0,
        "reco_cand_tanL": reco_cand_tanL,
        "reco_cand_omega": reco_cand_omega,
        "reco_cand_dxy": reco_cand_dxy,  # impact parameter in xy
        "reco_cand_dz": reco_cand_dz,  # impact parameter in z
        "reco_cand_d3": reco_cand_d3,  # impact parameter in 3D
        "reco_cand_dxy_f2D": reco_cand_dxy_f2D,  # impact parameter in xy (PCA found in 2DA)
        "reco_cand_dz_f2D": reco_cand_dz_f2D,  # impact parameter in z (PCA found in 2DA)
        "reco_cand_d3_f2D": reco_cand_d3_f2D,  # impact parameter in 3D (PCA found in 2DA)
        "reco_cand_signed_dxy": reco_cand_signed_dxy,  # impact parameter in xy (jet sign)
        "reco_cand_signed_dz": reco_cand_signed_dz,  # impact parameter in z (jet sign)
        "reco_cand_signed_d3": reco_cand_signed_d3,  # impact parameter in 3d (jet sign)
        "reco_cand_signed_dxy_f2D": reco_cand_signed_dxy_f2D,  # impact parameter in xy (jet sign) (PCA found in 2DA)
        "reco_cand_signed_dz_f2D": reco_cand_signed_dz_f2D,  # impact parameter in z (jet sign) (PCA found in 2DA)
        "reco_cand_signed_d3_f2D": reco_cand_signed_d3_f2D,  # impact parameter in 3d (jet sign) (PCA found in 2DA)
        "reco_cand_dxy_err": reco_cand_dxy_err,  # xy impact parameter error
        "reco_cand_dz_err": reco_cand_dz_err,  # z impact parameter error
        "reco_cand_d3_err": reco_cand_d3_err,  # 3d impact parameter error
        "reco_cand_dxy_f2D_err": reco_cand_dxy_f2D_err,  # xy impact parameter error (PCA found in 2DA)
        "reco_cand_dz_f2D_err": reco_cand_dz_f2D_err,  # z impact parameter error (PCA found in 2DA)
        "reco_cand_d3_f2D_err": reco_cand_d3_f2D_err,  # 3d impact parameter error (PCA found in 2DA)
        "reco_cand_d0": reco_cand_d0,  # track parameter, xy distance to referrence point
        "reco_cand_z0": reco_cand_z0,  # track parameter, z distance to referrence point
        "reco_cand_signed_d0": reco_cand_signed_d0,  # track parameter, xy distance to referrence point (jet sign)
        "reco_cand_signed_z0": reco_cand_signed_z0,  # track parameter, z distance to referrence point (jet sign)
        "reco_cand_d0_err": reco_cand_d0_err,  # track parameter error
        "reco_cand_z0_err": reco_cand_z0_err,  # track parameter error
        "reco_cand_PCA_x": reco_cand_PCA_x,  # closest approach to PV (x-comp)
        "reco_cand_PCA_y": reco_cand_PCA_y,  # closest approach to PV (y-comp)
        "reco_cand_PCA_z": reco_cand_PCA_z,  # closest approach to PV (z-comp)
        "reco_cand_PCA_x_err": reco_cand_PCA_x_err,  # PCA error (x-comp)
        "reco_cand_PCA_y_err": reco_cand_PCA_y_err,  # PCA error (y-comp)
        "reco_cand_PCA_z_err": reco_cand_PCA_z_err,  # PCA error (z-comp)
        "reco_cand_PV_x": reco_cand_PV_x,  # primary vertex (PX) x-comp
        "reco_cand_PV_y": reco_cand_PV_y,  # primary vertex (PX) y-comp
        "reco_cand_PV_z": reco_cand_PV_z,  # primary vertex (PX) z-comp
        "gen_jet_p4s": vector.awk(ak.zip({"mass": gen_jets.mass, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z})),
        "gen_jet_tau_decaymode": gen_tau_jet_info["gen_jet_tau_decaymode"],
        "gen_jet_tau_vis_energy": gen_tau_jet_info["gen_jet_tau_vis_energy"],
        "gen_jet_tau_charge": gen_tau_jet_info["tau_gen_jet_charge"],
        "gen_jet_tau_p4s": gen_tau_jet_info["tau_gen_jet_p4s"],
        "gen_jet_full_tau_p4s": gen_tau_jet_info["tau_gen_jet_p4s_full"],
        "gen_jet_tau_decay_vertex_x": gen_tau_jet_info["tau_gen_jet_DV_x"],
        "gen_jet_tau_decay_vertex_y": gen_tau_jet_info["tau_gen_jet_DV_y"],
        "gen_jet_tau_decay_vertex_z": gen_tau_jet_info["tau_gen_jet_DV_z"],
        "reco_cand_matched_gen_energy": get_jet_matched_constituent_gen_energy(
            arrays, reco_jet_constituent_indices, num_ptcls_per_jet, mc_p4, gen_tau_daughters
        ),
    }
    data = {key: ak.flatten(value, axis=1) for key, value in data.items()}

    ## remove ZH bkg part
    if remove_background:
        removal_mask = data["gen_jet_tau_decaymode"] != -1
        data = {key: value[removal_mask] for key, value in data.items()}
    return data


def process_single_file(
    input_path: str,
    output_dir: str,
    sample: str,
    tree_path: str = "events",
    branches: list = ["MCParticles", "MergedRecoParticles", "SiTracks_Refitted_1", "PrimaryVertices"],
):
    file_name = os.path.basename(input_path).replace(".root", ".parquet")
    output_ntuple_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_ntuple_path):
        # try:
        start_time = time.time()
        remove_bkg = sample == "ZH_Htautau"
        arrays = load_single_file_contents(input_path, tree_path, branches)
        data = process_input_file(arrays, remove_background=remove_bkg)
        save_record_to_file(data, output_ntuple_path)
        end_time = time.time()
        print(f"Finished processing in {end_time-start_time} s.")
    # except Exception:
    #     print(f"Broken input file at {input_path}")
    else:
        print("File already processed, skipping.")


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def process_all_input_files(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    for sample in cfg.samples_to_process:
        output_dir = cfg.samples[sample].output_dir
        input_dir = cfg.samples[sample].input_dir
        os.makedirs(output_dir, exist_ok=True)
        input_wcp = os.path.join(input_dir, "*.root")
        if cfg.test_run:
            n_files = 10
        else:
            n_files = None
        input_paths = glob.glob(input_wcp)[:n_files]
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=10)
            pool.starmap(process_single_file, zip(input_paths, repeat(output_dir), repeat(sample)))
        else:
            for path in input_paths:
                process_single_file(path, output_dir, sample)


if __name__ == "__main__":
    process_all_input_files()
