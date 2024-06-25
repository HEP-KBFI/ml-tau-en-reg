import awkward as ak
import glob
import tqdm
import numpy as np
import vector


def to_p4(p4_obj):
    return vector.awk(
        ak.zip(
            {
                "mass": p4_obj.tau,
                "x": p4_obj.x,
                "y": p4_obj.y,
                "z": p4_obj.z,
            }
        )
    )


def load_sample(path):
    columns = [
        #basic reco inputs
        "reco_jet_p4s",
        "reco_cand_p4s",
        "reco_cand_charge",
        "reco_cand_pdg",

        #advanced reco inputs: track impact parameters
        "reco_cand_dz",
        "reco_cand_dz_err",
        "reco_cand_dxy",
        "reco_cand_dxy_err",

        #targets
        "gen_jet_p4s", #generated jet p4
        "gen_jet_tau_p4s", #tau visible momentum, excluding neutrino
        "gen_jet_tau_decaymode", #tau decay mode, as in 
    ]
    data = []
    for fi in tqdm.tqdm(list(glob.glob(path + "/*.parquet"))):
        ret = ak.from_parquet(fi, columns=columns)
        ret = ak.Array({k: ret[k] for k in ret.fields})

        #apply a cut on minimum genjet pt
        ret = ret[to_p4(ret["gen_jet_p4s"]).pt>10]
        data.append(ret)
    data = ak.concatenate(data)

    #shuffle data
    perm = np.random.permutation(len(data))
    data = data[perm]

    return data


def split_train_test(data, split=0.8):
    ndata = len(data)
    ntrain = int(ndata*split)
    data_train = data[:ntrain]
    data_test = data[ntrain:]
    print(f"N={ndata}, Ntrain={len(data_train)} Ntest={len(data_test)}")
    return data_train, data_test


if __name__ == "__main__":

    for sample_long, sample_short in [
        ("p8_ee_qq_ecm380", "qq"),
        ("p8_ee_ZH_Htautau_ecm380", "zh"),
        ("p8_ee_Z_Ztautau_ecm380", "z")
    ]:
        data = load_sample("/local/joosep/ml-tau-en-reg/ntuples/20240625_all_2M/" + sample_long)
        data_train, data_test = split_train_test(data)
        ak.to_parquet(data_train, sample_short + "_train.parquet", row_group_size=1024)
        ak.to_parquet(data_test, sample_short + "_test.parquet", row_group_size=1024)
