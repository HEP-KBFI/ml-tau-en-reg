from enreg.tools import general as g
from enreg.tools.visualization import base as pl



def plot_decaymode_reconstruction(sig_data, algorithm_output_dir, classifier_cut, cfg):
    output_path = os.path.join(algorithm_output_dir, "decaymode_reconstruction.pdf")
    gen_tau_decaymodes = get_reduced_decaymodes(sig_data.gen_jet_tau_decaymode.to_numpy())
    reco_tau_decaymodes = get_reduced_decaymodes(sig_data.tau_decaymode.to_numpy())
    mapping = {
        0: r"$h^{\pm}$",
        1: r"$h^{\pm}\pi^{0}$",
        2: r"$h^{\pm}\pi^{0}\pi^{0}$",
        10: r"$h^{\pm}h^{\mp}h^{\pm}$",
        11: r"$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$",
        15: "Other",
    }
    gen_tau_decaymodes_ = gen_tau_decaymodes[sig_data.tauClassifier > classifier_cut]
    reco_tau_decaymodes_ = reco_tau_decaymodes[sig_data.tauClassifier > classifier_cut]
    categories = [value for value in mapping.values()]
    decaymode_info = {
        "gen": list(gen_tau_decaymodes_),
        "reco": list(reco_tau_decaymodes_),
        "categories": list(categories),
    }
    g.save_to_json(decaymode_info, output_path.replace(".pdf", ".json"))
    pl.plot_decaymode_correlation_matrix(
        true_cats=gen_tau_decaymodes_,
        pred_cats=reco_tau_decaymodes_,
        categories=categories,
        output_path=output_path,
        y_label=r"Reconstructed \tau decay mode",
        x_label=r"Generated \tau decay mode",
    )


def plot_tauClassifier_correlation(
    sig_data: ak.Array,
    output_path: str
):
    """ Plots the tauClassifier dependence on eta, pt and phi of the reference object

    Args:
        sig_data : ak.Array
            Data from the signal sample.
        output_path : str
            Destination for the plot to be saved.

    Returns:
        None
    """
    p4s = g.reinitialize_p4(sig_data["reco_jet_p4s"])
    tc = sig_data["tauClassifier"]
    for var in ["eta", "pt", "phi"]:
        variable = getattr(p4s, var)
        plt.scatter(variable, tc, alpha=0.3, marker="x")
        plt.title(var)
        # output_path = os.path.join(output_dir, f"tauClassifier_corr_{var}.pdf")
        plt.savefig(output_path, format="pdf")
        plt.close("all")


def get_data_masks(
    data: ak.Array,
    ref_obj: str,
    theta_min: int = 10,
    theta_max: int = 170,
    pt_min: int = 20
):
    """Retrieves the masks for both numerator and denominator by first initializing the objects and then doing the selection.

    Args:
        data : ak.Array
            The data used to create the masks
        ref_obj : str
            The reference object used for creating the denominator mask
        theta_min : int
            [default: 10] The minimum value for the cut in theta.
        theta_max : int
            [default: 170] The maximum value for the cut in theta.
        pt_min : int
            [default: 20] The minimum value for the cut in p_T.

    Returns:
        numerator_mask : ak.Array
            Mask for the data used for the numerator
        denominator_mask : ak.Array
            Mask for the data used for the denominator
    """
    denominator_mask = create_object_mask(g.reinitialize_p4(data[ref_obj]))
    numerator_mask = create_object_mask(g.reinitialize_p4(data.tau_p4s))
    numerator_mask = numerator_mask * denominator_mask
    return numerator_mask, denominator_mask


def create_object_mask(
    target: ak.Array,
    theta_min: int = 10,
    theta_max: int = 170,
    pt_min: int = 20
):
    """ Creates a mask to select objects.

    Args:
        target : ak.Array
            The object used to create the mask
        theta_min : int
            [default: 10] The minimum value for the cut in theta.
        theta_max : int
            [default: 170] The maximum value for the cut in theta.
        pt_min : int
            [default: 20] The minimum value for the cut in p_T.

    Returns:
        mask : ak.Array
    """
    pt_mask = target.pt > 20
    theta_mask1 = abs(np.rad2deg(target.theta)) < 170
    theta_mask2 = abs(np.rad2deg(target.theta)) > 10
    mask = pt_mask * theta_mask1 * theta_mask2
    return mask


def calculate_efficiencies_fakerates(raw_numerator_data, denominator_data, classifier_cuts):
    eff_fakes = []
    n_all = len(denominator_data)
    for cut in classifier_cuts:
        n_passing_cuts = len(raw_numerator_data[raw_numerator_data.tauClassifier > cut])
        eff_fake = n_passing_cuts / n_all
        eff_fakes.append(eff_fake)
    return eff_fakes


def calculate_region_eff_fake(raw_numerator_data, denominator_data, classifier_cuts, region):
    eff_fakes = []
    raw_numerator_data_p4 = g.reinitialize_p4(raw_numerator_data.tau_p4s)
    if region == "barrel":
        region_mask = 90 - np.abs(np.rad2deg(raw_numerator_data_p4.theta) - 90) >= 45
    elif region == "endcap":
        region_mask = 90 - np.abs(np.rad2deg(raw_numerator_data_p4.theta) - 90) < 45
    else:
        raise ValueError("Incorrect region")
    n_all = len(denominator_data[region_mask])
    for cut in classifier_cuts:
        classifier_mask = raw_numerator_data.tauClassifier > cut
        n_passing_cuts = len(raw_numerator_data[classifier_mask * region_mask])
        eff_fake = n_passing_cuts / n_all
        eff_fakes.append(eff_fake)
    return eff_fakes


# plot_all_metrics juppideks teha ja toolid siia lisada


# get_regional_tauClassifiers juppideks




def create_eff_fake_table(eff_data, fake_data, classifier_cuts, output_dir):
    for algorithm in eff_data.keys():
        algorithm_output_dir = os.path.join(output_dir, algorithm)
        eff_uncertainties = []
        efficiencies = []
        fake_uncertainties = []
        fakerates = []
        eff_denom = len(eff_data[algorithm]["denominator"].tauClassifier)
        # eff_denom_err = 1 / np.sqrt(eff_denom)
        fake_denom = len(fake_data[algorithm]["denominator"].tauClassifier)
        # fake_denom_err = 1 / np.sqrt(fake_denom)
        for classifier_cut in classifier_cuts:
            eff_num = sum(eff_data[algorithm]["numerator"].tauClassifier > classifier_cut)
            fake_num = sum(fake_data[algorithm]["numerator"].tauClassifier > classifier_cut)
            # eff_num_err = 1 / np.sqrt(eff_num)
            # fake_num_err = 1 / np.sqrt(fake_num)
            fakerate = fake_num / fake_denom
            efficiency = eff_num / eff_denom
            fake_binomial_err = np.sqrt(np.abs(fakerate * (1 - fakerate) / fake_denom))
            eff_binomial_err = np.sqrt(np.abs(efficiency * (1 - efficiency) / eff_denom))
            efficiencies.append(efficiency)
            eff_uncertainties.append(eff_binomial_err)
            fakerates.append(fakerate)
            fake_uncertainties.append(fake_binomial_err)
        create_table_entries(
            efficiencies, eff_uncertainties, fakerates, fake_uncertainties, classifier_cuts, algorithm_output_dir
        )


def create_table_entries(efficiencies, eff_err, fakerates, fake_err, classifier_cuts, output_dir):
    efficiencies = np.array(efficiencies)
    eff_err = np.array(eff_err)
    fakerates = np.array(fakerates)
    fake_err = np.array(fake_err)
    inverse_fake = 1 / fakerates
    relative_fake_err = fake_err / fakerates
    rel_fake_errs = inverse_fake * relative_fake_err
    working_points = {"Loose": 0.5, "Medium": 0.7, "Tight": 0.9}
    wp_values = {}
    for wp_name, wp_value in working_points.items():
        diff = abs(np.array(efficiencies) - wp_value)
        idx = np.argmin(diff)
        if not diff[idx] / wp_value > 0.3:
            cut = classifier_cuts[idx]
            wp_values[wp_name] = {
                "tauClassifier": cut,
                "fakerate": fakerates[idx],
                "efficiency": efficiencies[idx],
                "eff_err": eff_err[idx],
                "fake_err": fake_err[idx],
                "1/fake": inverse_fake[idx],
                "1/fake_err": rel_fake_errs[idx],
            }
        else:
            wp_values[wp_name] = {
                "tauClassifier": -1,
                "efficiency": wp_value,
                "fakerate": -1,
                "eff_err": -1,
                "fake_err": -1,
                "1/fake": -1,
                "1/fake_err": -1,
            }
    output_path = os.path.join(output_dir, "paper_table_entries.json")
    with open(output_path, "wt") as out_file:
        json.dump(wp_values, out_file, indent=4)
    return wp_values


def save_wps(efficiencies, classifier_cuts, algorithm_output_dir):
    working_points = {"Loose": 0.5, "Medium": 0.7, "Tight": 0.9}
    wp_file_path = os.path.join(algorithm_output_dir, "working_points.json")
    wp_values = {}
    for wp_name, wp_value in working_points.items():
        diff = abs(np.array(efficiencies) - wp_value)
        idx = np.argmin(diff)
        if not diff[idx] > 0.1:
            cut = classifier_cuts[idx]
        else:
            cut = -1
        wp_values[wp_name] = cut
    with open(wp_file_path, "wt") as out_file:
        json.dump(wp_values, out_file, indent=4)
    return wp_values["Tight"]
    # return wp_values["Medium"]