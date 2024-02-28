import os
import hydra
from omegaconf import DictConfig
from enreg.tools import general as g
from enreg.tools.metrics import energy_regression as er


@hydra.main(config_path="../config", config_name="benchmarking", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(os.path.expandvars(cfg.output_dir), exist_ok=True)
    algorithm_info = {}
    for algorithm, properties in cfg.metrics.regression.algorithms.items():
        if not properties.compare or properties.load_from_json:
            continue
        ntuples_dir = properties.ntuples_dir
        dataset_info = {}
        for dataset in cfg.comparison_datasets:
            sample_data = {}
            for sample in cfg.comparison_samples:
                input_dir = os.path.join(ntuples_dir, dataset, sample)
                unmasked_data =  g.load_all_data(input_loc=input_dir, n_files=cfg.n_comparison_files) # Hiljem kui teada täpselt mis columns vaja, siis lisada see
                wp_value = cfg.metrics.regression.classifier_WPs[cfg.metrics.regression.cls_wp]
                mask = (unmasked_data.gen_jet_tau_vis_energy > 1) * (unmasked_data.tauClassifier > wp_value)
                sample_data[sample] = unmasked_data[mask]
            dataset_info[dataset] = sample_data
        algorithm_info[algorithm] = dataset_info
    er.plot_energy_regression(algorithm_info, cfg)


if __name__ == '__main__':
    main()