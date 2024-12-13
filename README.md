# End-to-end ML reconstruction and identification of hadronically decaying tau lepton.

The aim of this project is to develop and test end-to-end machine learning methods for reconstruction and identification of hadronically decaying tau lepton, while also providing a thouroughly validated and tested dataset for evaluating the performances of said algorithms.

<img src="images/idea.png" width="100%"/>

Tau leptons can decay both leptonically and hadronically, however only hadronic decays are targeted with this project:

<img src="images/tau_decays.png" width="50%"/>

## Future Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13881061.svg)](https://doi.org/10.5281/zenodo.13881061)

The dataset contains 2 signal samples (ZH->Ztautau and Z->tautau) and one background sample (Z->qq).
While the validation plots can be reproduced with [this script](notebooks/data_intro.ipynb), here is a selection of these:

The generator-level hadronically decaying tau visible transverse momentum:

<img src="images/gen_tau_visible_pt.png" width="50%"/>

The jet substructure of two neutral-hadronless decay modes:

<img src="images/jet_2D_shapes_ZH_DM0.png" width="50%"/>

<img src="images/jet_2D_shapes_ZH_DM3.png" width="50%"/>

---

---

## Papers:

The results of these studies have been divided across two separate papers, with the first one covering tau identification and the latter covering both kinematic and decay mode reconstruction.

### TauID [![DOI:10.1016/j.cpc.2024.109095](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109095-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109095) [![arXiv](https://img.shields.io/badge/arXiv-2307.07747-b31b1b.svg)](https://arxiv.org/abs/2307.07747)

**"Tau lepton identification and reconstruction: a new frontier for jet-tagging ML algorithms"**

_[Published in: Comput.Phys.Commun. 298 (2024) 109095]_

In this paper, we studied the performance of state-of-the-art methods and compared them with the ML architectures initially designed for jet-tagging.

<img src="images/ROC.png" width="50%"/>

<img src="images/ParticleTransformer_tauClassifier.png" width="50%"/>

---

### Tau reconstruction [![DOI:10.1016/j.cpc.2024.109399](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109399-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109399) [![arXiv](https://img.shields.io/badge/arXiv-2407.06788-b31b1b.svg)](https://arxiv.org/abs/2407.06788)

_[Published in: Comput.Phys.Commun. 307 (2025) 109399]_

**"A unified machine learning approach for reconstructing hadronically decaying tau leptons"**

Here we demonstrated how three different types of models with a varying degree of expressiveness and priors can be employed for hadronically decaying tau kinematic reconstruction and decay mode reconstruction.

<img src="images/resolutions_comp.png" width="50%"/>

<img src="images/best_losses.png" width="50%"/>

<img src="images/ZH_dm_precision.png" width="50%"/>

<img src="images/ZH_cm_ParticleTransformer.png" width="50%"/>
