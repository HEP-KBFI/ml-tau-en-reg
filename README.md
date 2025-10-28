# Machine learned reconstruction and identification of hadronically decaying tau lepton.

The aim of this project is to develop and test end-to-end machine learning methods for reconstruction and identification of hadronically decaying tau lepton, while also providing a thouroughly validated and tested dataset for evaluating the performances of said algorithms.

<img src="images/idea.png" width="100%"/>

Tau leptons can decay both leptonically and hadronically, however only hadronic decays are targeted with this project:

<img src="images/tau_decays.png" width="50%"/>

---

---

# Papers:

The results of these studies have been divided across two separate papers, with the first one covering tau identification and the latter covering both kinematic and decay mode reconstruction.

## TauID [![DOI:10.1016/j.cpc.2024.109095](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109095-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109095) [![arXiv](https://img.shields.io/badge/arXiv-2307.07747-b31b1b.svg)](https://arxiv.org/abs/2307.07747)

**"Tau lepton identification and reconstruction: a new frontier for jet-tagging ML algorithms"**

_[Published in: Comput.Phys.Commun. 298 (2024) 109095]_

In this paper, we studied the performance of state-of-the-art methods and compared them with the ML architectures initially designed for jet-tagging.

<img src="images/ROC_tauClassifier.png" width="100%"/>


---

## Tau kinematic and decay mode reconstruction [![DOI:10.1016/j.cpc.2024.109399](http://img.shields.io/badge/DOI-10.1016/j.cpc.2024.109399-f9f107.svg)](https://doi.org/10.1016/j.cpc.2024.109399) [![arXiv](https://img.shields.io/badge/arXiv-2407.06788-b31b1b.svg)](https://arxiv.org/abs/2407.06788)

_[Published in: Comput.Phys.Commun. 307 (2025) 109399]_

**"A unified machine learning approach for reconstructing hadronically decaying tau leptons"**

Here we demonstrated how three different types of models with a varying degree of expressiveness and priors can be employed for hadronically decaying tau kinematic reconstruction and decay mode reconstruction.


<img src="images/best_losses_reso.png" width="100%"/>


<img src="images/ZH_cm_ParticleTransformer_dm_precision.png" width="100%"/>

---

## Tau reconstruction and identification using a foundation model [![DOI:10.1016/j.cpc.2024.109399](http://img.shields.io/badge/DOI-10.21468/SciPostPhysCore.8.3.046-f9f107.svg)](https://doi.org/10.21468/SciPostPhysCore.8.3.046) [![arXiv](https://img.shields.io/badge/arXiv-2503.19165-b31b1b.svg)](https://arxiv.org/abs/2503.19165)

_[Published in: SciPost Phys. Core 8, 046 (2025)]_

**"Reconstructing hadronically decaying tau leptons with a jet foundation model"**

In this paper we demonstrate how a jet foundation model, Omnijet-alpha, can successfully be utilized for a out-of-domain and out-of-context tasks such as hadronically decaying tau lepton identification, kinematic reconstruction and decay mode classification.


<img src="images/10k_dm_reg_id.png" width="100%"/>

<img src="images/tsne.png" width="100%"/>
