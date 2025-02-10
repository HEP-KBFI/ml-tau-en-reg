#!/bin/bash

rm -Rf training-outputs/*

#From scratch
singularity exec --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha/gabbro ~/HEP-KBFI/singularity/pytorch.simg.2 python3 enreg/scripts/trainModel.py \
    output_dir=training-outputs/from_scratch trainSize=1e5 fraction_valid=0.1 training.num_epochs=2 model_type=OmniParT \
    training_type=dm_multiclass models.OmniParT.version=from_scratch
mv batch_*.pkl embeds/from_scratch/

#Fixed backbone
singularity exec --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha/gabbro ~/HEP-KBFI/singularity/pytorch.simg.2 python3 enreg/scripts/trainModel.py \
    output_dir=training-outputs/fixed_backbone trainSize=1e5 fraction_valid=0.1 training.num_epochs=2 model_type=OmniParT \
    training_type=dm_multiclass models.OmniParT.version=fixed_backbone models.OmniParT.bb_path=weights/OmniJet_generative_model_FiduciaryCagoule_254.ckpt
mv batch_*.pkl embeds/fixed_backbone/

singularity exec --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha/gabbro ~/HEP-KBFI/singularity/pytorch.simg.2 python3 enreg/scripts/trainModel.py \
    output_dir=training-outputs/finetuned_finetuned_binary_classification trainSize=1e5 fraction_valid=0.1 training.num_epochs=2 model_type=OmniParT \
    training_type=dm_multiclass models.OmniParT.version=fixed_backbone \
    models.OmniParT.bb_path=weights/trainfrac_1e6/binary_classification/OmniParT_fine_tuning/model_best.pt
mv batch_*.pkl embeds/finetuned_binary_classification/

singularity exec --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha/gabbro ~/HEP-KBFI/singularity/pytorch.simg.2 python3 enreg/scripts/trainModel.py \
    output_dir=training-outputs/finetuned_dm_multiclass trainSize=1e5 fraction_valid=0.1 training.num_epochs=2 model_type=OmniParT \
    training_type=dm_multiclass models.OmniParT.version=fixed_backbone \
    models.OmniParT.bb_path=weights/trainfrac_1e6/dm_multiclass/OmniParT_fine_tuning/model_best.pt
mv batch_*.pkl embeds/finetuned_dm_multiclass/

singularity exec --env PYTHONPATH=`pwd`:`pwd`/enreg/omnijet_alpha/gabbro ~/HEP-KBFI/singularity/pytorch.simg.2 python3 enreg/scripts/trainModel.py \
    output_dir=training-outputs/finetuned_jet_regression trainSize=1e5 fraction_valid=0.1 training.num_epochs=2 model_type=OmniParT \
    training_type=dm_multiclass models.OmniParT.version=fixed_backbone \
    models.OmniParT.bb_path=weights/trainfrac_1e6/jet_regression/OmniParT_fine_tuning/model_best.pt
mv batch_*.pkl embeds/finetuned_jet_regression/