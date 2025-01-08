#!/bin/bash

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o] [-j] [-d] [-b] [-r] [-m]
  -o : This is used to specify the output directory.
  -j : [OPTIONAL] Use this flag to run jet_regression task.
  -d : [OPTIONAL] Use this flag to run decay mode classification.
  -b : [OPTIONAL] Use this flag to run binary classifiction.
  -m : [OPTIONAL] Use this flag to run the training on 'manivald'. By default it is run on LUMI
  -r : [OPTIONAL] Use this flag to specify the number of repetitions to be run of each task for each algorithm
EOF
  exit 1
}

OUTPUT_EXISTS=false
RUN_ON_LUMI=true
RUN_JET_REGRESSION=false
RUN_DM_MULTICLASS=false
RUN_BINARY_CLS=false
NUMBER_REPETITIONS=3
while getopts 'o:jdbmr:' OPTION; do
  case $OPTION in
    o)
        BASE_DIR=$OPTARG
        OUTPUT_EXISTS=true
        ;;
    j) RUN_JET_REGRESSION=true ;;
    d) RUN_DM_MULTICLASS=true ;;
    b) RUN_BINARY_CLS=true ;;
    m) RUN_ON_LUMI=false ;;
    r) NUMBER_REPETITIONS=$OPTARG ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"

if [ "$OUTPUT_EXISTS" = false ] ; then
    echo "Output directory needs to be specified with the flag -o"
    exit 1
fi

echo Output will be saved into: $BASE_DIR
echo Run jet_regression: $RUN_JET_REGRESSION
echo Run dm_multiclass: $RUN_DM_MULTICLASS
echo Run binary_cls: $RUN_BINARY_CLS
echo Number of repetitions to be run: $NUMBER_REPETITIONS

TRAIN_SAMPS=z_train.parquet
TEST_SAMPS=z_test.parquet,zh_test.parquet
CLS_TRAIN_SAMPS=z_train.parquet,qq_train.parquet
CLS_TEST_SAMPS=z_test.parquet,zh_test.parquet,qq_test.parquet

if  [ "$RUN_ON_LUMI" = true ] ; then
    TRAINING_SCRIPT=enreg/scripts/submit-lumi-gpu.sh
else
    TRAINING_SCRIPT=enreg/scripts/train-pytorch-gpu.sh
fi

for i in `seq 1 $NUMBER_REPETITIONS`; do
    for trainSize in 2e3 1e4 1e5 1e6; do
        export OUTDIR=$BASE_DIR/v$i/trainfrac_$trainSize

        if [ "$RUN_JET_REGRESSION" = true ] ; then
            # echo Submitting jet_regression jobs for repetition $i and $trainSize training jets
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=LorentzNet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=DeepSet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniDeepSet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=ParticleTransformer comet.experiment=ParT_jr_$trainSize
            sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_jr_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_jr_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=jet_regression model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_jr_$trainSize
        fi

        if [ "$RUN_DM_MULTICLASS" = true ] ; then
            # echo Submitting dm_multiclass jobs for repetition $i and $trainSize training jets
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=LorentzNet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=DeepSet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=ParticleTransformer comet.experiment=ParT_dm_$trainSize
            sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_dm_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_dm_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_dm_$trainSize
        fi

        if [ "$RUN_BINARY_CLS" = true ] ; then
            # echo Submitting binary_cls jobs for repetition $i and $trainSize training jets
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=LorentzNet comet.experiment=LorentzNet_bc_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=DeepSet comet.experiment=DeepSet_bc_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$TRAIN_SAMPS] test_samples=[$TEST_SAMPS] training_type=dm_multiclass model_type=OmniDeepSet comet.experiment=OmniDeepSet_bc_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=ParticleTransformer comet.experiment=ParT_bc_$trainSize
            sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=fine_tuning comet.experiment=OmniParT_ft_bc_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=from_scratch comet.experiment=OmniParT_fs_bc_$trainSize
            # sbatch $TRAINING_SCRIPT trainSize=$trainSize output_dir=$OUTDIR training_samples=[$CLS_TRAIN_SAMPS] test_samples=[$CLS_TEST_SAMPS] training_type=binary_classification model_type=OmniParT models.OmniParT.version=fixed_backbone comet.experiment=OmniParT_fb_bc_$trainSize
        fi

    done
done
