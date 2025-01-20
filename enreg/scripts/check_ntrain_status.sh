#!/bin/bash

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash enreg/scripts/$PROGNAME [-o]
  -o : This is used to specify the run directory.
EOF
  exit 1
}


OUTPUT_EXISTS=false
CHECK_JET_REGRESSION=false
CHECK_DM_MULTICLASS=false
CHECK_BINARY_CLS=false
TRAIN_FRACS=(2e3 1e4 1e5 1e6)
ALGORITHMS=(OmniParT_fine_tuning OmniParT_from_scratch OmniParT_fixed_backbone)
TASKS=(jet_regression dm_multiclass binary_classification)
while getopts 'o:jdbr:' OPTION; do
  case $OPTION in
    o)
        BASE_DIR=$OPTARG
        OUTPUT_EXISTS=true
        ;;
    j) CHECK_JET_REGRESSION=true ;;
    d) CHECK_DM_MULTICLASS=true ;;
    b) CHECK_BINARY_CLS=true ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"


NUMBER_TOTAL_REPETITIONS=`ls -d $BASE_DIR/* | wc -l`
for FRAC in ${TRAIN_FRACS[@]}; do
    echo "-------------"
    echo TrainFrac: $FRAC
    for TASK in ${TASKS[@]}; do
        echo "    Task:" $TASK
        for ALGO in ${ALGORITHMS[@]}; do
            echo "        Algorithm:" $ALGO
            COMPLETED=`ls $BASE_DIR/*/trainfrac_$FRAC/$TASK/$ALGO/model_best.pt | wc -l`
            echo "            Completed" $COMPLETED/$NUMBER_TOTAL_REPETITIONS
        done
    done
done
