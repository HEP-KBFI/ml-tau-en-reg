#!/bin/bash

 ZH_BASE_NTUPLE_DIR="/local/laurits/ml-tau-en-reg/ntuples/20240924_lowered_recoPtCut/p8_ee_ZH_Htautau_ecm380/"
 ZH_OUTPUT_DIR="/home/laurits/HPS_recoCut0_ntuples/ZH"
 ZH_FILES=$(ls $ZH_BASE_NTUPLE_DIR)
 xargs -n 1 <<< "$ZH_FILES" | tr ' ' ',' | xargs -I {} sbatch enreg/scripts/HPS/submission.sh output_dir=$ZH_OUTPUT_DIR files=[{}] base_ntuple_dir=$ZH_BASE_NTUPLE_DIR

#  ##################################
  Z_BASE_NTUPLE_DIR="/local/laurits/ml-tau-en-reg/ntuples/20240924_lowered_recoPtCut/p8_ee_Z_Ztautau_ecm380/"
  Z_OUTPUT_DIR="/home/laurits/HPS_recoCut0_ntuples/Z"
  Z_FILES=$(ls $Z_BASE_NTUPLE_DIR)
  xargs -n 1 <<< "$Z_FILES" | tr ' ' ',' | xargs -I {} sbatch enreg/scripts/HPS/submission.sh output_dir=$Z_OUTPUT_DIR files=[{}] base_ntuple_dir=$Z_BASE_NTUPLE_DIR

 ##################################
 QQ_BASE_NTUPLE_DIR="/local/laurits/ml-tau-en-reg/ntuples/20240924_lowered_recoPtCut/p8_ee_qq_ecm380/"
 QQ_OUTPUT_DIR="/home/laurits/HPS_recoCut0_ntuples/QQ"
 QQ_FILES=$(ls $QQ_BASE_NTUPLE_DIR)
 xargs -n 1 <<< "$QQ_FILES" | tr ' ' ',' | xargs -I {} sbatch enreg/scripts/HPS/submission.sh output_dir=$QQ_OUTPUT_DIR files=[{}] base_ntuple_dir=$QQ_BASE_NTUPLE_DIR
