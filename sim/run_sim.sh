#!/bin/bash
#SBATCH -p main
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
set -x

env
df -h

NEV=100
NUM=$1 #random seed
SAMPLE=$2 #main card

#Change these as needed
OUTDIR=/local/joosep/clic_edm4hep/2024_03/
SIMDIR=`pwd`/sim
WORKDIR=/scratch/local/$USER/${SAMPLE}_${SLURM_JOB_ID}
FULLOUTDIR=${OUTDIR}/${SAMPLE}

mkdir -p $FULLOUTDIR

mkdir -p $WORKDIR
cd $WORKDIR

cp $SIMDIR/${SAMPLE}.cmd card.cmd
cp $SIMDIR/pythia.py ./
cp $SIMDIR/clic_steer.py ./
cp -R $SIMDIR/PandoraSettings ./
cp -R $SIMDIR/clicRec_e4h_input.py ./

echo "Random:seed=${NUM}" >> card.cmd
cat card.cmd

#Use a tagged version of Key4HEP
source /cvmfs/sw.hsf.org/spackages7/key4hep-stack/2023-04-08/x86_64-centos7-gcc11.2.0-opt/urwcv/setup.sh

#Run generation
k4run $SIMDIR/pythia.py -n $NEV --Dumper.Filename out.hepmc --Pythia8.PythiaInterface.pythiacard card.cmd

#Run simulation
ddsim --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml \
      --outputFile out_sim_edm4hep.root \
      --steeringFile clic_steer.py \
      --inputFiles out.hepmc \
      --numberOfEvents $NEV \
      --random.seed $NUM

#Run reconstruction
k4run clicRec_e4h_input.py -n $NEV --EventDataSvc.input out_sim_edm4hep.root --PodioOutput.filename out_reco_edm4hep.root

#Copy the outputs
cp out_reco_edm4hep.root reco_${SAMPLE}_${NUM}.root
cp out.hepmc sim_${SAMPLE}_${NUM}.hepmc
cp reco_${SAMPLE}_${NUM}.root $FULLOUTDIR/
cp sim_${SAMPLE}_${NUM}.hepmc $FULLOUTDIR/

rm -Rf $WORKDIR
