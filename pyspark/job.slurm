#!/bin/bash
#SBTCH -A sxg125_csds312 -p classc
#SBATCH --job-name=spark-pi      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module load gcc/6.3.0
module load python/3.8.6
module load spark/3.2.0

mkdir $PFSDIR/course-project
WORKDIR=$PFSDIR/course-project

cp -R ./Data $WORKDIR/Data
cp p3env $WORKDIR/
cp requirements.txt $WORKDIR/
cp *.py $WORKDIR/

cd $WORKDIR

source p3venv/bin/activate
pip install -r requirements.txt

spark-submit --executor-cores 4 --executor-memory 5G spark_standalone.py 
cp -ru * $SLURM_SUBMIT_DIR