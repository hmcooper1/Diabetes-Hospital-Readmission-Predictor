#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=1000gb
#PBS -N xgb

cd /rds/general/user/hc724/projects/hda_24-25/live/ML/Group14/Models

module load anaconda3/personal
source activate test1

python xgb_cv.py
