#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=1000gb
#PBS -N rf

cd /rds/general/project/hda_24-25/live/TDS/hc724/Models/shap

module load anaconda3/personal
source activate test1

python shap_values_rf.py