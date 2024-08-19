@echo off

title Uploading script
echo Uploading to the cluster

REM Variables
set user=dvezzaro
set source_dir=C:/Users/david/Desktop/Diffusers/boxdiff-test/*
set destination_dir=/home/dvezzaro/hf_boxdiff

echo - local source folder:%source_dir%
echo - remote destination folder:%destination_dir%

echo Uploading ...
scp -r -J %user%@labta.math.unipd.it %source_dir% %user%@labsrv8.math.unipd.it:%destination_dir%
echo Done

echo Connecting to ssh...
REM ssh -tt -J %user%@labta.math.unipd.it %user%@labsrv8.math.unipd.it "srun --job-name=hf_boxdiff --chdir=/home/dvezzaro/hf_boxdiff --partition=allgroups --time=0-01:00:00 --mem=12G --cpus-per-task=6 --gres=gpu:1 --pty bash -c 'source /conf/shared-software/anaconda/etc/profile.d/conda.sh && conda activate hf_boxdiff && python /home/dvezzaro/hf_boxdiff/run_gligen_boxdiff.py --prompt \"A rabbit wearing sunglasses looks very proud\" --gligen_phrases [\"a rabbit\",\"sunglasses\"] --P 0.2 --L 1 --seeds [1,2,3,4,5,6,7,8,9] --token_indices [2,4] --bbox [[67,87,366,512],[66,130,364,262]] --refine False && conda deactivate'"
ssh -tt -J %user%@labta.math.unipd.it %user%@labsrv8.math.unipd.it "srun --job-name=hf_boxdiff --chdir=/home/dvezzaro/hf_boxdiff --partition=interactive --time=0-01:00:00 --mem=12G --cpus-per-task=1 --gres=gpu:1 --pty bash -c 'source /conf/shared-software/anaconda/etc/profile.d/conda.sh && conda activate hf_boxdiff && python /home/dvezzaro/hf_boxdiff/run_gligen_boxdiff.py && conda deactivate'"