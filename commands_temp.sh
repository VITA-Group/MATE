python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth

python test_film_task.py --gpu 0 \
  --episode 100 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --save-file saved-task-emb/film-normalize_msgan-1e-6.npy \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth

python train_film.py --gpu 2,3 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_no-final-relu \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --no-final-relu

condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_gpus = 2' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_gpus = 1' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 16000' \
  -append 'request_gpus = 4' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 30000'

python test_film.py --gpu 1 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load 

python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --no-final-relu \
  --load 

# LOAD and TRAIN
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.1 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,4 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_norm-film_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 --film-normalize \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.02 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.02 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.05 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.05 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.005 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.005 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.005  \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.005 \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.005/last_epoch.pth

# LOAD AND FIX

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.1 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.05 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.05 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.02 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.02 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 2,3 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_msgan1e-6_norm-film_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 --film-normalize \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.005 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.005 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

python test_film_task.py --gpu 0 \
  --episode 100 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --save-dir saved-task-emb --save-file load-naive_train-film-dualBN_lr0.01_epoch21.npy \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.01/epoch_21.pth

python test_film_film_output.py --gpu 0 \
  --episode 100 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --save-dir saved-film-outputs/load-naive_train-film-dualBN_lr0.01_epoch21 \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_train-film-dualBN_lr0.01/epoch_21.pth


############################
# miniImageNet experiments #
############################

# ResNet12 baseline
python train.py --gpu 0,1,2,3 --save-path ./experiments/miniImageNet_MetaOptNet_SVM_baseline --train-shot 15 \
  --head SVM --network ResNet --dataset miniImageNet --eps 0.1

# ResNet18 baseline
python train.py --gpu 0,1,2,3 --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet18_baseline --train-shot 15 \
  --head SVM --network ResNet18 --dataset miniImageNet --eps 0.1

# TESTING
python test.py --gpu 0,1,2,3 --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth --episode 1000 \
  --way 5 --shot 1 --query 15 --head SVM --network ResNet --dataset miniImageNet


python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN/epoch_21.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN/epoch_21.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN/epoch_22.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN/epoch_22.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_fixpre/epoch_21.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_fixpre/epoch_21.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_fixpre/epoch_22.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_fixpre/epoch_22.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.02_lambda10/epoch_11.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.02_lambda10/epoch_11.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.02_lambda10/epoch_12.pth;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.02_lambda10/epoch_12.pth

python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth

# ResNet12 MATE - old + fixpre
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_fixpre \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --fix-preprocess

# ResNet12 MATE - old
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN

# ResNet12 load-train
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.02_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.02 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.02_lambda10_fixpre \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --lr 0.02 --lambda-epochs 10,40,50 --num-epoch 20 --fix-preprocess \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.05_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.05 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.05_lambda10_fixpre \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --lr 0.05 --lambda-epochs 10,40,50 --num-epoch 20 --fix-preprocess \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
  --save-epoch 1 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 --second-lr \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01/best_model.pth

# ResNet12 load-train + msgan
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 1 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 --second-lr \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01/best_model.pth

# ResNet12 load-fix
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.02_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.02 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.05_lambda10 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.05 --lambda-epochs 10,40,50 --num-epoch 20 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01/last_epoch.pth

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
  --save-epoch 1 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 --second-lr \
  --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01/best_model.pth

# ResNet12 load-fix + msgan
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
  --episodes-per-batch 8 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/miniImageNet_MetaOptNet_SVM_ResNet12_baseline/epoch_21.pth

