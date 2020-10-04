python test.py --gpu 0,1 \
--load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNet
python test.py --gpu 0,1 \
--load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth \
--way 5 --shot 5 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNet

python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNetRFS
python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 5 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNetRFS

python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head LR --network ResNetRFS
python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 5 --query 15 --dataset CIFAR_FS --episode 1000 \
--head LR --network ResNetRFS

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.1_lambda10 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.1 --lambda-epochs 10,40,50 --num-epoch 20 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-4_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-4 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

## LOAD-FIX

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.1_lambda10 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.1 --lambda-epochs 10,40,50 --num-epoch 20 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.1_lambda10 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.1 --lambda-epochs 10,40,50 --num-epoch 20 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-4_lr0.1_lambda10 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.1 --lambda-epochs 10,40,50 --num-epoch 20 \
--film-reg-type MSGAN --film-reg-level 1e-4 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-5_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-5 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-4_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-4 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth


###### TESTING
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01/epoch_22.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01/epoch_22.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-4_lr0.01/epoch_24.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01/best_model.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-4_lr0.01/epoch_24.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.1_lambda10/epoch_13.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.1_lambda10/best_model.pth"
load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01/epoch_22.pth"
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path; python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;

python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
--head LR --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path; python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
--head LR --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;

condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 32000'   -append 'request_gpus = 4' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 30000'

#######
# MiniImageNet
#######

# TESTING
load_path="./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth"
python test.py --gpu 0,1 --load $load_path --episode 1000 \
--way 5 --shot 1 --query 15 --head SVM --network ResNetRFS --dataset miniImageNet
python test.py --gpu 0,1 --load $load_path --episode 1000 \
--way 5 --shot 5 --query 15 --head SVM --network ResNetRFS --dataset miniImageNet

python test.py --gpu 0,1 --load $load_path --episode 1000 \
--way 5 --shot 1 --query 15 --head LR --network ResNetRFS --dataset miniImageNet
python test.py --gpu 0,1 --load $load_path --episode 1000 \
--way 5 --shot 5 --query 15 --head LR --network ResNetRFS --dataset miniImageNet

# test_film
load_path="./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth"
load_path="./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01/epoch_22.pth"
load_path="./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01/epoch_22.pth"
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset miniImageNet \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset miniImageNet \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;

# ResNet12 load-train + lr 0.1
python train_film.py --gpu 0,1,2,3 \
--save-path ./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.1 \
--save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
--episodes-per-batch 8 --val-episodes-per-batch 1 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.1 \
--load-naive-backbone --load ./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth

# ResNet12 load-train + lr 0.1 + msgan 1e-6
python train_film.py --gpu 0,1,2,3 \
--save-path ./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.1 \
--save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
--episodes-per-batch 8 --val-episodes-per-batch 1 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.1 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth

# ResNet12 load-train + lr 0.01
python train_film.py --gpu 0,1,2,3 \
--save-path ./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
--save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
--episodes-per-batch 8 --val-episodes-per-batch 1 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
--load-naive-backbone --load ./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth



# ResNet12 load-fix
python train_film.py --gpu 0,1,2,3 \
--save-path ./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
--save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
--episodes-per-batch 8 --val-episodes-per-batch 1 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--load-naive-backbone --load ./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth

# ResNet12 load-fix + msgan 1e-6
python train_film.py --gpu 0,1,2,3 \
--save-path ./experiments/miniImageNet_RFS_SVM_ResNet12-FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
--save-epoch 10 --train-shot 15 --dataset miniImageNet --eps 0.1 \
--episodes-per-batch 8 --val-episodes-per-batch 1 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/miniImageNet_RFS_SVM_ResNet12_baseline/resnet12_last.pth
