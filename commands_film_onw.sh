# MPI cluster submit
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_gpus = 2' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

# MPI cluster submit miniImageNet
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 16000' \
  -append 'request_gpus = 4' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 30000'

# Testing
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_gpus = 1' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

# ==============================================================================
# Training
# ==============================================================================


# python train.py --gpu 0,1 --save-path "./experiments/CIFAR_FS_ResNet12_baseline" --train-shot 5 \
# --head SVM --network ResNet12 --dataset CIFAR_FS

python train.py --gpu 0,1 --save-path "./experiments/CIFAR_FS_ResNet12_ResNet12_baseline" --train-shot 5 \
--head SVM --network ResNet --dataset CIFAR_FS
python test.py --gpu 0  --episode 1000 --way 5 --shot 1 --query 15 \
  --head SVM --network ResNet --dataset CIFAR_FS \
  --load ./experiments/CIFAR_FS_ResNet12_ResNet12_baseline/epoch_21.pth;python test.py --gpu 0  --episode 1000 --way 5 --shot 5 --query 15 \
  --head SVM --network ResNet --dataset CIFAR_FS \
  --load ./experiments/CIFAR_FS_ResNet12_ResNet12_baseline/epoch_21.pth;


# Joint train from scratch
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN


load_model=./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN/epoch_25.pth; python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model;python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model


# Load pretrained; train jointly with MSGAN reg 1e-6
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth


load_model=./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_msgan1e-6_lr0.01/epoch_23.pth; python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model;python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model


# Load pretrained; train film and dual-BN only
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_train-film-dualBN_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN --train-film-dualBN --lr 0.01 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth


load_model=./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_train-film-dualBN_lr0.01/epoch_25.pth; python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model;python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model


# Load pretrained; train film and dual-BN only with MSGAN reg 1e-6
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_train-film-dualBN_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN --train-film-dualBN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth


load_model=./experiments/CIFAR_FS_ResNet12_FiLM-SVM-OnW_dual-BN_load-naive_train-film-dualBN_msgan1e-6_lr0.01/epoch_25.pth; python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model;python test_film.py --gpu 0 \
  --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load $load_model


















python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth




# Train - MATE
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN

# Train - FiLM with Normalization
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_msgan-reg-1e-6-detach-normalize \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --film-normalize --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-6

# Train - MATE, load from pre-trained naive backbone
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_load-naive \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN \
  --load-naive-backbone --load pathtopretrained

# Train - MATE, load from pre-trained naive backbone
#   Only train FiLM and BN with task code
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_load-naive_train-film-dualBN \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --dual-BN --train-film-dualBN \
  --load-naive-backbone --load pathtopretrained

# ==============================================================================
# Testing
# ==============================================================================

# Test for accuracy - Baseline
python test.py --gpu 0  --episode 1000 --way 5 --shot 5 --query 15 \
  --head SVM --network ResNet --dataset CIFAR_FS \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

# Test for accuracy - FiLM
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --film-normalize --dual-BN \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth

# Test model and save task codes
python test_film_task.py --gpu 0 \
  --episode 100 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_OnW --film-normalize --dual-BN \
  --save-file saved-task-emb/msgan-1e-8_ortho-1e-10.npy \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-OnW_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth
