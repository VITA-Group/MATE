python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNetRFS

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

python train_film.py --gpu 0,1 \
--save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01 \
--save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
--film-reg-type MSGAN --film-reg-level 1e-6 \
--load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth

load_path="./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_msgan1e-6_lr0.01/epoch_22.pth"
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 --dataset CIFAR_FS \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
--head SVM --network ResNetRFS_FiLM \
--task-embedding FiLM_SVM_WGrad --dual-BN \
--load $load_path;
