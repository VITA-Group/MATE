# MPI cluster submit
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_GPUs = 2' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

# MPI cluster submit miniImageNet
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 16000' \
  -append 'request_GPUs = 4' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 30000'

# Testing
condor_submit_bid 250 -interactive -append 'request_cpus = 8' -append 'request_memory = 12000' \
  -append 'request_GPUs = 1' -append 'requirements = TARGET.CUDAGlobalMemoryMb > 15000'

# Train - FiLM with Normalization
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-6-detach-normalize \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-6

# Test for accuracy
python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth

# Test model and save task codes
python test_film_task.py --gpu 0 \
  --episode 100 --way 5 --shot 5 --query 15 --dataset CIFAR_FS \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --film-normalize --dual-BN \
  --save-file saved-task-emb/msgan-1e-8_ortho-1e-10.npy \
  --load ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-8_ortho_1e-10/epoch_21.pth
