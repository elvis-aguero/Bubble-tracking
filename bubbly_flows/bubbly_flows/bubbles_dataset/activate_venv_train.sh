module load python/3.11
module load cudnn/9
module load cuda/12
source ~/xanylabeling_data/ultralytics-bubble/bin/activate

echo "Environment activated. "
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
