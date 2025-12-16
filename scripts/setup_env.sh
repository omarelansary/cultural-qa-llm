#!/bin/bash
# Load HPC modules (check your cluster docs, usually something like this)
# module load Anaconda3
# module load cuda/12.1
module load GCCcore/13.2.0
module load bzip2/1.0.8

# Create env only if it doesn't exist
conda create -n llama_project python=3.10 -y
source activate llama_project

echo "=== torch CUDA check ==="
python - << 'EOF'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
EOF

# Install dependencies
pip install -r requirements.txt