# Install PyTorch
echo "Installing PyTorch..."
# Check if CUDA is available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA is available. Installing PyTorch with CUDA support."
    # Get the CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -i "cuda version" | awk '{print $9}')
    echo "CUDA version: $CUDA_VERSION"
    # Install PyTorch with CUDA support
    conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
else
    echo "CUDA is not available. Installing PyTorch without CUDA support."
    # Install PyTorch without CUDA support
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install SAM2
echo "Installing SAM2..."
pip install "git+https://github.com/facebookresearch/sam2.git"

# Install timm
echo "Installing timm..."
pip install timm

# Install imageio
echo "Installing imageio..."
pip install imageio 'av<14'