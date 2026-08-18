# Source this file from zsh before running ARX5 diffusion scripts:
#   source diffusion_env.zsh

source /home/star/mambaforge/etc/profile.d/conda.sh
conda activate /media/star/Elyos_PSSD/ARX5/robodiff-local

# Avoid leaking paths from the ARX5 SDK Python 3.10 environment into robodiff
# Python 3.9. Mixed site-packages can break binary modules such as NumPy.
unset PYTHONHOME
export PYTHONPATH=/media/star/Elyos_PSSD/ARX5/arx5_diffusion:/opt/ros/noetic/lib/python3/dist-packages

# Some local proxy variables on this machine break HuggingFace/diffusers imports.
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

cd /media/star/Elyos_PSSD/ARX5/arx5_diffusion

echo "ARX5 diffusion environment ready."
echo "Project: /media/star/Elyos_PSSD/ARX5/arx5_diffusion"
echo "Conda: /media/star/Elyos_PSSD/ARX5/robodiff-local"
echo "Examples:"
echo "  python train.py --help"
echo "  python collect_data.py --help"
