# Source this file from zsh before running Python examples:
#   source arx_env.zsh

source /home/star/mambaforge/etc/profile.d/conda.sh
conda activate /media/star/Elyos_PSSD/ARX5/arx-py310-local

export PYTHONPATH=/media/star/Elyos_PSSD/ARX5/arx5-sdk-main/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/media/star/Elyos_PSSD/ARX5/arx5-sdk-main/lib/x86_64:${LD_LIBRARY_PATH}

cd /media/star/Elyos_PSSD/ARX5/arx5-sdk-main/python

echo "ARX5 environment ready."
echo "Example: python examples/test_joint_control.py X5 can1"
