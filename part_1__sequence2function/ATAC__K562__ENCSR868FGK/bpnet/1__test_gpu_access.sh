#!/bin/bash

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python -c "import tensorflow as tf; print(f\"TF version: {tf.__version__}\"); print(f\"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}\"); print(f\"CUDA built: {tf.test.is_built_with_cuda()}\")"


