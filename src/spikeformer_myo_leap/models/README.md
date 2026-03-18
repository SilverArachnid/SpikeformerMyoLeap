# Models Package

This folder contains one model family per file for EMG-to-pose regression.

## Files

- `__init__.py`: Re-exports the public model interfaces.
- `registry.py`: Maps model names to model classes for training and evaluation entry points.
- `spikeformer.py`: Spikeformer-based regressor with explicit multi-head spike attention.
- `transformer.py`: Standard Transformer encoder regressor baseline.
- `cnn_lstm.py`: CNN-LSTM regressor baseline.
- `cnn.py`: Temporal CNN regressor baseline.
- `spiking_cnn.py`: Spiking CNN regressor baseline.
