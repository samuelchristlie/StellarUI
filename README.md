StellarUI
=======
A powerful and modular GUI with a graph/nodes interface for generative models 
-----------

This UI allows advanced design and execution pipelines using node-based interface.

# Installation

## Clone Repository

```git clone https://github.com/samuelchristlie/StellarUI.git```

Insert your checkpoints and VAE into the folder

## Install Dependencies

```pip install -r requirements.txt```


## AMD
AMD GPU requires `rocm` and `pytorch`

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2"```

## NVIDIA

Nvidia users requires `xformers`


# Running

```python run.py```
