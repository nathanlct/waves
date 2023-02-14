# Waves

Applying reinforcement learning to the stabilization of ODE/PDE systems.

## Installation

```
conda env create -f environment.yml
conda activate waves
```

## Usage

Create new models in `models`.

## train.py

```bash
# specifying an initial condition
python train.py SimStabilizeMidObs
```

## simulate.py

```bash
# specifying an initial condition
python simulate.py SimStabilizeMidObs --kwargs "dict(f=lambda x: x*x+x)" --y0 "lambda x: np.cos(8 * x * np.pi) * 0.01" --dt 3e-4 --dx 1e-3 --xmin 0 --xmax 1 --tmax 2 --no_save

# default initial condition generator, high render
python simulate.py SimStabilizeMidObs --kwargs "dict(f=lambda x: x*x+x)" --dt 3e-5 --dx 1e-3 --xmin 0 --xmax 1 --tmax 2 --render_speed 0.5 --render_fps 60

# example using SimControlHeat
python simulate.py SimControlHeat --y0 "lambda x: 2 * np.sin(np.pi * x)" --dt 0.45e-4 --dx 1e-2 --xmin 0 --xmax 1 --tmax 1.0 --render_speed 0.5
```
