"""simulate.py"""
from waves.utils import create_log_dir, parse_sim_args

import argparse
import numpy as np


# parse CLI params
parser = argparse.ArgumentParser()
parser.add_argument(
    "sim", type=str, help='Simulation to use, eg. "SimStabilizeMidObs".'
)
parser.add_argument(
    "--kwargs",
    type=str,
    default="dict()",
    help='Additional simulation kwargs, eg. "dict(f=lambda x: x*x)".',
)
parser.add_argument(
    "--y0",
    type=str,
    default=None,
    help="Initial condition as a (ideally vectorized) function of space, "
    'eg. "lambda x: np.cos(x*np.pi)". By default, uses the simulation\'s default, if there is one.',
)
parser.add_argument(
    "--u",
    type=str,
    default=None,
    help="Control input (action) as a function of time."
    'eg. "lambda t: np.exp(-0.2*t)". By default, uses a constant input of 0.',
)
parser.add_argument(
    "--dx",
    type=float,
    default=None,
    help="Space sampling interval. By default, use the simulation's default.",
)
parser.add_argument(
    "--xmin",
    type=float,
    default=None,
    help="Space left boundary. By default, use the simulation's default.",
)
parser.add_argument(
    "--xmax",
    type=float,
    default=None,
    help="Space right boundary. By default, use the simulation's default.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    help="Time sampling interval. By default, use the simulation's default.",
)
parser.add_argument(
    "--tmax", type=float, default=200.0, help="Duration of the simulation."
)
parser.add_argument(
    "--no_display",
    action="store_true",
    help="Set this to not display the rendered video.",
)
parser.add_argument(
    "--no_save", action="store_true", help="Set this to not save the rendered video."
)
parser.add_argument(
    "--render_fps",
    type=float,
    default=60.0,
    help="Frames per second of the rendered video "
    "(large values will take a longer time to generate and may be incorrect in the displayed video "
    "but will be correct in the saved one). Default is 60 FPS.",
)
parser.add_argument(
    "--render_dpi",
    type=float,
    default=100.0,
    help="Resolution (dots per inch) of the rendered video "
    "(large values will take a longer time to generate and may be incorrect in the displayed video "
    "but will be correct in the saved one). Default is 100 DPI.",
)
parser.add_argument(
    "--render_speed",
    type=float,
    default=1.0,
    help="Speed of the rendered video, "
    "eg. 1 is real time, 0.5 is twice slower, 2.0 is twice faster "
    "(low values will take a longer time to generate and may be incorrect in the displayed video "
    "but will be correct in the saved one). Default is 1.0.",
)
args = parser.parse_args()

# create sim
sim_class, sim_kwargs = parse_sim_args(args)
sim = sim_class(**sim_kwargs)

# run sim
control = eval(args.u) if args.u is not None else lambda _: np.zeros(sim.n_controls)
while sim.t < args.tmax:
    sim.step(control(sim.t))

# create experiment dir
if not args.no_save:
    logdir = create_log_dir(subfolder="simulate")
    print(f"> {logdir}")

# render
if not args.no_save:
    render_path = logdir / "render.mp4"
sim.render(
    path=render_path if not args.no_save else None,
    display=not args.no_display,
    fps=args.render_fps,
    dpi=args.render_dpi,
    speed=args.render_speed,
)
if not args.no_save:
    print(f"> {render_path}")
