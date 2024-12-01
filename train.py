"""train.py"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import json
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from waves.callbacks import TensorboardCallback
from waves.env import WavesEnv
from waves.utils import create_log_dir, parse_env_args


# parse CLI params
parser = argparse.ArgumentParser()
parser.add_argument(
    "sim", type=str, help='Simulation to use, eg. "SimStabilizeMidObs".'
)

# sim params (should have default values defined in models/ so should be able to be left blank)
parser.add_argument(
    "--sim_kwargs",
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

# env params
parser.add_argument(
    "--tmax", type=float, default=100.0, help="Duration (in time) of one episode."
)
parser.add_argument(
    "--action_min",
    type=str,
    default="-1.0",
    help="Minimum control value. Could use a simulation-specific "
    'constant, eg. "self.sim.k * 10".',
)
parser.add_argument(
    "--action_max",
    type=str,
    default="1.0",
    help="Maximum control value. Could use a simulation-specific "
    'constant, eg. "self.sim.k * 10".',
)
parser.add_argument(
    "--n_past_states",
    type=int,
    default=0,
    help="Number of previous states to add in the current state (memory).",
)
parser.add_argument(
    "--n_steps_per_action",
    type=int,
    default=1,
    help="Number of simulation steps for each environment step "
    "(ie. the same action is applied several times if this is set to a value large than 1).",
)

# training params
parser.add_argument(
    "--cpus", type=int, default=1, help="Number of CPUs to use for training."
)
parser.add_argument(
    "--steps", type=float, default=1e8, help="Number of timesteps to train for."
)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
parser.add_argument(
    "--n_steps",
    type=int,
    default=1024,
    help="The number of steps to run for each environment per update "
    "(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel).",
)
parser.add_argument("--batch_size", type=int, default=1024, help="Minibatch size.")
parser.add_argument(
    "--n_epochs", type=int, default=5, help="Number of SGD iterations per epoch."
)
parser.add_argument("--gamma", type=float, default=0.99, help="Gamma factor.")
parser.add_argument(
    "--network_depth",
    type=int,
    default=2,
    help="Number of hidden layers in the policy and value networks.",
)
parser.add_argument(
    "--hidden_layer_size",
    type=int,
    default=256,
    help="Number of cells per hidden layer in the policy and value networks.",
)

parser.add_argument(
    "--verbose",
    default=False,
    action="store_true",
    help="If set, prints training status periodically.",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=500000,
    help="Every so many steps, a checkpoint of the model will be saved.",
)

args = parser.parse_args()


# create config
env_kwargs = parse_env_args(args)

if __name__ == "__main__":
    # create experiment dir
    exp_dir = create_log_dir(subfolder="train")
    print(f"> {exp_dir}")

    # save config
    with open(exp_dir / "configs.json", "w", encoding="utf-8") as f:
        config_save = dict(args=vars(args), env_kwargs=copy.deepcopy(env_kwargs))
        config_save["env_kwargs"]["sim_class"] = str(
            config_save["env_kwargs"]["sim_class"]
        )
        json.dump(config_save, f, ensure_ascii=False, indent=4)

    # create env
    vec_env = make_vec_env(
        WavesEnv,
        n_envs=args.cpus,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if args.cpus > 1 else DummyVecEnv,
    )
    eval_env = WavesEnv(**env_kwargs)

    # create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=int(args.verbose),
        tensorboard_log=exp_dir / "tensorboard",
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        policy_kwargs={
            "net_arch": [
                {
                    "pi": [args.hidden_layer_size] * args.network_depth,
                    "vf": [args.hidden_layer_size] * args.network_depth,
                }
            ]
        },
    )

    # create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval // args.cpus,
        save_path=str(exp_dir / "checkpoints"),
        name_prefix="model",
        verbose=2 if args.verbose else 0,
    )
    tensorboard_callback = TensorboardCallback(eval_env)

    # train model
    model.learn(
        total_timesteps=int(args.steps),
        callback=[checkpoint_callback, tensorboard_callback],
    )

    # save model
    model_path = exp_dir / "model"
    model.save(model_path)
    print(f"> {model_path}")

