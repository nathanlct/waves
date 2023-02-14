from datetime import datetime
from pathlib import Path

from models import *


def create_log_dir(subfolder):
    now = datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
    timestamp = datetime.now().timestamp()
    log_dir = Path(f"logs/{subfolder}/{int(timestamp)}_{now}/")
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir

def parse_sim_args(args):
    # retrieve sim
    available_sims = dict(filter(lambda kv: kv[0].startswith('Sim'), globals().items()))
    try:
        sim_class = available_sims[args.sim]
    except:
        raise ValueError(f'Simulation "{args.sim}" does not exist. Available simulations are: {", ".join(available_sims.keys())}.')

    # create sim
    try:
        sim_params = {}
        if args.y0 is not None: sim_params['y0'] = args.y0 
        if args.dt is not None: sim_params['dt'] = args.dt 
        if args.dx is not None: sim_params['dx'] = args.dx 
        if args.xmin is not None: sim_params['xmin'] = args.xmin 
        if args.xmax is not None: sim_params['xmax'] = args.xmax 
        sim_kwargs = {
            **sim_params,
            **eval(args.kwargs)
        }
    except TypeError as err:
        raise TypeError(f'{args.sim}.{err}. Consider adding or removing some arguments using --kwargs.')

    # return sim
    return sim_class, sim_kwargs