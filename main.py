import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


"""
Onde périodique qui se déplace à vitesse lambda=1:
yt + yx = 0
y(t,0) = y(t,1)

On ajoute un contrôle u(t) = f(t, y(t, 1/2))  
(ie on a un feedback par rapport à y en x=1/2 aulieu de la sortie x=1)
et y(t,0) devient y(t,1) + u(t)

^ on sait stabiliser (ie on veut ||y(t,.)||L2 -> 0) 
en reconstruisant la sortie avec un délai (ie on a de la mémoire)

par contre on ne sait plus rien faire lorsque l'équation devient
yt + (1+y²)yx = 0
ou yt + (1+2y)yx = 0
"""

# create experiment dir
now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
timestamp = datetime.now().timestamp()
exp_dir = Path(f'logs/{int(timestamp)}_{now}/')
exp_dir.mkdir(parents=True, exist_ok=False)
print(f'Created exp dir at {exp_dir}')

# hyperparameters
duration = 2
dt = 0.001
y0 = lambda x: (x*x *x * np.sin(x * np.pi * 2.0) ) * 0.499
# y0 = lambda x: np.sin(x * np.pi * 2.0) * 0.499
n_plots = 10
dx = 0.01
min_x = 0.0
max_x = 1.0
f = lambda x: x*x + x
k = -1.0

# discretize y
x_lst = np.arange(min_x, max_x + 1e-9, dx)
y_lst = [y0(x) for x in x_lst]

n_steps = int(duration / dt)
plot_interval = n_steps // (n_plots - 1)
for step in range(n_steps):
    y_lst[0] = y_lst[-1] + k * y_lst[len(y_lst) // 2]
    # compute yx (dy/dx)
    yx_lst = [-1]
    for i in range(1,len(x_lst)):
        yx = (f(y_lst[i]) - f(y_lst[i-1])) / dx
        yx_lst.append(yx)

    # dy/dt = -dy/dx  =>  y(t+dt) = y(t) - dt * dy/dx
    for i in range(1, len(y_lst)):
        y_lst[i] = y_lst[i] - dt * yx_lst[i]
    
    if step % plot_interval == 0:
        plt.figure()
        plt.plot(x_lst, y_lst)
        # plt.plot(x_lst, yx_lst)
        fig_path = exp_dir / f'{step+1}.png'
        plt.savefig(fig_path)
        print(f'Saved {fig_path}')
