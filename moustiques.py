import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models.sim_ode_discrete import SimODEDiscrete


sim = SimODEDiscrete(
    K=50578.0,
    y0=lambda x: np.random.uniform(low=0.0, high=250000, size=4),
    dt=1e-2,
)
sim.reset()

def get_action(MMS, F, noise=False):
   action = 0
   if MMS < 200:
       if F == 0 or np.log(F) < np.log(200) - 4:
           action = 0
       elif np.log(F) < np.log(200) - 3:
           action = 500000 * (4 + np.log(F / 200))
       else:
           action = 500000
   else:
       if F == 0 or np.log(MMS) > np.log(F) + 4:
           action = 0
       elif np.log(MMS) > np.log(F) + 3:
           action = 500000 * (4 - np.log(MMS / F))
       else:
           action = 500000
   if noise:
       action += np.random.normal(loc=0.0, scale=10.0)
   return action

def get_action(total_males, total_females):
    if total_males < 200:
        if total_females == 0 or np.log(total_females) < np.log(200) - 4:
            action = 0
        else:
            action = 200000
    else:
        if total_females == 0 or np.log(total_males) > np.log(total_females) + 4:
            action = 0
        else:
            action = 200000
    return action

def get_action(total_males, total_females):
    if total_females == 0 or (total_males != 0 and np.log(total_males / total_females) > 4):
        action = 0
    else:
        action = 200000  # limite 165435
    return action

u_lst = []
while sim.t <= 3000:
    total_males = sim.y[1] + sim.y[3]
    total_females = sim.y[2] * (1 + sim.gammas * sim.y[3] / sim.y[1])
    action = get_action(total_males, total_females)
    u_lst.append(action)
    sim.step(u=[action])

y_lst = np.array(sim.y_lst)
t_lst = np.array(sim.t_lst)
u_lst = np.array([0] + u_lst)

fig = plt.figure(layout='constrained', figsize=(15, 8))

gs = GridSpec(5, 7, figure=fig)

for k in range(5):
    ax = fig.add_subplot(gs[k, :3])
    ax.plot(t_lst, y_lst[:, k] if k < 4 else u_lst)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(['E(t)', 'M(t)', 'F(t)', 'Ms(t)', 'u(t)'][k])
    if k == 4:
        ax.set_title(f'Integral of u: {int(np.sum(u_lst) * sim.dt)}')

for k in range(2):
    ax = fig.add_subplot(gs[:3, 3+2*k:5+2*k])

    total_males = y_lst[:, 1] + y_lst[:, 3]
    total_females = y_lst[:, 2] * (1 + sim.gammas * y_lst[:, 3] / y_lst[:, 1])

    ax.plot(total_males, total_females, 'r')
    ax.scatter(total_males[0], total_females[0], s=30,
               marker='o', color='r', zorder=100)
    ax.scatter(total_males[-1], total_females[-1], s=50,
               marker='x', color='r', zorder=100)

    x_total_males = [np.linspace, np.geomspace][k] \
        (*np.maximum(1e-5, ax.get_xlim()), 500)
    y_total_females = [np.linspace, np.geomspace][k] \
        (*np.maximum(1e-5, ax.get_ylim()), 500)
    X, Y = np.meshgrid(x_total_males, y_total_females)
    Z = np.zeros((len(x_total_males), len(y_total_females)))
    for i in range(len(x_total_males)):
        for j in range(len(y_total_females)):
            Z[j, i] = get_action(x_total_males[i], y_total_females[j])
    pcm = ax.pcolormesh(X, Y, Z)

    ax.set_xlabel('Total males' + (' (log scale)' if k == 1 else ''))
    ax.set_ylabel('Total females' + (' (log scale)' if k == 1 else ''))

    if k == 1:
        ax.set_xscale('log')
        ax.set_yscale('log')

for k in range(2):
    ax = fig.add_subplot(gs[3:, 3+2*k:5+2*k])
    ax.plot(y_lst[0], label=f'y0 ({"10^" if k == 1 else ""}{y_lst[0].astype(int) if k == 0 else np.round(np.log10(y_lst[0]), 1)})')
    ax.plot(y_lst[-1], label=f'yf ({"10^" if k == 1 else ""}{y_lst[-1].astype(int) if k == 0 else np.round(np.log10(y_lst[-1]), 1)})')
    plt.xticks(ticks=[0, 1, 2, 3], labels=['E', 'M', 'F', 'Ms'])
    if k == 1:
        ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode="expand", borderaxespad=0.)

fig.suptitle('Moustiques')
plt.savefig('moustiques.png')
