import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
from collections import defaultdict


class Simulation:
    """
    Simulating

        y_t + f(y)_x = 0
        y(t,0) = y(t,1) + u(t, y(t,1/2))
    
    for some control u
    """
    def __init__(self, f, dt, dx, xmin, xmax):
        self.dt = dt
        self.dx = dx

        self.xmin = xmin
        self.xmax = xmax
        
        self.f = f
        self.y0 = None

    def reset(self, y0=None):
        if self.y0 is None and y0 is not None:
            self.y0 = y0

        self.t = 0
        self.x = np.arange(self.xmin, self.xmax + 1e-9, self.dx)
        self.y = np.array(list(map(self.y0, self.x)))

        self.data = defaultdict(list)
        self.save_data()

    def save_data(self):
        self.data['t'].append(self.t)
        self.data['x'].append(np.copy(self.x))
        self.data['y'].append(np.copy(self.y))

    def step(self, u=0):
        self.t += self.dt

        # initial condition with control u
        self.y[0] = self.y[-1] + u

        # compute df(y)/dx
        self.yx = np.diff(self.y) / self.dx

        # dy/dt + df(y)/dx = 0  =>  y(t+dt) = y(t) - dt * df(y)/dx
        self.y[1:] -= self.dt * self.yx
        
        self.save_data()

    def get_obs(self):
        return np.array([self.t, self.y[len(self.y) // 2]])

    def norm_y(self):
        return (self.dx * np.sum(self.y * self.y)) ** 0.5

    def render(self, path=None):
        """
        Renders the evolution of y since t=0 (ie since the last reset).
        Shows it using plt.show, unless path is specified in which case
        it is saved at that path and not showed. Path must end with ".gif".
        """
        fig = plt.figure()

        y_line = plt.plot(self.data['x'][0], self.data['y'][0])[0]
        time_label = plt.text(0.8, 0.9, '', fontsize=10, transform=fig.axes[0].transAxes)

        def animate(i):
            y_line.set_data(self.data['x'][i], self.data['y'][i])
            time_label.set_text(f't = {self.data["t"][i]:.3f} s')
            return [y_line, time_label]

        anim_fps = 30
        anim = animation.FuncAnimation(
            fig=fig,
            func=animate, 
            frames=range(0, len(self.data['t']), int(1 / (self.dt * anim_fps))),
            interval=1000/anim_fps,
            repeat=True,
            repeat_delay=500,
            blit=True, 
        )

        if path is None:
            plt.show()
        else:
            anim.save(path)


if __name__ == '__main__':
    sim = Simulation(
        f=lambda x: x*x + x,
        dt=1e-3,
        dx=1e-2,
        xmin=0,
        xmax=1,
    )

    sim.reset(
        y0=lambda x: np.sin(x * np.pi * 2.0) * 0.499
    )

    while sim.t < 5.0:
        sim.step(-1)

    sim.render(path='test.gif')
