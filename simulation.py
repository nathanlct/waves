import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
from collections import defaultdict
from abc import ABC
from celluloid import Camera


class Simulation(ABC):
    def __init__(self, dt, dx, xmin, xmax, y0=None):
        self.dt = dt
        self.dx = dx

        self.xmin = xmin
        self.xmax = xmax

        self.reset(y0 if y0 is not None else lambda _: 0)

    def reset(self, y0=None):
        if y0 is not None:
            self.y0 = y0

        self.n_steps = 0
        self.t = 0
        self.x = np.arange(self.xmin, self.xmax + 1e-9, self.dx)
        self.y = np.array(list(map(self.y0, self.x)))

        self.data = defaultdict(list)
        self.save_data()

    def save_data(self):
        self.data['t'].append(self.t)
        self.data['x'].append(np.copy(self.x))
        self.data['y'].append(np.copy(self.y))

    def update_y(self):
        raise NotImplementedError

    def step(self, u=0):
        self.t += self.dt
        self.n_steps += 1
        self.update_y(u)
        self.save_data()

    def get_obs(self):
        raise NotImplementedError

    def norm_y(self):
        return (self.dx * np.sum(self.y * self.y)) ** 0.5

    def render(self, path='sim.mp4', fps=30.0, dpi=300, accel=1.0):
        """
        Renders the evolution of y since t=0 (ie since the last reset).
        """
        fig = plt.figure(dpi=dpi)
        plt.xlim(self.xmin, self.xmax)
        max_abs_y0 = np.max(np.abs(self.data['y'][0]))
        plt.ylim(-max_abs_y0, max_abs_y0)
        plt.grid(color='green', linewidth=0.5, linestyle='--')
        camera = Camera(fig)

        if fps == np.inf:
            fps = 1 / self.dt * 2
        snap_interval = 1.0 / fps
        last_time_snapped = - 2 * snap_interval
        for x_data, y_data, t_data in zip(self.data['x'], self.data['y'], self.data['t']):
            if t_data >= last_time_snapped + snap_interval:
                last_time_snapped = t_data
                plt.plot(x_data, y_data, color='black')
                plt.text(0.8, 0.9, f't = {t_data:.3f} s', fontsize=10, transform=fig.axes[0].transAxes)
                camera.snap()

        animation = camera.animate(interval=snap_interval * 1000 / accel)
        animation.save(path)
        print('>', path)


class SimControlHeat(Simulation):
    """
    Simulating
        y_t - y_xx = y^3   
    Boundary conditions
        y(t,0) = u1
        y(t,1) = u2
        for some control u = (u1, u2)
        that can observe y(0,x) and t
    """
    def update_y(self, u=(0, 0)):
        # boundary conditions
        u1, u2 = u
        self.y[0] = u1
        self.y[-1] = u2

        # compute y_xx = (y(t,n+1)-2y(t,n)+y(t,n-1))/dx^2
        yxx = np.diff(self.y, n=2) / (self.dx * self.dx)

        # compute y^3
        y3 = np.power(self.y, 3)[1:-1]

        # y_t = (y(t+dt) - y(t)) / dt => y(t+dt) = y(t) + dt * (y^3 + y_xx)
        self.y[1:-1] += self.dt * (y3 + yxx)

    def get_obs(self):
        return np.concatenate([self.t], self.data['y'][0])


class SimStabilizeMidObs(Simulation):
    """
    Simulating
        y_t + f(y)_x = 0
        y(t,0) = y(t,1) + u(t, y(t,1/2))
    for some control u
    """
    def __init__(self, f, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.f = f

    def update_y(self, u=0):
        # boundary condition with control u dt
        self.y[0] = self.y[-1] + u  # - 0.2 * self.y[len(self.y) // 2] + u

        # compute df(y)/dx
        yx = np.diff(self.f(self.y)) / self.dx

        # dy/dt + df(y)/dx = 0  =>  y(t+dt) = y(t) - dt * df(y)/dx
        self.y[1:] -= self.dt * yx

    def get_obs(self):
        return np.array([self.t, self.y[len(self.y) // 2]])


if __name__ == '__main__':
    if True:
        sim = SimControlHeat(dt=1e-4, dx=1e-3, xmin=0, xmax=1, y0=lambda x: 5 * np.sin(np.pi * x))
        sim.reset()
        while sim.n_steps <= 10:
            sim.step((0, 0))
        sim.render(path='test.mp4', fps=np.inf, dpi=300, accel=0.0005)
        
    if False:
        sim = SimStabilizeMidObs(f=lambda x: x * x + x, dt=0.3 * 1e-3, dx=1e-3, xmin=0, xmax=1,)

        sim.reset(y0=lambda x: np.cos(x * np.pi * 8.0) * 0.01)
        while sim.t < 2.0:
            sim.step(0)
        sim.render(path='test.mp4', fps=120.0, dpi=300, accel=0.25)
