import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
from collections import defaultdict
from abc import ABC


class Simulation(ABC):
    def __init__(self, dt, dx, xmin, xmax, y0=None, y0_generator=None, tmax=None):
        """Initialize the simulation.
        
        dt: time delta (in seconds)
        dx: space delta
        xmin: space lower bound
        xmax: space upper bound
        n_steps_per_action: number of times the simulation is updated with the same action
            (increase this to reduce the number of environment steps while keeping the same
            number of simulation steps, with the same control action applied several times)
        y0: initial condition function (ideally vectorized)
            input: np.array of x's ; output: np.array of the same size of y0's
        y0_generator: function called at each reset, that takes in no parameters and returns
                      and y0 initial condition function. If None, y0 must be specified and will
                      be kept at each reset.
        tmax: maximum simulation time. Note that this is not actually used for resetting the
            simulation, it is purely informational as to the maximum time a simulation
            is expected to run.
        """
        # store config
        self.dt = dt
        self.dx = dx
        self.xmin = xmin
        self.xmax = xmax
        self.tmax = tmax

        # initialize simulation
        if y0 is None and y0_generator is None:
            raise ValueError("At least one of y0 or y0_generator must be specified.")
        self.y0_generator = y0_generator
        self.reset(y0)

    def reset(self, y0=None):
        """Reset the simulation. May be given a new initial condition y0 (function taking x as an input), if not keep the previous one."""
        # otherwise if we specify a new initial condition, use that
        if y0 is not None:
            self.y0 = y0
        # otherwise if we have an initial condition generator, sample a new initial condition from it
        elif self.y0_generator is not None:
            self.y0 = self.y0_generator()

        # reset counters
        self.t = 0

        # reset initial
        self.x = np.arange(self.xmin, self.xmax + 1e-9, self.dx)

        # create initial condition, try using function in a vectorized way otherwise create elements one by one
        try:
            self.y = np.array(self.y0(self.x))
        except:
            self.y = np.array(list(map(self.y0, self.x)))

        # initialize storage
        self.t_lst = [self.t]
        self.y_lst = [np.copy(self.y)]

    @property
    def n_controls(self):
        """Return the number of control inputs, ie. the size of the vector u."""
        raise NotImplementedError

    def update_y(self, u):
        """Update self.y in-place given a control input u. May also access self.x, self.dt and self.dy."""
        raise NotImplementedError

    def step(self, u=0):
        """Execute one step of the simulation."""
        # apply control and increment time
        self.t += self.dt
        self.update_y(u)

        # save for later plotting/metrics
        # TODO i think there is no need to do this during training, takes time
        self.t_lst.append(self.t)
        self.y_lst.append(np.copy(self.y))

    def get_obs(self):
        """Return a 1D numpy array given as an input observation to the control."""
        raise NotImplementedError

    def reward(self):
        """Return the reward for the current timestep."""
        raise NotImplementedError

    def norm_y(self):
        """Compute the approximate square root of the integral of y."""
        return (self.dx * np.sum(self.y * self.y)) ** 0.5

    def render(
        self, path=None, display=True, fps=30.0, dpi=100, speed=1.0, no_video=True
    ):
        """Render the simulation from t=0 to the current time."""
        # compute which frames should be display to get the desired FPS and speed
        # we have `n_frames` frames over `total_time` seconds
        # we wish `fps` frames per second over T/`speed` seconds, so `fps`*`total_time`/`speed` frames displayed
        # so we're showing every `n_frames` / (`fps`*`total_time`/`speed`) frames

        if no_video == True:
            curves = []
            for i in range(len(self.y_lst[0])):
                curves.append([val[i] for val in self.y_lst])
            # plt.plot(
            #     self.t_lst,
            #     [
            #         (curves[3][i] + curves[1][i]) / curves[1][i]
            #         for i in range(len(curves[1]))
            #     ],
            # )
            for i, curve in enumerate(curves):
                plt.plot(self.t_lst, curve)
            plt.show()
            return None
        frames = np.array(self.y_lst)
        n_frames = frames.shape[0]
        total_time = np.max(self.t_lst)
        show_every_n_frames = int(n_frames * speed / (fps * total_time))
        frames_displayed = frames[::show_every_n_frames]
        times = self.t_lst[::show_every_n_frames]

        # create figure
        fig, ax = plt.subplots(dpi=dpi)

        # create initial t=0 plot
        # TODO might be faster not to plot all x's at a dx=1e-3 scale
        (line,) = ax.plot(self.x, frames_displayed[0])
        text = plt.text(0.8, 0.9, "", fontsize=10, transform=fig.axes[0].transAxes)

        # configure figure
        plt.xlim(self.xmin - 0.1, self.xmax + 0.1)
        try:
            plt.ylim(np.nanmin(self.y_lst) - 0.1, np.nanmax(self.y_lst) + 0.1)
        except:
            # inf
            pass
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("System Evolution")

        # animation function
        def animate(i):
            line.set_ydata(frames_displayed[i])
            text.set_text(f"t = {times[i]:.3f} s")
            return line, text

        # create animation
        anim = animation.FuncAnimation(
            fig,
            func=animate,
            frames=len(frames_displayed),
            interval=1000 / fps,
            repeat=True,
            blit=True,
        )

        # optionally display and/or save the animation
        if display:
            plt.show()
        if path is not None:
            anim.save(path)
