import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import numpy as np

def plot_shadow_curve(draw_keys,
                      x_dict_mean,
                      y_dict_mean,
                      x_dict_std,
                      y_dict_std,
                      ylim=None,
                      title=None,
                      xlabel=None,
                      ylabel=None,
                      save_name=None,
                      legend_dict=None,
                      linestyle_dict=None,
                      adjust=None,
                      colors=None,
                      legend_loc=None,
                      fill_alpha=0.2,
                      line_alpha=1.0,
                      linewidth=3,
                      img_size=(7, 5),
                      axis_size=15,
                      title_size=15,
                      legend_size=15):

    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig = plt.figure(figsize=img_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%01.2lf'))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w'] if colors is None else colors
    for key_idx in range(len(draw_keys)):
        key = draw_keys[key_idx]
        plt.fill_between(x_dict_std[key],
                         y_dict_mean[key] - y_dict_std[key],
                         y_dict_mean[key] + y_dict_std[key],
                         alpha=fill_alpha,
                         color=colors[key_idx],
                         edgecolor="w",
                         )
        plt.plot(x_dict_mean[key],
                 y_dict_mean[key],
                 color=colors[key_idx],
                 alpha=line_alpha,
                 linewidth=linewidth,
                 label=key if legend_dict is None else legend_dict[key],
                 linestyle='-' if linestyle_dict is None else linestyle_dict[key])

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if legend_size is not None:
        plt.legend(fontsize=legend_size, loc=legend_loc)  # upper left, upper right, lower left
    if title is not None:
        plt.title(title, fontsize=title_size)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=axis_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=axis_size)
    if adjust is not None:
        plt.subplots_adjust(left=adjust[0], right=adjust[1], top=adjust[2], bottom=adjust[3])
    if not save_name:
        plt.show()
    else:
        plt.savefig('{0}.png'.format(save_name))

    plt.close()


class Plot2D:
    """
    Contains 2D plotting functions (done through matplotlib)
    """

    def __init__(self, l={}, cmds=[], mode="static", interval=None,
                 n=None, rows=1, cols=1, gif=None, legend=False, **kwargs):
        """
        Initialize a Plot2D.
        """
        assert (mode in ["static", "dynamic"])
        assert (type(l) == dict)
        assert (type(cmds) == list)
        self.legend = legend
        self.gif = gif
        self.fps = 60
        self.l = l
        self.mode = mode
        if interval is None:
            self.interval = 200  # 200ms
        else:
            self.interval = interval
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, **kwargs)
        self.empty = True
        self.cmds = cmds
        self.n = n
        self.reset()
        if self.mode == "dynamic":
            if n is None:
                self.anim = FuncAnimation(self.fig, self.step,
                                          blit=False, interval=self.interval, repeat=False)
            else:
                self.anim = FuncAnimation(self.fig, self.step,
                                          blit=False, interval=self.interval, frames=range(n + 1),
                                          repeat=False)

    def reset(self):
        """
        Reset and draw initial plots.
        """
        self.t = 0
        self.clear()
        self.data = {}
        self.objs = {}
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=None, t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=None, t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def step(self, frame=None):
        """
        Increment the timer.
        """
        self.t += 1
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=self.objs[i], t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=self.objs[i], t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def getax(self, loc=None):
        """
        Get the relevant axes object.
        """
        if loc is None:
            axobj = self.ax
        elif type(loc) == int or (type(loc) == list and len(loc) == 1):
            loc = int(loc)
            axobj = self.ax[loc]
        else:
            assert (len(loc) == 2)
            axobj = self.ax[loc[0], loc[1]]
        return axobj

    def imshow(self, X, loc=None, o=None, **kwargs):
        """
        Imshow X.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            im = axobj.imshow(X, **kwargs)
            cbar = self.fig.colorbar(im, ax=axobj)
            return [im, cbar]
        else:
            im, cbar = o
            im.set_data(X)
            im.set_clim(np.min(X), np.max(X))
            # cbar.set_clim(np.min(X), np.max(X))
            # matplotlib.cm.ScalarMappable.set_clim
            return [im, cbar]

    def line(self, X, Y, loc=None, o=None, **kwargs):
        """
        Line plot X/Y where `loc` are the subplot indices.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            return axobj.plot(X, Y, **kwargs)[0]
        else:
            o.set_data(X, Y)
            return o

    def line_binary(self, X, Y, loc=None, o=None, trends=None,
                    trend_colors=['grey', 'pink'], **kwargs):
        """
        Line with two colors.
        """
        assert (trends != None)
        self.empty = False
        axobj = self.getax(loc=loc)
        ret = []
        n = len(X)
        lw = 3
        if n == 0:
            return None
        if o is not None and len(o) > 0:
            for oo in o:
                oo.remove()
        for i in range(n - 1):
            ret += [axobj.plot(X[i:i + 2], Y[i:i + 2], color=trend_colors[0] \
                if trends[i] == "-" else trend_colors[1], linewidth=lw, **kwargs)[0]]
        return ret

    def show(self, *args, **kwargs):
        """
        Show the entire plot in a nonblocking way.
        """
        if not self.empty:
            if not plt.get_fignums():
                # print("Figure closed!")
                return
            if hasattr(self, "shown") and self.shown == True:
                plt.draw()
                plt.pause(0.001)
                return
            if self.gif is None:
                plt.show(*args, **kwargs)
                self.shown = True
            else:
                assert (self.n is not None)
                plt.show(*args, **kwargs)
                self.shown = True
                self.anim.save(self.gif, writer=ImageMagickWriter(fps=self.fps,
                                                                  extra_args=['-loop', '1']),
                               progress_callback=lambda i, n: print("%d/%d" % (i, n)))

    def clear(self):
        """
        Clear the figure.
        """
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        item2.cla()
                else:
                    item.cla()
        else:
            self.ax.cla()
        self.empty = True