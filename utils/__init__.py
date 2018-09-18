import sys
import numpy
import PyQt5
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from collections import deque
from time import time


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


class Report:
    def __init__(self, epochs, mode='t'):
        self._message = ''
        self._epochs = epochs
        self._mode = mode
        self._total_loss = 0.0
        self._total_sse = 0.0
        self._iterations = 0
        self._durations = deque(maxlen=10)
        self._time = time()

    def report(self, loss, sse):
        self._total_loss += loss
        self._total_sse += sse

        self._iterations += 1

        t1 = time()
        self._durations.append(t1 - self._time)
        self._time = t1

        self._message = \
            '\r\x1b[0;{m};40m'\
            'Epoch: {e}'\
            ' Iteration: {i}'\
            ' Loss: {l:.3e}'\
            ' Avg: {la:.3e}'\
            ' SSE: {s:.3e}'\
            ' Avg: {sa:.3e}'\
            ' It./s: {r:.3f}'\
            '\x1b[0m     '.format(
                m=37 if self._mode == 't' else 33,
                e=self._epochs,
                i=self._iterations,
                l=loss,
                la=self.avg_loss,
                s=sse,
                sa=self.avg_sse,
                r=self.speed
            )
        print(self._message, file=sys.stderr, end='')

    @property
    def avg_loss(self):
        try:
            return self._total_loss / self._iterations
        except ZeroDivisionError:
            return 0.0

    @property
    def avg_sse(self):
        try:
            return self._total_sse / self._iterations
        except ZeroDivisionError:
            return 0.0

    @property
    def speed(self):
        try:
            return len(self._durations) / sum(self._durations)
        except ZeroDivisionError:
            return 0.0

    @property
    def iterations(self):
        return self._iterations
