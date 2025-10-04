import time


class PMTimer(object):
    """Context class for benchmarking

    ::

        >>> period = 0.2
        >>> with PMTimer() as timer:
        ...     time.sleep(period)
        >>> timer.wall_interval >= period
        True
    """

    def __init__(self, cpu_clock=time.clock, wall_clock=time.time):
        self.cpu_clock = cpu_clock
        self.wall_clock = wall_clock
        self.clock_start = None
        self.wall_start = None
        self.clock_interval = None
        self.wall_interval = None

    def __enter__(self):
        self.clock_start = self.cpu_clock()
        self.wall_start = self.wall_clock()
        return self

    def __exit__(self, *args):
        self.clock_interval = time.clock() - self.clock_start
        self.wall_interval = time.time() - self.wall_start

    def __str__(self):
        return "clock: %0.03f sec, wall: %0.03f sec." \
            % (self.clock_interval, self.wall_interval)

    def to_dict(self):
        return {'time_wall': self.wall_interval,
                'time_cpu': self.clock_interval}
