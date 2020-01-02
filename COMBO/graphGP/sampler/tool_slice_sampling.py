import numpy as np


def univariate_slice_sampling(logp, x0, width=1.0, max_steps_out=10):
    """
    Univariate Slice Sampling using doubling scheme
    :param logp: numeric(float) -> numeric(float), a log density function
    :param x0: numeric(float)
    :param width:
    :param max_steps_out:
    :return: numeric(float), sampled x1
    """
    for scaled_width in np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]) * width:

        lower = x0 - scaled_width * np.random.rand()
        upper = lower + scaled_width
        llh0 = logp(x0)
        slice_h = np.log(np.random.rand()) + llh0
        llh_record = {}

        # Step Out (doubling)
        steps_out = 0
        logp_lower = logp(lower)
        logp_upper = logp(upper)
        llh_record[float(lower)] = logp_lower
        llh_record[float(upper)] = logp_upper
        while (logp_lower > slice_h or logp_upper > slice_h) and (steps_out < max_steps_out):
            if np.random.rand() < 0.5:
                lower -= (upper - lower)
            else:
                upper += (upper - lower)
            steps_out += 1
            try:
                logp_lower = llh_record[float(lower)]
            except KeyError:
                logp_lower = logp(lower)
                llh_record[float(lower)] = logp_lower
            try:
                logp_upper = llh_record[float(upper)]
            except KeyError:
                logp_upper = logp(upper)
                llh_record[float(upper)] = logp_upper

        # Shrinkage
        start_upper = upper
        start_lower = lower
        n_steps_in = 0
        while not np.isclose(lower, upper):
            x1 = (upper - lower) * np.random.rand() + lower
            llh1 = logp(x1)
            if llh1 > slice_h and accept(logp, x0, x1, slice_h, scaled_width, start_lower, start_upper, llh_record):
                return x1
            else:
                if x1 < x0:
                    lower = x1
                else:
                    upper = x1
            n_steps_in += 1
        # raise RuntimeError('Shrinkage collapsed to a degenerated interval(point)')
    return x0  # just returning original value


def accept(logp, x0, x1, slice_h, width, lower, upper, llh_record):
    acceptance = False
    while upper - lower > 1.1 * width:
        mid = (lower + upper) / 2.0
        if (x0 < mid and x1 >= mid) or (x0 >= mid and x1 < mid):
            acceptance = True
        if x1 < mid:
            upper = mid
        else:
            lower = mid
        try:
            logp_lower = llh_record[float(lower)]
        except KeyError:
            logp_lower = logp(lower)
            llh_record[float(lower)] = logp_lower
        try:
            logp_upper = llh_record[float(upper)]
        except KeyError:
            logp_upper = logp(upper)
            llh_record[float(upper)] = logp_upper
        if acceptance and slice_h >= logp_lower and slice_h >= logp_upper:
            return False
    return True


if __name__ == '__main__':
    pass