import numpy as np
import quaternion
import scipy
from scipy.interpolate import interp2d, interp1d
def compute_output_time(all_sources, sample_rate=200):
    """
    Compute the output reference time from all data sources. The reference time range must be within the time range of
    all data sources.
    :param data_all:
    :param sample_rate:
    :return:
    """
    interval = 1. / sample_rate
        
    min_t = max([data[0, 0] for data in all_sources.values()]) + interval
    max_t = min([data[-1, 0] for data in all_sources.values()]) - interval
    # print(min_t, max_t)
    return np.arange(min_t, max_t, interval)

def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.

    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

def interpolate_quaternion_linear(data, ts_in, ts_out):
    """
    This function interpolate the input quaternion array into another time stamp.
    Args:
        data: Nx4 array containing N quaternions.
        ts_in: input_timestamp- N-sized array containing time stamps for each of the input quaternion.
        ts_out: output_timestamp- M-sized array containing output time stamps.
    Return:
        Mx4 array containing M quaternions.
    """

    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)

def process_data_source(raw_data, output_time, method):
    input_time = raw_data[:, 0]
    if method == 'vector':
        output_data = interpolate_vector_linear(raw_data[:, 1:], input_time, output_time)
    elif method == 'quaternion':
        assert raw_data.shape[1] == 5
        output_data = interpolate_quaternion_linear(raw_data[:, 1:], input_time, output_time)
    else:
        raise ValueError('Interpolation method must be "vector" or "quaternion"')
    return output_data

def _find_nearest(target, source):
    # Find indices of nearest values of source in target (e.g match two timestamps) Assume two arrays are sorted.
    source = np.atleast_1d(source)
    assert np.all(np.diff(target) >= 0), 'Target array is not sorted'
    assert np.all(np.diff(source) >= 0), 'Value array is not sorted'

    indices, s = [], 0
    for v in source:
        i = np.abs(target[s:] - v).argmin()
        indices.append(s + i)
        s = i
    indices = np.asarray(indices)
    return indices

def find_sparse(time_interval, flp_time, flp_data):
    # Pick data points from flp data series at a given interval
    timestamp = np.arange(flp_time[0], flp_time[-1], time_interval)
    new_time_indices = _find_nearest(flp_time, timestamp)
    return np.concatenate([flp_time[new_time_indices][:, None], flp_data[new_time_indices]], axis=1)


def match_timestamps(timestamp, flp_time, flp_data, allow_err=0, interpolate=False):
    """
    Match FLP data to the nearest timestamps
    :param timestamp: array (l) timestamp of data points in seconds
    :param flp_time: array (n) timestamp of fla data points in seconds
    :param flp_data: ndarray (nxm), data series
    :param allow_err: use data before and after the data series (in seconds)
    :param interpolate: interpolate data to match the timestamps
    :return: matched [indices of timestamp, data...(m), timestamp] x n
    """
    valid_points = \
    np.where(np.logical_and(flp_time >= timestamp[0] - allow_err, flp_time <= timestamp[-1] + allow_err))[0]
    flp_time, flp_data = flp_time[valid_points], flp_data[valid_points]
    new_time_indices = _find_nearest(timestamp, flp_time)
    if interpolate:
        new_data = interp1d(flp_time, flp_data, axis=0, bounds_error=False, fill_value="extrapolate")(
            timestamp[new_time_indices])
        return np.concatenate([new_time_indices[:, None], new_data, timestamp[new_time_indices][:, None]], axis=1)
    else:
        return np.concatenate([new_time_indices[:, None], flp_data, flp_time[:, None]], axis=1)

