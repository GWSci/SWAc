import numpy
import swacmod.timer as t


def numpy_get_precipitation(data, nodes):
	token = t.start_timing("series and params")
	series, params = data['series'], data['params']
	t.stop_timing(token)

	token = t.start_timing("rainfall_zone_mapping = params['rainfall_zone_mapping']")
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	t.stop_timing(token)

	token = t.start_timing("zone_rf")
	zone_rf = numpy.array([rainfall_zone_mapping[node][0] - 1 for node in nodes])
	t.stop_timing(token)

	token = t.start_timing("coef_rf")
	coef_rf = numpy.array([rainfall_zone_mapping[node][1] for node in nodes])
	t.stop_timing(token)

	token = t.start_timing("rainfall_ts")
	rainfall_ts = numpy.transpose(series['rainfall_ts'][:, zone_rf]) * coef_rf[:, None]
	t.stop_timing(token)
	return rainfall_ts
