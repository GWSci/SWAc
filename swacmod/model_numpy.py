import numpy


def numpy_get_precipitation(data, nodes):
	series, params = data['series'], data['params']
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	zone_rf = numpy.array([rainfall_zone_mapping[node][0] - 1 for node in nodes])
	coef_rf = numpy.array([rainfall_zone_mapping[node][1] for node in nodes])
	rainfall_ts = numpy.transpose(series['rainfall_ts'][:, zone_rf]) * coef_rf[:, None]
	return rainfall_ts
