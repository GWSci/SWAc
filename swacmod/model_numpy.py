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

def lazy_get_precipitation(data, nodes):
	token = t.start_timing("lazy_get_precipitation")
	series, params = data['series'], data['params']
	rainfall_zone_mapping = params['rainfall_zone_mapping']
	zone_rf = numpy.array([rainfall_zone_mapping[node][0] - 1 for node in nodes])
	coef_rf = numpy.array([rainfall_zone_mapping[node][1] for node in nodes])
	result = Lazy_Precipitation(zone_rf, coef_rf, series['rainfall_ts'])
	t.stop_timing(token)
	return result

class Lazy_Precipitation:
	def __init__(self, zone_rf, coef_rf, rainfall_ts):
		self.zone_rf = zone_rf
		self.coef_rf = coef_rf
		self.rainfall_ts = rainfall_ts
	
	def __getitem__(self, subscript):
		zone_rf = self.zone_rf[subscript]
		coef_rf = self.coef_rf[subscript]
		rainfall_ts = self.rainfall_ts[:, zone_rf] * coef_rf
		return rainfall_ts
