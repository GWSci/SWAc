import numpy as np
from swacmod.utils import monthdelta2, weekdelta

def get_swabs(data, output, node):
	"""Surface water abstractions"""
	params = data['params']
	applicable_nodes = params['swabs_locs']
	return _get_swdis_and_swabs_helper(data, node, applicable_nodes, 'swabs_f', 'swabs_ts')

def get_swdis(data, output, node):
	"""Surface water discharges"""
	params = data['params']
	applicable_nodes = params['swdis_locs']
	return _get_swdis_and_swabs_helper(data, node, applicable_nodes, 'swdis_f', 'swdis_ts')

def _get_swdis_and_swabs_helper(data, node, applicable_nodes, flag_name, ts_name):
	series = data['series']
	dates = series['date']
	result = np.zeros(len(dates))
	if node in applicable_nodes:
		areas = data['params']['node_areas']
		fac = 1000.0
		freq_flag = data['params'][flag_name]
		start_date = dates[0]

		series_ts = series[ts_name]
		area = areas[node]
		zone = applicable_nodes[node] - 1

		if freq_flag == 0:
			# daily input just populate
			result = series_ts[:, zone] / area * fac

		elif freq_flag == 1:
			# if weeks convert to days
			for iday, day in enumerate(dates):
				week = weekdelta(start_date, day)
				result[iday] = (series_ts[week, zone] / area * fac)

		elif freq_flag == 2:
			global _months
			# if months convert to days
			months = np.zeros(len(dates), dtype=np.int64)
			for iday, day in enumerate(dates):
				months[iday] = monthdelta2(start_date, day)
			result = (series_ts[months, zone] / area * fac)

	return {ts_name: result}
