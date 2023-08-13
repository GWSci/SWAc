import unittest
import numpy as np

class test_get_pefac(unittest.TestCase):
	def test_x(self):
		data = {
			"series": {
				"date": [0, 0, 0, 0, 0], # Doesn't seem to matter what this is. Only its length is used.
				"months": np.array([0, 2, 1, 4, 3], dtype=np.int64),
			},
			"params": {
				"kc_list": np.array([
					[ 2,  3,  5],
					[ 7, 11, 13],
					[17, 19, 23],
					[29, 31, 37],
					[41, 43, 47],
				], dtype=np.float64),
				"lu_spatial": np.array([
					[53, 59, 61],
					[67, 71, 73],
				], dtype=np.float64),
				"fao_process": "enabled",
				"canopy_process": "enabled",
			},
		}
		output = {
			"pe_ts": np.array([79, 83, 97, 101, 103], dtype=np.float64),
		}
		node = 1
		expected = get_pefac_oracle(data, output, node)["pefac"]
		actual = get_pefac_optimised(data, output, node)["pefac"]
		np.testing.assert_array_equal(expected, actual)

# Taken from the original implementation of pefac
def get_pefac_oracle(data, output, node):
	"""E) Vegetation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
	series, params = data['series'], data['params']
	days = len(series['date'])
	pefac = np.zeros(days)
	var1 = 0.0
	pe = output['pe_ts']
	kc = params['kc_list'][series['months']]
	zone_lu = np.array(params['lu_spatial'][node],
									  dtype=np.float64)
	len_lu = len(params['lu_spatial'][node])

	fao = params['fao_process']
	canopy = params['canopy_process']

	if fao == 'enabled' or canopy == 'enabled':
		for day in range(days):
			var1 = 0.0
			for z in range(len_lu):
				var1 = var1 + (kc[day, z] * zone_lu[z])
			pefac[day] = pe[day] * var1

	return {'pefac': np.array(pefac)}

def get_pefac_optimised(data, output, node):
	"""E) Vegetation-factored Potential Evapotranspiration (PEfac) [mm/d]."""
	series, params = data['series'], data['params']
	days = len(series['date'])

	fao = params['fao_process']
	canopy = params['canopy_process']
	calculate_pefac = fao == 'enabled' or canopy == 'enabled'
	if not calculate_pefac:
		return {'pefac': np.zeros(days)}

	pe = output['pe_ts']
	kc = params['kc_list'][series['months']]
	zone_lu = np.array(params['lu_spatial'][node], dtype=np.float64)
	len_lu = len(params['lu_spatial'][node])

	var1_arr = np.zeros(days)

	var1_arr = np.sum(kc[:, 0:len_lu] * zone_lu[0:len_lu], axis=1)
	pefac = pe * var1_arr

	return {'pefac': np.array(pefac)}