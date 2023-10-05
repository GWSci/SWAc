from datetime import date
import swacmod.nitrate as nitrate
import numpy as np
import unittest

class Test_Nitrate_Cumulative_Distribution_Function(unittest.TestCase):
	# def test_x(self):
	# 	DTW = 100
	# 	print("t,f,n_prop_i")
	# 	for t in range(0, 100000, 1000):
	# 		cumulative = nitrate._calculate_cumulative_proportion_reaching_water_table(DTW, t)
	# 		day = nitrate._calculate_daily_proportion_reaching_water_table(DTW, t)
	# 		print(f"{t},{cumulative},{day}")			

	def test_sum_of_daily_proportion_should_equal_1(self):
		DTW = 100
		sum = 0
		for t in range(1, 1000000):
			sum += nitrate._calculate_daily_proportion_reaching_water_table(DTW, t)
		self.assertAlmostEqual(1, sum, places=2)

	def test_maximum_of_daily_proportion_should_appear_after_several_years(self):
		DTW = 100
		t = 0
		previous_nitrate = 0
		while True:
			current_nitrate = nitrate._calculate_daily_proportion_reaching_water_table(DTW, t + 1)
			if current_nitrate < previous_nitrate:
				break
			t += 1
			previous_nitrate = current_nitrate
		self.assertEqual(6927, t)
	
	def test_calculate_daily_proportion_reaching_water_table_arr(self):
		data = {
			"series": {
				"date" : np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
			}, "params": {
				"node_areas" : {
					3: [2500]
				}, "nitrate_depth_to_water" : {
					3: [100]
				}
			},
		}
		output = None
		node = 3
		expected = np.array([0.0, 1.110223e-16, 8.770762e-15])
		actual = nitrate._calculate_daily_proportion_reaching_water_table_arr(data, output, node)
		np.testing.assert_array_almost_equal(expected, actual)

	
	def test_total_mass_leached_on_day_for_zero_days(self):
		proportion_reaching_water_table_array_per_day = np.array([])
		mi_array_kg_per_day = np.array([])
		mass_reaching_water_table_kg = []
		expected_total_mass_on_day_kg = np.array([])
		actual = nitrate. _calculate_total_mass_on_day_kg(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_1_day(self):
		proportion_reaching_water_table_array_per_day = np.array([0.3])
		mi_array_kg_per_day = np.array([100])
		mass_reaching_water_table_kg = [[30.0]]
		expected_total_mass_on_day_kg = np.array([30.0])
		actual = nitrate._calculate_total_mass_on_day_kg(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_two_days(self):
		proportion_reaching_water_table_array_per_day = np.array([0.3, 0.4])
		mi_array_kg_per_day = np.array([100, 200])
		mass_reaching_water_table_kg = [
			[30.0,  40.0],
			[ 0.0,  60.0,  80.0],
		]
		expected_total_mass_on_day_kg = np.array(
			[30.0, 100.0])
		actual = nitrate._calculate_total_mass_on_day_kg(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day(self):
		proportion_reaching_water_table_array_per_day = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
		mi_array_kg_per_day = np.array([100, 200, 0, 300, 250])
		mass_reaching_water_table_kg = [
			[0.0, 30.0,  40.0,  20.0,  10.0],
			[0.0,  0.0,  60.0,  80.0,  40.0,  20.0],
			[0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
			[0.0,  0.0,   0.0,   0.0,  90.0, 120.0,  60.0, 30.0],
			[0.0,  0.0,   0.0,   0.0,   0.0,  75.0, 100.0, 50.0, 30.0],
		]
		expected_total_mass_on_day_kg = np.array(
			[0.0, 30.0, 100.0, 100.0, 140.0])
		actual = nitrate._calculate_total_mass_on_day_kg(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)
