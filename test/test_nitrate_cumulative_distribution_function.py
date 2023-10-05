import math
import numpy as np
import unittest

class Test_Nitrate_Cumulative_Distribution_Function(unittest.TestCase):
	def test_sum_of_daily_proportion_should_equal_1(self):
		DTW = 100
		sum = 0
		for t in range(1, 1000000):
			sum += calculate_daily_proportion_reaching_water_table(DTW, t)
		self.assertAlmostEqual(1, sum, places=2)

	def test_maximum_of_daily_proportion_should_appear_after_several_years(self):
		DTW = 100
		t = 0
		previous_nitrate = 0
		while True:
			nitrate = calculate_daily_proportion_reaching_water_table(DTW, t + 1)
			if nitrate < previous_nitrate:
				break
			t += 1
			previous_nitrate = nitrate
		self.assertEqual(6927, t)
	
	def test_total_mass_leached_on_day_for_zero_days(self):
		daily_proportion_reaching_water_table = np.array([])
		mi_kg_per_day = np.array([])
		mass_reaching_water_table_kg = []
		expected_total_mass_on_day_kg = np.array([])
		actual = calculate_total_mass_on_day(daily_proportion_reaching_water_table, mi_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_1_day(self):
		daily_proportion_reaching_water_table = np.array([0.3])
		mi_kg_per_day = np.array([100])
		mass_reaching_water_table_kg = [
			[30.0]
		]
		expected_total_mass_on_day_kg = np.array(
			[30.0])
		actual = calculate_total_mass_on_day(daily_proportion_reaching_water_table, mi_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	# def test_total_mass_leached_on_day(self):
	# 	daily_proportion_reaching_water_table = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
	# 	mi_kg_per_day = np.array([100, 200, 0, 300, 250])
	# 	mass_reaching_water_table_kg = [
	# 		[0.0, 30.0,  40.0,  20.0,  10.0],
	# 		[0.0,  0.0,  60.0,  80.0,  40.0,  20.0],
	# 		[0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
	# 		[0.0,  0.0,   0.0,   0.0,  90.0, 120.0,  60.0, 30.0],
	# 		[0.0,  0.0,   0.0,   0.0,   0.0,  75.0, 100.0, 50.0, 30.0],
	# 	]
	# 	expected_total_mass_on_day_kg = np.array(
	# 		[0.0, 30.0, 100.0, 100.0, 140.0])
	# 	actual = calculate_total_mass_on_day(daily_proportion_reaching_water_table, mi_kg_per_day)
	# 	np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

def calculate_daily_proportion_reaching_water_table(DTW, t):
	f_t = calculate_cumulative_proportion_reaching_water_table(DTW, t)
	f_t_prev = calculate_cumulative_proportion_reaching_water_table(DTW, t - 1)
	return -(f_t - f_t_prev)

def calculate_cumulative_proportion_reaching_water_table(DTW, t):
	if (t <= 0):
		return 1

	a = 1.38
	μ = 1.58
	σ = 3.96

	numerator = math.log((1.7/0.0029) * (DTW/t), a) - μ
	denominator = σ * math.sqrt(2)

	result = 0.5 * math.erfc(- numerator / denominator)
	return result

def calculate_total_mass_on_day(daily_proportion_reaching_water_table, mi_kg_per_day):
	length = daily_proportion_reaching_water_table.size
	result = np.zeros(length)
	for i in range(length):
		for j in range(length):
			result[i] = daily_proportion_reaching_water_table[j] * mi_kg_per_day[j]
	return result