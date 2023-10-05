import math
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