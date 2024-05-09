import swacmod.nitrate_proportion_reaching_water_table as nitrate_proportion
import unittest

class Test_Nitrate_Cumulative_Distribution_Function(unittest.TestCase):
	def test_sum_of_daily_proportion_should_equal_1(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		alpha = 1.7
		effective_porosity = 1.0 / 0.0029
		DTW = 100
		sum = 0
		for t in range(1, 1000000):
			sum += nitrate_proportion._calculate_daily_proportion_reaching_water_table(a, μ, σ, alpha, effective_porosity, DTW, t)
		self.assertAlmostEqual(1, sum, places=2)
