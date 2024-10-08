from datetime import date
import swacmod.model as m
import swacmod.nitrate as nitrate
import swacmod.nitrate_proportion_reaching_water_table as nitrate_proportion
import numpy as np
import swacmod.timer as timer
import unittest

class Test_Nitrate_Cumulative_Distribution_Function(unittest.TestCase):
	def test_maximum_of_daily_proportion_should_appear_after_several_years(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		alpha = 1.7
		effective_porosity = 1.0 / 0.0029
		DTW = 100
		t = 0
		previous_nitrate = 0
		while True:
			current_nitrate = nitrate_proportion._calculate_daily_proportion_reaching_water_table(a, μ, σ, alpha, effective_porosity, DTW, t + 1)
			if current_nitrate < previous_nitrate:
				break
			t += 1
			previous_nitrate = current_nitrate
		self.assertEqual(6927, t)
	
	def test_calculate_daily_proportion_reaching_water_table_arr(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.a = 1.38
		blackboard.μ = np.array([1.58])
		blackboard.σ = 3.96
		blackboard.alpha = 1.7
		blackboard.effective_porosity = np.array([1.0 / 0.0029])
		blackboard.node = 3
		blackboard.time_switcher = timer.make_time_switcher()
		blackboard.days = np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
		blackboard.cell_area_m_sq = 50
		blackboard.nitrate_depth_to_water = [0.001]

		expected = np.array([0.0, 0.793244, 0.120028])
		actual = nitrate._calculate_proportion_reaching_water_table_array_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_cumulative_proportion_reaching_water_table_varies_with_params(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		alpha = 1.7
		effective_porosity = 1.0 / 0.0029
		DTW = 0.001
		t = 1
		original = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, alpha, effective_porosity, DTW, t)
		different_a = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(100, μ, σ, alpha, effective_porosity, DTW, t)
		different_μ = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(a, 0.05, σ, alpha, effective_porosity, DTW, t)
		different_σ = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(a, μ, 0.01, alpha, effective_porosity, DTW, t)
		different_alpha = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, 100, effective_porosity, DTW, t)
		different_effective_porosity = nitrate_proportion._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, alpha, 0.1, DTW, t)
		self.assertAlmostEqual(0.793244345253982, original)
		self.assertAlmostEqual(0.6657750500569044, different_a)
		self.assertAlmostEqual(0.6668990408184825, different_μ)
		self.assertAlmostEqual(1.0, different_σ)
		self.assertAlmostEqual(0.00873035927116339, different_alpha)
		self.assertAlmostEqual(0.999999999999708, different_effective_porosity)

	def test_calculate_daily_proportion_reaching_water_table_arr_when_dtw_is_0(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.a = 1.38
		blackboard.μ = np.array([1.58])
		blackboard.σ = 3.96
		blackboard.alpha = 1.7
		blackboard.effective_porosity = np.array([1.0 / 0.0029])
		blackboard.time_switcher = timer.make_time_switcher()
		blackboard.days = np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
		blackboard.cell_area_m_sq = 2500
		blackboard.nitrate_depth_to_water = [0.0]
		blackboard.output = None
		blackboard.node = 3

		actual = nitrate._calculate_proportion_reaching_water_table_array_per_day(blackboard)
		np.testing.assert_array_almost_equal([1.0, 0.0, 0.0], actual)
	
	def test_total_mass_leached_on_day_for_zero_days(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.proportion_reaching_water_table_array_per_day = np.array([])
		blackboard.mi_array_kg_per_day = np.array([])

		expected_total_mass_on_day_kg = np.array([])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_1_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.proportion_reaching_water_table_array_per_day = np.array([0.3])
		blackboard.mi_array_kg_per_day = np.array([100.0])

		expected_total_mass_on_day_kg = np.array([30.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_two_days(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.proportion_reaching_water_table_array_per_day = np.array([0.3, 0.4])
		blackboard.mi_array_kg_per_day = np.array([100.0, 200.0])

		expected_total_mass_on_day_kg = np.array(
			[30.0, 100.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.proportion_reaching_water_table_array_per_day = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
		blackboard.mi_array_kg_per_day = np.array([100.0, 200.0, 0.0, 300.0, 250.0])

		expected_total_mass_on_day_kg = np.array(
			[0.0, 30.0, 100.0, 100.0, 140.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)
