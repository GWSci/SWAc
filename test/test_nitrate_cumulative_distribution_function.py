from datetime import date
import swacmod.model as m
import swacmod.nitrate as nitrate
import numpy as np
import swacmod.timer as timer
import unittest

class Test_Nitrate_Cumulative_Distribution_Function(unittest.TestCase):
	def test_sum_of_daily_proportion_should_equal_1(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		DTW = 100
		sum = 0
		for t in range(1, 1000000):
			sum += nitrate._calculate_daily_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
		self.assertAlmostEqual(1, sum, places=2)

	def test_maximum_of_daily_proportion_should_appear_after_several_years(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		DTW = 100
		t = 0
		previous_nitrate = 0
		while True:
			current_nitrate = nitrate._calculate_daily_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t + 1)
			if current_nitrate < previous_nitrate:
				break
			t += 1
			previous_nitrate = current_nitrate
		self.assertEqual(6927, t)
	
	def test_calculate_daily_proportion_reaching_water_table_arr(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		data = {
			"time_switcher": timer.make_time_switcher(),
			"series": {
				"date" : np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
			}, "params": {
				"node_areas" : {
					3: 50
				}, "nitrate_depth_to_water" : {
					3: [0.001]
				}
			},
		}
		output = None
		node = 3
		blackboard = nitrate.NitrateBlackboard()
		expected = np.array([0.0, 0.793244, 0.120028])
		proportion_0 = None
		proportion_100 = None
		actual = nitrate._calculate_proportion_reaching_water_table_array_per_day(data, output, node, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, proportion_0, proportion_100, blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_cumulative_proportion_reaching_water_table_varies_with_params(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		DTW = 0.001
		t = 1
		original = nitrate._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
		different_a = nitrate._calculate_cumulative_proportion_reaching_water_table(100, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
		different_μ = nitrate._calculate_cumulative_proportion_reaching_water_table(a, 0.05, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
		different_σ = nitrate._calculate_cumulative_proportion_reaching_water_table(a, μ, 0.01, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, DTW, t)
		different_mean_hydraulic_conductivity = nitrate._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, 100, mean_velocity_of_unsaturated_transport, DTW, t)
		different_mean_velocity_of_unsaturated_transport = nitrate._calculate_cumulative_proportion_reaching_water_table(a, μ, σ, mean_hydraulic_conductivity, 0.1, DTW, t)
		self.assertEqual(0.793244345253982, original)
		self.assertEqual(0.6657750500569044, different_a)
		self.assertEqual(0.6668990408184825, different_μ)
		self.assertEqual(1.0, different_σ)
		self.assertEqual(0.00873035927116339, different_mean_hydraulic_conductivity)
		self.assertEqual(0.9998369172257393, different_mean_velocity_of_unsaturated_transport)

	def test_calculate_daily_proportion_reaching_water_table_arr_when_dtw_is_0(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		data = {
			"time_switcher": timer.make_time_switcher(),
			"series": {
				"date" : np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
			}, "params": {
				"node_areas" : {
					3: 2500
				}, "nitrate_depth_to_water" : {
					3: [0.0]
				}
			},
		}
		output = None
		node = 3
		proportion_0 = "xxx"
		proportion_100 = None
		blackboard = nitrate.NitrateBlackboard()
		actual = nitrate._calculate_proportion_reaching_water_table_array_per_day(data, output, node, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, proportion_0, proportion_100, blackboard)
		self.assertEqual("xxx", actual)

	def test_calculate_daily_proportion_reaching_water_table_arr_when_dtw_is_100(self):
		a = 1.38
		μ = 1.58
		σ = 3.96
		mean_hydraulic_conductivity = 1.7
		mean_velocity_of_unsaturated_transport = 0.0029
		data = {
			"time_switcher": timer.make_time_switcher(),
			"series": {
				"date" : np.array([date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30)])
			}, "params": {
				"node_areas" : {
					3: 2500
				}, "nitrate_depth_to_water" : {
					3: [100.0]
				}
			},
		}
		output = None
		node = 3
		proportion_0 = None
		blackboard = nitrate.NitrateBlackboard()
		proportion_100 = "xxx"
		actual = nitrate._calculate_proportion_reaching_water_table_array_per_day(data, output, node, a, μ, σ, mean_hydraulic_conductivity, mean_velocity_of_unsaturated_transport, proportion_0, proportion_100, blackboard)
		self.assertEqual("xxx", actual)
	
	def test_total_mass_leached_on_day_for_zero_days(self):
		proportion_reaching_water_table_array_per_day = np.array([])
		mi_array_kg_per_day = np.array([])
		mass_reaching_water_table_kg = []
		expected_total_mass_on_day_kg = np.array([])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_1_day(self):
		proportion_reaching_water_table_array_per_day = np.array([0.3])
		mi_array_kg_per_day = np.array([100.0])
		mass_reaching_water_table_kg = [[30.0]]
		expected_total_mass_on_day_kg = np.array([30.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day_for_two_days(self):
		proportion_reaching_water_table_array_per_day = np.array([0.3, 0.4])
		mi_array_kg_per_day = np.array([100.0, 200.0])
		mass_reaching_water_table_kg = [
			[30.0,  40.0],
			[ 0.0,  60.0,  80.0],
		]
		expected_total_mass_on_day_kg = np.array(
			[30.0, 100.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)

	def test_total_mass_leached_on_day(self):
		proportion_reaching_water_table_array_per_day = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
		mi_array_kg_per_day = np.array([100.0, 200.0, 0.0, 300.0, 250.0])
		mass_reaching_water_table_kg = [
			[0.0, 30.0,  40.0,  20.0,  10.0],
			[0.0,  0.0,  60.0,  80.0,  40.0,  20.0],
			[0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0],
			[0.0,  0.0,   0.0,   0.0,  90.0, 120.0,  60.0, 30.0],
			[0.0,  0.0,   0.0,   0.0,   0.0,  75.0, 100.0, 50.0, 30.0],
		]
		expected_total_mass_on_day_kg = np.array(
			[0.0, 30.0, 100.0, 100.0, 140.0])
		actual = m.calculate_mass_reaching_water_table_array_kg_per_day(proportion_reaching_water_table_array_per_day, mi_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected_total_mass_on_day_kg, actual)
