from datetime import date
import unittest
import numpy as np
import os
from swacmod import compile_model
import swacmod.model as m
import swacmod.nitrate as nitrate
import swacmod.timer as timer

class Test_Nitrate(unittest.TestCase):
	def test_calculate_daily_HER(self):
		input_rainfall_ts = np.array([110.0, 220.0, 330.0])
		input_ae = np.array([10.0, 20.0, 30.0])
		expected = np.array([100.0, 200.0, 300.0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def test_calculate_daily_HER_can_be_zero(self):
		input_rainfall_ts = np.array([110.0, 0.0, 330.0])
		input_ae = np.array([10.0, 0.0, 330.0])
		expected = np.array([100.0, 0.0, 0.0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def test_calculate_daily_HER_cannot_be_less_than_zero(self):
		input_rainfall_ts = np.array([110.0, -2.0, 0.0, 3.0])
		input_ae = np.array([10.0, 0.0, 1.0, 5.0])
		expected = np.array([100.0, 0.0, 0.0, 0])
		self.assert_her(input_rainfall_ts, input_ae, expected)

	def assert_her(self, input_rainfall_ts, input_ae, expected):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.rainfall_ts = input_rainfall_ts
		blackboard.ae = input_ae
		actual = nitrate._calculate_her_array_mm_per_day(blackboard)
		np.testing.assert_array_equal(expected, actual)

	def test_cumulative_fraction_leaked_per_year(self):
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 10.0, 0.05)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 70.0, 0.32)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 110.0, 0.5)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 200.0, 0.7025)
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 310.0, 0.95)

	def test_cumulative_fraction_leaked_per_year_can_be_more_than_1(self):
		self.assert_cumulative_fraction_leaked_per_year(10.0, 110.0, 310.0, 376.666666, 1.1)

	def test_cumulative_fraction_leaked_per_year_when_gradient_is_zero(self):
		self.assert_cumulative_fraction_leaked_per_year(110.0, 110.0, 110.0, 376.666666, 0.5)

	def test_cumulative_fraction_leaked_per_year_cannot_be_less_than_than_0(self):
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 5, 0.03)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 4, 0.02)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 3, 0.01)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 2, 0.00)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 1, 0.00)
		self.assert_cumulative_fraction_leaked_per_year(7.0, 52.0, 102.0, 0, 0.00)

	def assert_cumulative_fraction_leaked_per_year(self, her_at_5_percent, her_at_50_percent, her_at_95_percent, her, expected):
		actual = m._cumulative_fraction_leaked_per_year(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)
		self.assertAlmostEqual(expected, actual)

	def test_cumulative_fraction_leaked_per_day(self):
		her_at_5_percent = 10.0
		her_at_50_percent = 110.0
		her_at_95_percent = 310.0

		testee = lambda her: m._cumulative_fraction_leaked_per_day(
				her_at_5_percent, her_at_50_percent, her_at_95_percent, her)

		self.assertAlmostEqual(0.05 / 365.25, testee(10.0 / 365.25))
		self.assertAlmostEqual(0.32 / 365.25, testee(70.0 / 365.25))
		self.assertAlmostEqual(0.5 / 365.25, testee(110.0 / 365.25))
		self.assertAlmostEqual(0.7025 / 365.25, testee(200.0 / 365.25))
		self.assertAlmostEqual(0.95 / 365.25, testee(310.0 / 365.25))

	def test_calculate_total_mass_leached_from_cell_on_days(self):
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([], testee([], np.array([])))
		np.testing.assert_array_equal([2000.0], testee([date(2023, 1, 1)], np.array([20.0])))
		np.testing.assert_array_equal([2000.0, 8000.0,], testee([date(2023, 1, 1), date(2023, 1, 2)], np.array([20.0, 80.0])))

	def test_calculate_total_mass_leached_from_cell_on_days_limits_by_max_load_for_the_year(self):
		max_load_per_year = 10000 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], np.array([150 * 365.25])))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], np.array([60 * 365.25, 60 * 365.25, 60 * 365.25])))

	def test_calculate_total_mass_leached_from_cell_on_days_resets_limit_on_1st_october(self):
		max_load_per_year = 10000 * 365.25
		her_for_60_percent = 60 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], np.array([150 * 365.25])))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0, 0.6 * max_load_per_year, 0.4 * max_load_per_year, 0],
			testee(
				[date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
				np.array([her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent, her_for_60_percent])))

	def test_calculate_total_mass_leached_from_cell_on_days_resets_limit_on_1st_october_when_her_equals_zero(self):
		max_load_per_year = 10000 * 365.25
		her_for_60_percent = 60 * 365.25
		testee = calculate_total_mass_leached_for_test
		np.testing.assert_array_equal([max_load_per_year], testee([date(2023, 1, 1)], np.array([150 * 365.25])))
		np.testing.assert_array_equal(
			[0.6 * max_load_per_year, 0.4 * max_load_per_year, 0, 0, 0.6 * max_load_per_year, 0.4 * max_load_per_year],
			testee(
				[date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
				np.array([her_for_60_percent, her_for_60_percent, her_for_60_percent, 0.0, her_for_60_percent, her_for_60_percent])))

	def test_calculate_m0_array_kg_per_day(self):
		max_load_per_year = 10000 * 365.25 * 4
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		max_load_per_cell_per_year = 10000 * 365.25
		expected = [0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0, 0.6 * max_load_per_cell_per_year, 0.4 * max_load_per_cell_per_year, 0]

		blackboard = nitrate.NitrateBlackboard()
		blackboard.node = 3
		blackboard.time_switcher = timer.make_time_switcher()
		blackboard.days = [date(2023, 9, 28), date(2023, 9, 29), date(2023, 9, 30), date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)]
		blackboard.cell_area_m_sq = 2500
		blackboard.her_array_mm_per_day = np.array([60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25, 60 * 365.25])
		# Node,UNIQUE,X,Y,LOAD0,HER_5_MaxL,HER_50_Max,HER_95_Max,5PercLoadM,50PercLoad,95PercLoad
		blackboard.nitrate_loading = [0, 0, 0, max_load_per_year, her_at_5_percent, her_at_50_percent, her_at_95_percent, 0, 0, 0]
		actual = nitrate._calculate_m0_array_kg_per_day(blackboard)

		np.testing.assert_array_equal(expected, actual)

	def test_calculate_Pro_is_normally_one_minus_the_other_three_proportions(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.her_array_mm_per_day = np.array([1.0, 1.0])
		blackboard.p_non = np.array([0.1, 0.01])
		blackboard.Pherperc = np.array([0.2, 0.02])
		blackboard.Psmd = np.array([0.4, 0.04])

		actual = nitrate._calculate_Pro(blackboard)

		expected = np.array([0.3, 0.93])
		np.testing.assert_allclose(expected, actual, )

	def test_calculate_Pro_is_zero_when_her_is_zero(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0])
		blackboard.p_non = np.array([0.1, 0.01])
		blackboard.Pherperc = np.array([0.2, 0.02])
		blackboard.Psmd = np.array([0.4, 0.04])

		actual = nitrate._calculate_Pro(blackboard)

		expected = np.array([0.0, 0.0])
		np.testing.assert_allclose(expected, actual, )

	def test_calculate_Pro_is_zero_when_her_is_negative(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.her_array_mm_per_day = np.array([-1.0, -1.0])
		blackboard.p_non = np.array([0.1, 0.01])
		blackboard.Pherperc = np.array([0.2, 0.02])
		blackboard.Psmd = np.array([0.4, 0.04])

		actual = nitrate._calculate_Pro(blackboard)

		expected = np.array([0.0, 0.0])
		np.testing.assert_allclose(expected, actual)

	def test_calculate_Psmd_is_normally_dSMD_divided_by_HER(self):
		dSMD_array_mm_per_day = [0.2, 30.0]
		her_array_mm_per_day = [10.0, 60.0]
		expected = [0.02, 0.5]
		self.assert_Psmd(expected, dSMD_array_mm_per_day, her_array_mm_per_day)

	def test_calculate_Psmd_is_zero_when_dSMD_is_negative(self):
		dSMD_array_mm_per_day = [-0.2, -30.0]
		her_array_mm_per_day = [10.0, 60.0]
		expected = [0.0, 0.0]
		self.assert_Psmd(expected, dSMD_array_mm_per_day, her_array_mm_per_day)

	def test_calculate_Psmd_is_zero_when_HER_is_zero(self):
		dSMD_array_mm_per_day = [0.2, 30.0]
		her_array_mm_per_day = [0.0, 0.0]
		expected = [0.0, 0.0]
		self.assert_Psmd(expected, dSMD_array_mm_per_day, her_array_mm_per_day)

	def test_calculate_Psmd_is_set_to_1_when_it_would_otherwise_be_greater(self):
		dSMD_array_mm_per_day = [5.0, 0.2]
		her_array_mm_per_day = [2.0, 0.001]
		expected = [1.0, 1.0]
		self.assert_Psmd(expected, dSMD_array_mm_per_day, her_array_mm_per_day)

	def assert_Psmd(self, expected_Psmd, input_dSMD_array_mm_per_day, input_her_array_mm_per_day):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.dSMD_array_mm_per_day = np.array(input_dSMD_array_mm_per_day)
		blackboard.her_array_mm_per_day = np.array(input_her_array_mm_per_day)

		actual = nitrate._calculate_Psmd(blackboard)

		expected = np.array(expected_Psmd)
		np.testing.assert_allclose(expected, actual)

	def test_calculate_Pro_is_zero_when_the_other_three_proportions_are_greater_than_one(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.her_array_mm_per_day = np.array([1.0, 1.0])
		blackboard.p_non = np.array([0.3, 0.6])
		blackboard.Pherperc = np.array([0.4, 0.7])
		blackboard.Psmd = np.array([0.5, 0.8])
		blackboard.runoff_recharge_mm_per_day = np.zeros(2)
		blackboard.macropore_att_mm_per_day = np.zeros(2)
		blackboard.macropore_dir_mm_per_day = np.zeros(2)
		blackboard.perc_through_root_mm_per_day = np.zeros(2)
		blackboard.dSMD_array_mm_per_day = np.zeros(2)
		blackboard.logging = DummyLogger()

		actual = nitrate._calculate_Pro(blackboard)

		expected = np.array([0.0, 0.0])
		np.testing.assert_allclose(expected, actual, )

	def test_calculate_m1_array_kg_per_day_for_zero_days_new(self):
		input_Psoilperc = []
		input_M_soil_tot_kg = []
		input_M_soil_in_kg = []
		expected = []
		self.assert_m1_array_kg_per_day(expected, input_Psoilperc, input_M_soil_tot_kg, input_M_soil_in_kg)

	def test_calculate_m1_array_kg_per_day_for_one_day_new(self):
		input_Psoilperc = [0.2]
		input_M_soil_tot_kg = [7.0]
		input_M_soil_in_kg = [15.0]
		expected = [3.0]
		self.assert_m1_array_kg_per_day(expected, input_Psoilperc, input_M_soil_tot_kg, input_M_soil_in_kg)

	def test_calculate_m1_array_kg_per_day_for_two_days_new(self):
		input_Psoilperc = [0.2, 0.1]
		input_M_soil_tot_kg = [10.0, 20.0]
		input_M_soil_in_kg = [15.0, 20.0]
		expected = [3.0, 3.0]
		self.assert_m1_array_kg_per_day(expected, input_Psoilperc, input_M_soil_tot_kg, input_M_soil_in_kg)

	def assert_m1_array_kg_per_day(self, expected, input_Psoilperc, input_M_soil_tot_kg, input_M_soil_in_kg):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.Psoilperc = np.array(input_Psoilperc)
		blackboard.M_soil_tot_kg = np.array(input_M_soil_tot_kg)
		blackboard.M_soil_in_kg = np.array(input_M_soil_in_kg)
		expected_numpy = np.array(expected)
		actual = nitrate._calculate_m1_array_kg_per_day(blackboard)
		np.testing.assert_array_equal(expected_numpy, actual)

	def test_calculate_m1a_array_kg_per_day_for_just_one_day(self):
		input_interflow_volume = np.array([40.0])
		input_infiltration_recharge = np.array([20.0])
		input_interflow_to_rivers = np.array([40.0])
		input_m1_array_kg_per_day = np.array([12.0])
		expected = np.array([2.4])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)

	def test_calculate_m1a_array_kg_per_day_for_three_days_with_accumulation_in_MiT(self):
		input_interflow_volume = np.array([40.0, 35.0, 90.0])
		input_infiltration_recharge = np.array([20.0, 30.0, 10.0])
		input_interflow_to_rivers = np.array([40.0, 35.0, 0.0])
		input_m1_array_kg_per_day = np.array([12, 25.2, 39.5])
		expected = np.array([2.4, 9.0, 5.0])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)
	
	def test_calculate_m1a_array_kg_per_day_when_interflow_store_components_equal_zero(self):
		input_interflow_volume = np.array([0.0, 0.0, 40.0])
		input_infiltration_recharge = np.array([0.0, 0.0, 20.0])
		input_interflow_to_rivers = np.array([0.0, 0.0, 40.0])
		input_m1_array_kg_per_day = np.array([4.0, 4.0, 4.0])
		expected = np.array([0.0, 0.0, 2.4])
		self.assert_m1a_array(input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected)

	def assert_m1a_array(self, input_interflow_volume, input_infiltration_recharge, input_interflow_to_rivers, input_m1_array_kg_per_day, expected):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m1_array_kg_per_day = input_m1_array_kg_per_day
		blackboard.interflow_volume = input_interflow_volume
		blackboard.infiltration_recharge = input_infiltration_recharge
		blackboard.interflow_to_rivers = input_interflow_to_rivers

		actual = nitrate._calculate_m1a_b_array_kg_per_day(blackboard)[0, :]
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_is_sum_of_runoff_att_and_dir_when_her_is_less_than_zero(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([2.0, 0.0, 0.0, 2.0])
		blackboard.Pherperc = np.array([1.0, 1.0, 1.0, 1.0])
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0, 0.0, 0.0])
		blackboard.Psmd = np.array([1.0, 1.0, 1.0, 1.0])
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0, 0.0, 0.0])
		blackboard.macropore_att_mm_per_day = np.array([0.0, 30.0, 0.0, 30.0])
		blackboard.macropore_dir_mm_per_day = np.array([0.0, 0.0, 500.0, 500.0])
		blackboard.her_array_mm_per_day = np.array([-1.0, -2.0, -3.0, -4.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		expected = np.array([2.0, 30.0, 500.0, 532.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_is_sum_of_runoff_att_and_dir_when_her_is_equal_to_zero(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([2.0, 0.0, 0.0, 2.0])
		blackboard.Pherperc = np.array([1.0, 1.0, 1.0, 1.0])
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0, 0.0, 0.0])
		blackboard.Psmd = np.array([1.0, 1.0, 1.0, 1.0])
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0, 0.0, 0.0])
		blackboard.macropore_att_mm_per_day = np.array([0.0, 30.0, 0.0, 30.0])
		blackboard.macropore_dir_mm_per_day = np.array([0.0, 0.0, 500.0, 500.0])
		blackboard.her_array_mm_per_day = np.array([0.0, 0.0, 0.0, 0.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		expected = np.array([2.0, 30.0, 500.0, 532.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_contains_runoff_and_both_macropore_terms_in_numerator_and_denominator_when_HER_is_positive(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([5.0])
		blackboard.Pherperc = np.array([0.0])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.Psmd = np.array([0.0])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.macropore_att_mm_per_day = np.array([3.0])
		blackboard.macropore_dir_mm_per_day = np.array([2.0])
		blackboard.her_array_mm_per_day = np.array([1.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		numerator = 5.0 + 3.0 + 2.0 - 1.0
		denominator = 5.0 + 3.0 + 2.0
		expected_value = numerator / denominator
		expected = np.array([expected_value])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_contains_HER_scaled_by_Pherperc_when_HER_is_positive(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([5.0])
		blackboard.Pherperc = np.array([0.2])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.Psmd = np.array([0.0])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.macropore_att_mm_per_day = np.array([3.0])
		blackboard.macropore_dir_mm_per_day = np.array([2.0])
		blackboard.her_array_mm_per_day = np.array([10.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		her_scaled_by_pherperc = 10.0 * 0.2
		runoff_and_macropore_minus_her = 5.0 + 3.0 + 2.0 - 10.0
		numerator = runoff_and_macropore_minus_her + her_scaled_by_pherperc
		denominator = 5.0 + 3.0 + 2.0
		expected_value = numerator / denominator
		expected = np.array([expected_value])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_contains_HER_scaled_by_Psmd_when_HER_is_positive(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([5.0])
		blackboard.Pherperc = np.array([0.0])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.Psmd = np.array([0.3])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.macropore_att_mm_per_day = np.array([3.0])
		blackboard.macropore_dir_mm_per_day = np.array([2.0])
		blackboard.her_array_mm_per_day = np.array([10.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		her_scaled_by_psmd = 10.0 * 0.3
		runoff_and_macropore_minus_her = 5.0 + 3.0 + 2.0 - 10.0
		numerator = runoff_and_macropore_minus_her + her_scaled_by_psmd
		denominator = 5.0 + 3.0 + 2.0
		expected_value = numerator / denominator
		expected = np.array([expected_value])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non_her_contains_all_terms_when_HER_is_positive(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_mm_per_day = np.array([5.0])
		blackboard.Pherperc = np.array([0.2])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.Psmd = np.array([0.3])
		blackboard.her_array_mm_per_day = np.array([0.0])
		blackboard.macropore_att_mm_per_day = np.array([3.0])
		blackboard.macropore_dir_mm_per_day = np.array([2.0])
		blackboard.her_array_mm_per_day = np.array([10.0])

		actual = nitrate._calculate_p_non_her(blackboard)

		her_scaled_by_pherperc = 10.0 * 0.2
		her_scaled_by_psmd = 10.0 * 0.3
		runoff_and_macropore_minus_her = 5.0 + 3.0 + 2.0 - 10.0
		numerator = her_scaled_by_pherperc + runoff_and_macropore_minus_her + her_scaled_by_psmd
		denominator = 5.0 + 3.0 + 2.0
		expected_value = numerator / denominator
		expected = np.array([expected_value])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_p_non(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_recharge_mm_per_day = np.array([100.0, 0.0, 0.0, 100.0, 100.0, 0.0])
		blackboard.macropore_att_mm_per_day = np.array([0.0, 40.0, 40.0, 0.0, 0.0, 40.0])
		blackboard.macropore_dir_mm_per_day = np.array([0.0, 60.0, 60.0, 0.0, 0.0, 60.0])
		blackboard.her_array_mm_per_day = np.array([10.0, 20.0, 0.0, -10.0, 10.0, 20.0])
		blackboard.p_non_her = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.4])
		actual = nitrate._calculate_p_non(blackboard)
		expected = np.array([10.0, 5.0, 0.0, 0.0, 8.0, 3.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_m2_array_kg_per_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.runoff_recharge_mm_per_day = np.array([100.0, 0.0, 0.0])
		blackboard.macropore_att_mm_per_day = np.array([0.0, 40.0, 40.0])
		blackboard.macropore_dir_mm_per_day = np.array([0.0, 60.0, 60.0])
		blackboard.her_array_mm_per_day = np.array([10.0, 20.0, 0.0])
		blackboard.m0_array_kg_per_day = np.array([50.0, 60.0, 60.0])
		blackboard.p_non_her = np.array([0.0, 0.0, 0.0])
		blackboard.p_non = nitrate._calculate_p_non(blackboard)
		actual = nitrate._calculate_m2_array_kg_per_day(blackboard)
		expected = np.array([500.0, 300.0, 0.0])

		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_mi_array_kg_per_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m1a_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		blackboard.m2_array_kg_per_day = np.array([40.0, 50.0, 60.0])

		actual = nitrate._calculate_mi_array_kg_per_day(blackboard)
		expected = np.array([140.0, 250.0, 360.0])

		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_convert_kg_to_tons_array(self):
		np.testing.assert_array_almost_equal(np.array([]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([])))
		np.testing.assert_array_almost_equal(np.array([1.0]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([1000.0])))
		np.testing.assert_array_almost_equal(np.array([0.5, 1.0, 3.0]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([500, 1000.0, 3000.0])))

	def make_blackboard_with_kg(self, kg):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.nitrate_reaching_water_table_array_from_this_run_kg_per_day = np.array(kg)
		return blackboard

	def test_convert_kg_to_tons_array(self):
		np.testing.assert_array_almost_equal(np.array([]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([])))
		np.testing.assert_array_almost_equal(np.array([1.0]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([1000.0])))
		np.testing.assert_array_almost_equal(np.array([0.5, 1.0, 3.0]), nitrate._convert_kg_to_tons_array(self.make_blackboard_with_kg([500, 1000.0, 3000.0])))

	def test__combine_nitrate_reaching_water_table_array_from_this_run_and_historical_run_tons_per_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.historical_nitrate_reaching_water_table_array_tons_per_day = np.array([10.0, 20.0, 30.0])
		blackboard.nitrate_reaching_water_table_array_from_this_run_tons_per_day = np.array([1.0, 2.0, 3.0])

		actual = nitrate._combine_nitrate_reaching_water_table_array_from_this_run_and_historical_run_tons_per_day(blackboard)

		expected = np.array([11.0, 22.0, 33.0])
		np.testing.assert_array_almost_equal(expected, actual)

	def make_data_output_and_node(self):
		max_load_per_year_kg_per_hectare = 1000
		her_at_5_percent = 10
		her_at_50_percent = 100
		her_at_95_percent = 190

		data = {
			"time_switcher": timer.make_time_switcher(),
			"params": {
				"node_areas": {7: 2500.0},
				"nitrate_calibration_a": 1.38,
				"nitrate_calibration_mu": {7: [1.58]},
				"nitrate_calibration_sigma": 3.96,
				"nitrate_calibration_alpha": 1.7,
				"nitrate_calibration_effective_porosity": {7: [1.0 / 0.0029]},
				"nitrate_depth_to_water": {7: [0.00205411]},
				"nitrate_loading": {7: [0, 0, 0, max_load_per_year_kg_per_hectare, her_at_5_percent, her_at_50_percent, her_at_95_percent]},
				"nitrate_process": "enabled",
			}, "series" : {
				"date": [date(2023, 1, 1), date(2023, 1, 2), ]
			},
		}
		output = {
			"rainfall_ts": np.array([130.0, 130.0]),
			"ae": np.array([50.0, 50.0]),
			"perc_through_root": np.array([40.0, 40.0]),
			"interflow_volume": np.array([2.0, 20.0]),
			"infiltration_recharge": np.array([1.0, 9.0]),
			"interflow_to_rivers": np.array([2.0, 6.0]),
			"runoff_recharge": np.array([8.0, 8.0]),
			"macropore_att": np.array([4.0, 4.0]),
			"macropore_dir": np.array([4.0, 4.0]),
			"rapid_runoff": np.array([32.0, 32.0]),
			"combined_recharge": np.array([5, 5]),
			"smd" : np.array([20.0, -4.0]),
			"p_smd" : np.array([-4.0, 0.0]),
			"tawtew": np.array([0.0, 6.0]),
			"historical_nitrate_reaching_water_table_array_tons_per_day": np.array([100.0, 200.0])
		}
		node = 7
		return data, output, node

	def test_calculate_nitrate(self):
		data, output, node = self.make_data_output_and_node()
		actual = nitrate.calculate_nitrate(data, output, node, logging = DummyLogger())
		np.testing.assert_array_almost_equal(np.array([80.0, 80.0]), actual["her_array_mm_per_day"])
		np.testing.assert_array_almost_equal(np.array([100.0, 100.0]), actual["m0_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([75.0, 43.478261]), actual["m1_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([15.0, 18.89441]), actual["m1a_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([10.0, 20.0]), actual["m2_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([15.0, 30.0]), actual["m3_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 6.521739]), actual["m4_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([25.0, 38.89441]), actual["mi_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.6]), actual["proportion_reaching_water_table_array_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 15.0]), actual["nitrate_reaching_water_table_array_from_this_run_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([100.0, 200.015]), actual["nitrate_reaching_water_table_array_tons_per_day"])

	def test_get_nitrate(self):
		data, output, node = self.make_data_output_and_node()
		actual = nitrate.get_nitrate(data, output, node)
		np.testing.assert_array_almost_equal(np.array([25.0, 38.89441]), actual["mi_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([100.0, 200.015]), actual["nitrate_reaching_water_table_array_tons_per_day"])

	def test_calculate_nitrate_when_disabled(self):
		max_load_per_year_kg_per_hectare = 1000
		her_at_5_percent = 10
		her_at_50_percent = 100
		her_at_95_percent = 190

		data = {
			"time_switcher": timer.make_time_switcher(),
			"params": {
				"node_areas": {7: 2500.0},
				"nitrate_depth_to_water": {7: [0.00205411]},
				"nitrate_loading": {7: [0, 0, 0, max_load_per_year_kg_per_hectare, her_at_5_percent, her_at_50_percent, her_at_95_percent]},
				"nitrate_process": "disabled",
			}, "series" : {
				"date": [date(2023, 1, 1), date(2023, 1, 2), ]
			},
		}
		output = {
			"rainfall_ts": np.array([130.0, 130.0]),
			"ae": np.array([50.0, 50.0]),
			"perc_through_root": np.array([40.0, 40.0]),
			"interflow_volume": np.array([1.0, 1.0]),
			"infiltration_recharge": np.array([2.0, 2.0]),
			"interflow_to_rivers": np.array([2.0, 2.0]),
			"runoff_recharge": np.array([8.0, 8.0]),
			"macropore_att": np.array([4.0, 4.0]),
			"macropore_dir": np.array([4.0, 4.0]),
			"rapid_runoff": np.array([32.0, 32.0]),
			"combined_recharge": np.array([5, 5]),
			"smd" : np.array([30.0, 6.0]),
			"p_smd" : np.array([30.0, 6.0]),
			"tawtew": np.array([0.0, 6.0]),
		}
		node = 7
		actual = nitrate.calculate_nitrate(data, output, node)
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["mi_array_kg_per_day"])
		np.testing.assert_array_almost_equal(np.array([0.0, 0.0]), actual["nitrate_reaching_water_table_array_tons_per_day"])

	def test_output_file_path(self):
		expected_filename = "aardvark_nitrate.csv"
		make_filename_function = nitrate.make_output_filename
		self.assert_output_file_path(expected_filename, make_filename_function)

	def test_mi_output_file_path(self):
		expected_filename = "aardvark_mi.csv"
		make_filename_function = nitrate.make_mi_output_filename
		self.assert_output_file_path(expected_filename, make_filename_function)

	def test_nitrate_surface_flow_output_file_path(self):
		expected_filename = "aardvark_stream_nitrate.csv"
		make_filename_function = nitrate.make_nitrate_surface_flow_filename
		self.assert_output_file_path(expected_filename, make_filename_function)
	
	def test_convert_mm_to_m(self):
		np.testing.assert_array_almost_equal(
			np.array([0.0, 1.0, 0.001]),
			nitrate._convert_mm_to_m(np.array([0.0, 1000.0, 1.0])))

	def assert_output_file_path(self, expected_filename, make_filename_function):
		data = {
			"params" : {
				"run_name" : "aardvark",
			}
		}
		expected = ""
		actual = make_filename_function(data)
		sep = os.path.sep
		expected = "output_files" + sep + expected_filename
		isPassed = actual.endswith(expected)
		message = f"Expected '{actual}' to end with '{expected}'."
		self.assertTrue(isPassed, message)

def calculate_total_mass_leached_for_test(days, her_per_day):
		max_load_per_year = 10000 * 365.25
		her_at_5_percent = 5 * 365.25
		her_at_50_percent = 50 * 365.25
		her_at_95_percent = 95 * 365.25

		return m._calculate_total_mass_leached_from_cell_on_days(
			max_load_per_year,
			her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			days,
			her_per_day)

class DummyLogger:
	def warning(self, str):
		pass
