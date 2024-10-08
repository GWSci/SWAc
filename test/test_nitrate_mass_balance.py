import unittest
import numpy as np
import swacmod.nitrate as nitrate

class Test_Nitrate_Mass_Balance(unittest.TestCase):
	def test_calculate_Pro(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.her_array_mm_per_day = np.array([1.0, 1.0, 1.0,     1.0, 0.0, -1.0, 1.0])
		blackboard.p_non = np.array(               [0.0, 0.0, 2.0/3.0, 0.2, 0.2, 0.2, 0.0])
		blackboard.Psmd = np.array(                [0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.8])
		blackboard.Pherperc = np.array(            [0.0, 0.5, 0.0,     0.3, 0.3, 0.3, 0.0])
		expected = np.array(            [1.0, 0.5, 1.0/3.0, 0.5, 0.0, 0.0, 0.2 ])
		actual = nitrate._calculate_Pro(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_m3_array_kg_per_day(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m0_array_kg_per_day = np.array([2.0, 6.0, 15.0, 14.0, 14.0, 14.0, 55.0])
		blackboard.Pro = np.array([1.0, 0.5, 1.0/3.0, 0.5, 0.0, 0.0, 0.2 ])
		expected = np.array([2.0, 3.0, 5.0, 7.0, 0.0, 0.0, 11.0])
		actual = nitrate._calculate_m3_array_kg_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)
	
	def test_calculate_dSMD_array_mm_per_day_for_zero_days(self):
		self.assert_dSMD_array_mm_per_day([], [], [])

	def test_calculate_dSMD_array_mm_per_day_for_one_day(self):
		self.assert_dSMD_array_mm_per_day([7.0], [7.0], [0.0])
		self.assert_dSMD_array_mm_per_day([4.0], [7.0], [3.0])
		self.assert_dSMD_array_mm_per_day([7.0], [7.0], [-3.0])

	def test_calculate_dSMD_array_mm_per_day_for_three_days(self):
		self.assert_dSMD_array_mm_per_day([90.0, 9.0, 1.0], [100.0, 10.0, 1.0], [10.0, 1.0, 0.0])
		self.assert_dSMD_array_mm_per_day([7.0, 100.0, 1000.0], [10.0, 100.0, 1000.0], [3.0, 0.0, -20.0])

	def assert_dSMD_array_mm_per_day(self, expected, input_smd, input_potential_smd):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.smd = np.array(input_smd)
		blackboard.p_smd = np.array(input_potential_smd)
		actual = nitrate._calculate_dSMD_array_mm_per_day(blackboard)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_Psoilperc_for_zero_days(self):
		input_perc_through_root_mm_per_day = np.array([])
		input_TAW_array_mm = np.array([])
		expected = []
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_greater_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([2.0, 30.0, 500.0])
		input_TAW_array_mm = np.array([8.0, 70.0, 500.0])
		expected = [0.2, 0.3, 0.5]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0])
		input_TAW_array_mm = np.array([1.0])
		expected = [0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_denominator_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0, 2.0])
		input_TAW_array_mm = np.array([0.0, -2.0])
		expected = [0.0, 0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def test_calculate_Psoilperc_when_percolation_less_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0])
		input_TAW_array_mm = np.array([2.0])
		expected = [0.0]
		self.assert_Psoilperc(expected, input_perc_through_root_mm_per_day, input_TAW_array_mm)

	def assert_Psoilperc(self, expected, input_perc_through_root_mm_per_day, input_TAW_array_mm):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.perc_through_root_mm_per_day = input_perc_through_root_mm_per_day
		blackboard.TAW_array_mm = input_TAW_array_mm
		actual = nitrate._calculate_Psoilperc(blackboard)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_Pherperc_for_zero_days(self):
		input_perc_through_root_mm_per_day = np.array([])
		input_her_array_mm_per_day = np.array([])
		expected = []
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_greater_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([14.0, 33.0, 65.0])
		input_her_array_mm_per_day = np.array([2.0, 3.0, 5.0])
		expected = [7.0, 11.0, 13.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_equal_to_zero(self):
		input_perc_through_root_mm_per_day = np.array([0.0])
		input_her_array_mm_per_day = np.array([2.0])
		expected = [0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_percolation_less_than_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0])
		input_her_array_mm_per_day = np.array([2.0])
		expected = [0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def test_calculate_Pherperc_when_HER_equals_zero(self):
		input_perc_through_root_mm_per_day = np.array([-1.0, 0.0, 1.0])
		input_her_array_mm_per_day = np.array([0.0, 0.0, 0.0])
		expected = [0.0, 0.0, 0.0]
		self.assert_Pherperc(expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day)

	def assert_Pherperc(self, expected, input_perc_through_root_mm_per_day, input_her_array_mm_per_day):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.perc_through_root_mm_per_day = input_perc_through_root_mm_per_day
		blackboard.her_array_mm_per_day = input_her_array_mm_per_day
		actual = nitrate._calculate_Pherperc(blackboard)
		expected_numpy = np.array(expected)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_M_soil_in_kg(self):
		expected = np.array([0.0, 2.0, 1.0, 1.0, 9.0])
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m0_array_kg_per_day = np.array([0.0, 2.0, 2.0, 2.0, 12.0])
		blackboard.Psmd = np.array([0.0, 1.0, 0.5, 0.0, 0.25])
		blackboard.Pherperc = Pherperc = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
		actual = nitrate._calculate_M_soil_in_kg(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_M_soil_tot_kg_for_zero_days(self):
		input_Msoil_in = []
		input_Psoilperc = []
		expected = []
		self.assert_M_soil_tot_kg_for_zero_days(expected, input_Msoil_in, input_Psoilperc)

	def test_M_soil_tot_kg_for_one_days(self):
		input_Msoil_in = [6.0]
		input_Psoilperc = [0.5]
		expected = [3.0]
		self.assert_M_soil_tot_kg_for_zero_days(expected, input_Msoil_in, input_Psoilperc)

	def test_M_soil_tot_kg_for_two_days(self):
		input_Msoil_in = [6.0, 9.0]
		input_Psoilperc = [0.5, 2.0/3.0]
		expected = [3.0, 4.0]
		self.assert_M_soil_tot_kg_for_zero_days(expected, input_Msoil_in, input_Psoilperc)

	def assert_M_soil_tot_kg_for_zero_days(self, expected, input_Msoil_in, input_Psoilperc):
		expected_numpy = np.array(expected)
		blackboard = nitrate.NitrateBlackboard()
		blackboard.M_soil_in_kg = np.array(input_Msoil_in)
		blackboard.Psoilperc = np.array(input_Psoilperc)
		actual = nitrate._calculate_M_soil_tot_kg(blackboard)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_calculate_M4_array_mm_per_day_for_zero_days(self):
		input_M_soil_tot_kg = []
		input_m1_array_kg_per_day = []
		expected = []
		self.assert_M4_array_mm_per_day(expected, input_M_soil_tot_kg, input_m1_array_kg_per_day)

	def test_calculate_M4_array_mm_per_day_for_one_day(self):
		input_M_soil_tot_kg = [11.0]
		input_m1_array_kg_per_day = [3.0]
		expected = [8.0]
		self.assert_M4_array_mm_per_day(expected, input_M_soil_tot_kg, input_m1_array_kg_per_day)

	def test_calculate_M4_array_mm_per_day_for_three_days(self):
		input_M_soil_tot_kg = [11.0, 20.0]
		input_m1_array_kg_per_day = [3.0, 15.0]
		expected = [8.0, 5.0]
		self.assert_M4_array_mm_per_day(expected, input_M_soil_tot_kg, input_m1_array_kg_per_day)

	def assert_M4_array_mm_per_day(self, expected, input_M_soil_tot_kg, input_m1_array_kg_per_day):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.M_soil_in_kg = np.array(input_M_soil_tot_kg)
		blackboard.m1_array_kg_per_day = np.array(input_m1_array_kg_per_day)
		expected_numpy = np.array(expected)
		actual = nitrate._calculate_M4_array_mm_per_day(blackboard)
		np.testing.assert_array_almost_equal(expected_numpy, actual)

	def test_is_mass_balanced_for_empty_arrays(self):
		self.assert_masses_balanced([])
		self.assert_masses_balanced([0.0])
		self.assert_masses_not_balanced([1.234 - 5.678])
		self.assert_masses_balanced([0.0, 0.0, 0.0])
		self.assert_masses_not_balanced([0.0, 5.678 - 7.890, 0.0])
		self.assert_masses_balanced([-1e-8])
		self.assert_masses_balanced([1e-8])
		self.assert_masses_not_balanced([1e-7])
		self.assert_masses_not_balanced([-1e-7])

	def assert_masses_balanced(self, error = None):
		error_np = np.array(error)
		actual = nitrate._is_mass_balanced(error_np)
		self.assertTrue(actual)

	def assert_masses_not_balanced(self, error = None):
		error_np = np.array(error)
		actual = nitrate._is_mass_balanced(error_np)
		self.assertFalse(actual)

	def test_find_unbalanced_day_to_report(self):
		self.assert_unbalanced_day_to_report(0, [-1.0])
		self.assert_unbalanced_day_to_report(0, [5.0, 1.0])
		self.assert_unbalanced_day_to_report(1, [1.0, 5.0])
		self.assert_unbalanced_day_to_report(0, [-5.0, -1.0])
		self.assert_unbalanced_day_to_report(1, [-1.0, -5.0])
	
	def assert_unbalanced_day_to_report(self, expected, mass_balance_error_kg):
		mass_balance_error_kg_np = np.array(mass_balance_error_kg)
		actual = nitrate._find_unbalanced_day_to_report(mass_balance_error_kg_np)
		self.assertEqual(expected, actual)

	def test_total_NO3_to_receptors_kg(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m1_array_kg_per_day = np.array([1.0, 2.0, 3.0, 4.0])
		blackboard.m2_array_kg_per_day = np.array([20.0, 30.0, 40.0, 50.0])
		blackboard.m3_array_kg_per_day = np.array([100.0, 300.0, 500.0, 700.0])
		blackboard.m4_array_kg_per_day = np.array([9000.0, 8000.0, 7000.0, 6000.0])
		expected = np.array([9121.0, 8332.0, 7543.0, 6754.0])
		actual = nitrate._calculate_total_NO3_to_receptors_kg(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_calculate_mass_balance_error_kg(self):
		blackboard = nitrate.NitrateBlackboard()
		blackboard.m0_array_kg_per_day = np.array([10.0, 20.0, 30.0])
		blackboard.total_NO3_to_receptors_kg = np.array([3.0, 20.0, 32.0])
		expected = np.array([7.0, 0.0, -2.0])
		actual = nitrate._calculate_mass_balance_error_kg(blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_make_unbalanced_day_log_message(self):
		i = 1

		blackboard = nitrate.NitrateBlackboard()
		blackboard.node = 1
		blackboard.m0_array_kg_per_day = np.array([1.0, 2.0, 3.0])
		blackboard.m1_array_kg_per_day = np.array([10.0, 20.0, 30.0])
		blackboard.m2_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		blackboard.m3_array_kg_per_day = np.array([1000.0, 2000.0, 3000.0])
		blackboard.m4_array_kg_per_day = np.array([10000.0, 20000.0, 30000.0])
		blackboard.total_NO3_to_receptors_kg = np.array([123.0, 456.0, 789.0])
		blackboard.mass_balance_error_kg = np.array([1.0, 2.0, 1.0])

		actual = nitrate._make_unbalanced_day_log_message(i, blackboard)

		expected = "Nitrate masses do not balance for node 1 using the equation M0 = M1 + M2 + M3 + M4. The day with the largest mass balance error is at index 1 with a mass balance error of 2.0 kg. total_NO3_to_receptors = 456.0 kg; M0 = 2.0 kg; M1 = 20.0 kg; M2 = 200.0 kg; M3 = 2000.0 kg; M4 = 20000.0 kg."
		self.assertEqual(expected, actual)

	def test_check_masses_balance_when_they_are_unbalanced(self):
		logging = SpyLogger()
		
		blackboard = nitrate.NitrateBlackboard()
		blackboard.node = 1
		blackboard.m0_array_kg_per_day = np.array([1.0, 2.0, 3.0])
		blackboard.m1_array_kg_per_day = np.array([10.0, 20.0, 30.0])
		blackboard.m2_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		blackboard.m3_array_kg_per_day = np.array([1000.0, 2000.0, 3000.0])
		blackboard.m4_array_kg_per_day = np.array([10000.0, 20000.0, 30000.0])
		blackboard.total_NO3_to_receptors_kg = np.array([123.0, 456.0, 789.0])
		blackboard.mass_balance_error_kg = np.array([1.0, 2.0, 1.0])
		blackboard.logging = logging

		nitrate._check_masses_balance(blackboard)
		actual = logging.log

		expected = ["Nitrate masses do not balance for node 1 using the equation M0 = M1 + M2 + M3 + M4. The day with the largest mass balance error is at index 1 with a mass balance error of 2.0 kg. total_NO3_to_receptors = 456.0 kg; M0 = 2.0 kg; M1 = 20.0 kg; M2 = 200.0 kg; M3 = 2000.0 kg; M4 = 20000.0 kg."]
		self.assertEqual(expected, actual)

	def test_check_masses_balance_when_they_are_balanced(self):
		logging = SpyLogger()

		blackboard = nitrate.NitrateBlackboard()
		blackboard.node = 1
		blackboard.m0_array_kg_per_day = np.array([1.0, 2.0, 3.0])
		blackboard.m1_array_kg_per_day = np.array([10.0, 20.0, 30.0])
		blackboard.m2_array_kg_per_day = np.array([100.0, 200.0, 300.0])
		blackboard.m3_array_kg_per_day = np.array([1000.0, 2000.0, 3000.0])
		blackboard.m4_array_kg_per_day = np.array([10000.0, 20000.0, 30000.0])
		blackboard.total_NO3_to_receptors_kg = np.array([123.0, 456.0, 789.0])
		blackboard.mass_balance_error_kg = np.array([0.0, 0.0, 0.0])
		blackboard.logging = logging
		
		nitrate._check_masses_balance(blackboard)
		actual = logging.log

		expected = []
		self.assertEqual(expected, actual)

class SpyLogger():
	def __init__(self):
		self.log = []

	def warning(self, str):
		self.log.append(str)
