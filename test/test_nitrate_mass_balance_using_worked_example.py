import numpy as np
import swacmod.nitrate as nitrate
import unittest

class Test_Nitrate_Mass_Balance_Using_Worked_Example(unittest.TestCase):
	def setUp(self):
		self.set_values_from_spreadsheet()
		self.set_input_parameters()

	def set_values_from_spreadsheet(self):
		# Values taken from example spreadsheet.
		# Names are the same as the spreadsheet with the following changes:
		#  - spaces replaced with underscores
		#  - unitless units removed
		#  - slashes replaced with per
		#  - parentheses removed
		self.Precipitation = 3.216888218
		self.AE = 0.18
		self.Percolation_through_root_zone = 2.241398327
		self.Macropore_recharge = 0
		self.Runoff_recharge = 0
		self.Potential_SMD = -2.241398327
		self.SMD = 0.152112247
		self.TAW = 142.08
		self.Msoil_tot_initial_kg = 23.78707016
	
		self.HER_mm_per_d = 3.036888218
		self.HER_mm_per_yr = 1109.223422
		self.M0_potential_kg_per_ha_per_d = 0.184176809
		self.M0_year_to_date_kg_per_ha = 7.810939513
		self.M0_actual_kg_per_ha_per_d = 0.184176809
		self.M0_kg_per_cell = 0.736707238
		self.dSMD_mm = 0.152112247
		self.Psmd = 0.050088194
		self.Psoilperc = 0.015530603
		self.Pherperc = 0.738057566
		self.Msoil_in_kg = 0.5806327
		self.M1_kg = 0.378445116
		self.Pnon = 0
		self.M2_kg = 0
		self.Pro = 0.211854239
		self.M3_kg = 0.1560746
		self.Msoil_tot_kg = 23.9892577
		self.M4_kg = 0.2021876
		self.Total_NO3_to_receptors_kg = 0.7367072
		self.Mass_Balance_Error_kg = 0.0000000

	def set_input_parameters(self):
		self.m0_array_kg_per_day = np.array([self.M0_kg_per_cell])
		self.m1_array_kg_per_day = np.array([self.M1_kg])
		self.Msoil_in_kg_array = np.array([self.Msoil_in_kg])
		self.her_array_mm_per_day = np.array([self.HER_mm_per_d])
		self.Psmd_array = np.array([self.Psmd])
		self.Psoilperc_array = np.array([self.Psoilperc])
		self.Pherperc_array = np.array([self.Pherperc])
		self.dSMD_array_mm_per_day = np.array([self.dSMD_mm])
		self.p_non_array = np.array([self.Pnon])
		self.p_non_array = np.array([self.Pnon])
		self.Pro_array = np.array([self.Pro])

		self.blackboard = nitrate.NitrateBlackboard()
		self.blackboard.rainfall_ts = np.array([self.Precipitation])
		self.blackboard.ae = np.array([self.AE])
		self.blackboard.perc_through_root_mm_per_day = np.array([self.Percolation_through_root_zone])
		self.blackboard.her_array_mm_per_day = np.array([self.HER_mm_per_d])
		self.blackboard.TAW_array_mm = np.array([self.TAW])
		self.blackboard.smd = np.array([self.SMD])
		self.blackboard.p_smd = np.array([self.Potential_SMD])

	def test_worked_example_HER(self):
		expected = self.her_array_mm_per_day
		actual = nitrate._calculate_her_array_mm_per_day(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_dSMD(self):
		expected = self.dSMD_array_mm_per_day
		actual = nitrate._calculate_dSMD_array_mm_per_day(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Psmd(self):
		expected = [self.Psmd]
		self.blackboard.her_array_mm_per_day = self.her_array_mm_per_day
		self.blackboard.dSMD_array_mm_per_day = self.dSMD_array_mm_per_day
		actual = nitrate._calculate_Psmd(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Psoilperc(self):
		expected = [self.Psoilperc]
		actual = nitrate._calculate_Psoilperc(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Pherperc(self):
		expected = [self.Pherperc]
		actual = nitrate._calculate_Pherperc(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Msoil_in_kg(self):
		expected = [self.Msoil_in_kg]
		self.blackboard.m0_array_kg_per_day = self.m0_array_kg_per_day
		self.blackboard.Psmd = self.Psmd_array
		self.blackboard.Pherperc = self.Pherperc_array
		actual = nitrate._calculate_M_soil_in_kg(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Pnon(self):
		expected = [self.Pnon]
		self.blackboard.runoff_recharge_mm_per_day = np.array([self.Runoff_recharge])
		self.blackboard.macropore_att_mm_per_day = np.array([self.Macropore_recharge / 2.0])
		self.blackboard.macropore_dir_mm_per_day = np.array([self.Macropore_recharge / 2.0])
		actual = nitrate._calculate_p_non(self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_M2(self):
		expected = [self.M2_kg]
		self.blackboard.p_non = self.p_non_array
		self.blackboard.m0_array_kg_per_day = self.m0_array_kg_per_day
		actual = nitrate._calculate_m2_array_kg_per_day(self.m0_array_kg_per_day, self.p_non_array, self.blackboard)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_Pro(self):
		expected = self.Pro_array
		actual = nitrate._calculate_Pro(self.her_array_mm_per_day, self.p_non_array, self.Pherperc_array, self.Psmd_array)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_M3(self):
		expected = [self.M3_kg]
		actual = nitrate._calculate_m3_array_kg_per_day(self.m0_array_kg_per_day, self.Pro_array)
		np.testing.assert_array_almost_equal(expected, actual)

	def test_worked_example_M4(self):
		expected = [self.M4_kg]
		actual = nitrate._calculate_M4_array_mm_per_day(self.Msoil_in_kg_array, self.m1_array_kg_per_day)
		np.testing.assert_array_almost_equal(expected, actual)
