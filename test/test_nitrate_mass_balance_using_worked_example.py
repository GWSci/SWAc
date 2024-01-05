import unittest

class Test_Nitrate_Mass_Balance_Using_Worked_Example(unittest.TestCase):
	def setUp(self):
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

	def test_x(self):
		self.assertEqual(1, 1)
