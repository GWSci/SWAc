import unittest
import numpy as np
import snow_melt

class SnowMeltParams:
	def __init__(self, length):
		self.Date = np.full(length, "1900-01-01")
		self.precip_mm = np.zeros(length)
		self.Tmax_C = np.zeros(length)
		self.Tmin_C = np.zeros(length)
		self.lat_deg = 0
		self.slope = 0
		self.aspect = 0
		self.tempHt = np.ones(length)
		self.windHt = np.ones(length)
		self.groundAlbedo = 0.25
		self.SurfEmissiv = 0.95
		self.windSp = np.ones(length) # Must be non-zero to prevent division by zero.
		self.forest = 0
		self.startingSnowDepth_m = 0
		self.startingSnowDensity_kg_m3 = 450

	def apply(self):
		params = self
		sm = snow_melt.SnowMelt()
		result = sm.SnowMelt(params.Date, params.precip_mm, params.Tmax_C, params.Tmin_C, params.lat_deg, params.slope, params.aspect, params.tempHt, params.windHt, params.groundAlbedo, params.SurfEmissiv, params.windSp, params.forest, params.startingSnowDepth_m, params.startingSnowDensity_kg_m3)
		return sm, result

class SnowMeltTest(unittest.TestCase):
	def setUp(self):
		self.params = None
		self.sm = None

	def test_converted_inputs_NewSnowDensity(self):
		self.params = SnowMeltParams(4)
		self.params.Tmax_C = np.array([-20, -16, -15, 10])
		self.params.Tmin_C = np.array([-20, -16, -15, 10])
		self.assert_attribute_equals([50.0, 50.0, 50.0, 135.0], "NewSnowDensity")

	def test_converted_inputs_NewSnowWatEq(self):
		self.params = SnowMeltParams(3)
		self.params.precip_mm = np.array([1000, 2000, 3000])
		self.params.Tmax_C = np.array([-5, 2, 10])
		self.params.Tmin_C = np.array([-6, -2, 8])
		self.assert_result_equals([1000.0, 0.0, 0.0], "NewSnowWatEq")

	def test_converted_inputs_NewSnow(self):
		self.params = SnowMeltParams(3)
		self.params.precip_mm = np.array([1000, 2000, 3000])
		self.params.Tmax_C = np.array([-5, 2, 10])
		self.params.Tmin_C = np.array([-6, -2, 8])
		self.assert_result_equals([12.1506682867558, 0, 0], "NewSnow")

	def test_converted_inputs_JDay(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-04-25"])
		self.assert_attribute_equals([1, 115], "JDay")

	def test_converted_inputs_lat_when_lat_deg_is_one_number(self):
		self.params = SnowMeltParams(2)
		self.params.lat_deg = 60
		self.assert_attribute_equals(1.0471975511966, "lat")

	def test_converted_inputs_rh(self):
		self.params = SnowMeltParams(2)
		self.params.windHt = np.array([100, 80])
		self.params.tempHt = np.array([20, 15])
		self.params.windSp = np.array([50, 200])
		self.assert_attribute_equals([0.000182524295254448, 4.3628558519474e-05], "rh")

	def test_converted_inputs_rh_is_expanded_to_list(self):
		self.params = SnowMeltParams(3)
		self.params.precip_mm = np.array([1000, 2000, 3000])
		self.params.windHt = 100
		self.params.tempHt = 20
		self.params.windSp = 50
		self.assert_attribute_equals([0.000182524295254448, 0.000182524295254448, 0.000182524295254448], "rh")

	def test_converted_inputs_atmospheric_emissivity(self):
		self.params = SnowMeltParams(2)
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.assert_attribute_equals([0.922614101280085, 0.833213567838195], "AE")

	def test_new_variables(self):
		self.params = SnowMeltParams(2)
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.assert_result_equals([False, False], "SnowDepth")

	def test_initial_values(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_result_equals(0, "SnowDepth")

	def test_values_after_loop_albedo_case_1(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-03"])
		self.params.precip_mm = np.array([1000, 3])
		self.params.Tmax_C = np.array([25, -5])
		self.params.Tmin_C = np.array([15, -10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_attribute_equals([0.44, 0.869814515763774], "Albedo")
		self.assert_attribute_equals([450, 75.5], "SnowDensity")
		self.assert_result_equals([700, 0], "SnowMelt")
		self.assert_result_equals([0, 0.0397350993377483], "SnowDepth")

	def test_values_after_loop_albedo_case_2(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_attribute_equals([0.44, 0.38], "Albedo")
		self.assert_result_equals([0, 0], "SnowWaterEq")
		self.assert_attribute_equals([0, 0], "DCoef")
		self.assert_result_equals([700, 0], "SnowMelt")
		self.assert_attribute_equals([0, 0], "SnowTemp")
		self.assert_attribute_equals([342820.472334099, 417255.629678636], "Energy")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([0, 0], "SnowDepth")
		self.assert_result_equals([0, 0], "SnowWaterEq")

	def test_values_after_loop_albedo_case_2_zerosnowMelt(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([-5, -5])
		self.params.Tmin_C = np.array([-15, -10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = -1
		self.assert_result_equals([0, 0], "SnowMelt")
		self.assert_attribute_equals([0, -0.020361747890789], "SnowTemp")
		self.assert_attribute_equals([39.4962486602358, 70.4081905073295], "SnowDensity")
		self.assert_result_equals([22.78697421981, 41.188389860667], "SnowDepth")
		self.assert_result_equals([900, 2900], "SnowWaterEq")

	def test_values_after_loop_albedo_case_2_SnowWaterDepthNonZero(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([-1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_result_equals([175.486131610864, 0], "SnowWaterEq")
		self.assert_attribute_equals([0, 6.2], "DCoef")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([524.513868389136, 175.486131610864], "SnowMelt")
		self.assert_result_equals([0.389969181357476, 0], "SnowDepth")

	def test_values_after_loop_albedo_case_2_subZero(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([-5, -5])
		self.params.Tmin_C = np.array([-15, -10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_attribute_equals([0.98, 0.98], "Albedo")
		self.assert_attribute_equals([-124548.590051023, -92000.1094066136], "Energy")
		self.assert_attribute_equals([189.27842069435, 108.328167789113], "SnowDensity")
		self.assert_result_equals([0, 0], "SnowMelt")
		self.assert_result_equals([8.98147815141162, 34.1554747533712], "SnowDepth")

	def test_values_after_loop_ground_albedo_case_2_capped(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.55
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_attribute_equals([0.55, 0.55], "Albedo")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([700, 0], "SnowMelt")
		self.assert_result_equals([0, 0], "SnowDepth")
		self.assert_result_equals([0, 0], "SnowWaterEq")

	def test_values_after_loop_albedo_case_3(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-03"])
		self.params.precip_mm = np.array([1000, 0])
		self.params.Tmax_C = np.array([-5, -5])
		self.params.Tmin_C = np.array([-5, -10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_attribute_equals([0.98, 0.751349585244717], "Albedo")
		self.assert_attribute_equals([219.521410579345, 233.55361989076], "SnowDensity")
		self.assert_result_equals([0, 0], "SnowMelt")
		self.assert_result_equals([7.74411933448078, 7.27884243795981], "SnowDepth")
		self.assert_result_equals([1700, 1700], "SnowWaterEq")

	def test_initial_SnowWaterEq_and_SnowDepth_non_zero(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([-1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.assert_result_equals([175.486131610864, 0], "SnowWaterEq")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([524.5138684, 175.4861316], "SnowMelt")
		self.assert_result_equals([0.389969181357476, 0], "SnowDepth")

	def test_initial_albedo_when_starting_snow_depth_is_zero(self):
		self.params = SnowMeltParams(2)
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.startingSnowDepth_m = 0
		self.assert_first_array_value_equals(0.25, "Albedo")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([0, 0], "SnowMelt")
		self.assert_result_equals([0, 0], "SnowDepth")
		self.assert_result_equals([0, 0], "SnowWaterEq")

	def test_initial_albedo_when_new_snow_is_greater_than_zero(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["1900-01-01", "1900-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([-5, 35])
		self.params.Tmin_C = np.array([-5, 10])
		self.params.startingSnowDepth_m = 7
		self.assert_first_array_value_equals(0.98, "Albedo")
		self.assert_attribute_equals([219.521410579345, 247.585829202175], "SnowDensity")
		self.assert_result_equals([0, 612.986782033192], "SnowMelt")
		self.assert_result_equals([7.74411933448078, 4.39045005713622], "SnowDepth")

	def test_initial_snow_density_when_not_450(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 7
		self.params.startingSnowDensity_kg_m3 = 200
		self.assert_first_array_value_equals(200, "SnowDensity")
		self.assert_attribute_equals([200, 450], "SnowDensity")
		self.assert_result_equals([700, 0], "SnowMelt")
		self.assert_result_equals([0, 0], "SnowDepth")
		self.assert_result_equals([0, 0], "SnowWaterEq")

	def test_initial_snow_melt_using_third_term(self):
		self.params = SnowMeltParams(2)
		self.params.Date = np.array(["2018-01-01", "2018-01-02"])
		self.params.precip_mm = np.array([1000, 2000])
		self.params.Tmax_C = np.array([25, 35])
		self.params.Tmin_C = np.array([15, 10])
		self.params.lat_deg = 60
		self.params.tempHt = 20
		self.params.windHt = 100
		self.params.groundAlbedo = 0.25
		self.params.SurfEmissiv = 0.95
		self.params.windSp = 50
		self.params.startingSnowDepth_m = 1000000
		self.params.startingSnowDensity_kg_m3 = 450
		self.assert_result_equals([1028.56427342964, 1250.2552784], "SnowMelt")
		self.assert_attribute_equals([450, 450], "SnowDensity")
		self.assert_result_equals([222219.936523837, 222217.158178774], "SnowDepth")
 
	def assert_attribute_equals(self, expected, name):
		self.sm, x = self.params.apply()
		actual = getattr(self.sm, name)
		np.testing.assert_almost_equal(expected, actual)

	def assert_result_equals(self, expected, name):
		index = ["Date", "Tmax_C", "Tmin_C", "precip_mm", "R_m", "NewSnowWatEq", "SnowMelt", "NewSnow", "SnowDepth", "SnowWaterEq"].index(name)
		self.sm, result = self.params.apply()
		actual = result[index]
		np.testing.assert_almost_equal(expected, actual)

	def assert_first_array_value_equals(self, expected, name):
		self.sm, x = self.params.apply()
		actual = getattr(self.sm, name)[0]
		np.testing.assert_almost_equal(expected, actual)

