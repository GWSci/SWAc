import unittest
import numpy as np
import swacmod.snow_melt as snow_melt

class SolarTest(unittest.TestCase):
	def test_declination(self):
		JDay = np.array([0, 30, 90, 100, 180, 260, 270, 350, 360, 365])
		actual = snow_melt.declination(JDay)
		expected = [
			-0.403968140285608, -0.314231430567405, 0.0712304824789748, 
			0.140296662792189, 0.403968140285608, 5.02350117010244e-17, 
			-0.0712304824789749, -0.4102, -0.403968140285608, -0.396222773943776
			]
		np.testing.assert_almost_equal(expected, actual)

	def test_PotentialSolar(self):
		lat = np.array([0, 0.7, 0.3, 0.1, -0.7, -1.2, -0.1, -0.5, 0.3, -0.3])
		JDay = np.array([0, 30, 90, 100, 180, 260, 270, 350, 360, 365])
		actual = snow_melt.PotentialSolar(lat, JDay)
		expected = [
			34390.9151501073, 16533.0848573286, 36884.6523269427, 
			37672.7831432051, 13151.0128676745, 13552.6915312709, 
			37538.5709912887, 42185.8525142677, 26318.1567946698, 
			39939.6487935873
			]
		np.testing.assert_almost_equal(expected, actual)

	def test_solarangle(self):
		lat = np.array([0, 0.7, 0.3, 0.1, -0.7, -1.2, -0.1, -0.5, 0.3, -0.3])
		JDay = np.array([0, 30, 90, 100, 180, 260, 270, 350, 360, 365])
		actual = snow_melt.solarangle(lat, JDay)
		expected = [
			1.16682818650929, 0.556564896227492, 1.34202680927387, 
			1.53049966400271, 0.466828186509289, 0.370796326794897, 
			1.54202680927387, 1.4809963267949, 0.866828186509289, 
			1.47457355285112
			]
		np.testing.assert_almost_equal(expected, actual)

	def test_slopefactor(self):
		lat = np.array([0, 0.7, 0.3, 0.1, -0.7, -1.2, -0.1, -0.5, 0.3, -0.3])
		Jday = np.array([0, 30, 90, 100, 180, 260, 270, 350, 360, 365])
		slope = np.array([0, 180, 90, 45, 135, 30, 125, 270, 200, 300]) * np.pi / 180
		aspect = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180]) * np.pi / 180
		actual = snow_melt.slopefactor(lat, Jday, slope, aspect)
		expected = [1, 0, 0, 0.721361519496336, 0, 0.642700682851802, 0, 0.0689762999591603, 0, 0.417247]
		np.testing.assert_almost_equal(expected, actual)

	def test_slopefactor_with_scalars(self):
		lat = 0.1
		Jday = 100
		slope = 45 * np.pi / 180
		aspect = 60 * np.pi / 180
		actual = snow_melt.slopefactor(lat, Jday, slope, aspect)
		expected = [0.721361519496336]
		np.testing.assert_almost_equal(expected, actual)

	def test_solar(self):
		lat = 0.5
		Jday = np.array([10, 20, 30])
		Tx = np.array([15, 20, 25])
		Tn = np.array([10, 19, -2])
		albedo = 0.2
		forest = 0
		slope = 0
		aspect = 0
		printWarn = False
		actual = snow_melt.Solar(lat,Jday,Tx,Tn,albedo,forest,slope,aspect,printWarn)
		expected = [3334.6, 85.506, 13802]
		np.testing.assert_almost_equal(expected, actual)

	def test_solar_with_warnings(self):
		lat = 60
		Jday = np.array([10, 20, 30])
		Tx = np.array([15, 20, 25])
		Tn = np.array([10, 19, -2])
		albedo = 0.2
		forest = 0
		slope = 0
		aspect = 0
		printWarn = True
		actual = snow_melt.Solar(lat,Jday,Tx,Tn,albedo,forest,slope,aspect,printWarn)
		expected = [435.72, 14.242, 2975.7]
		np.testing.assert_almost_equal(expected, actual)

	def test_solar_with_degrees_but_no_warnings(self):
		lat = 60
		Jday = np.array([10, 20, 30])
		Tx = np.array([15, 20, 25])
		Tn = np.array([10, 19, -2])
		albedo = 0.2
		forest = 0
		slope = 0
		aspect = 0
		printWarn = False
		actual = snow_melt.Solar(lat,Jday,Tx,Tn,albedo,forest,slope,aspect,printWarn)
		expected = [-4303.6, -108.43, -17115]
		np.testing.assert_almost_equal(expected, actual)


