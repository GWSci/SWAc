import unittest
import numpy as np
import swacmod.snow_melt as snow_melt

class TransmissivityTest(unittest.TestCase):
	def test_transmissivity_for_zero(self):
		actual = self.transmissivity_for([0], [0])
		np.testing.assert_almost_equal(0, actual)

	def test_transmissivity_for_short_list(self):
		actual = self.transmissivity_for([10, 20, 30, 40], [1, 2, 4, 8])
		np.testing.assert_almost_equal([0.1753024, 0.5660054, 0.7248780, 0.7471995], actual)

	def test_transmissivity_for_exactly_30(self):
		t = [
			1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
			16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
			]
		t0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		actual = self.transmissivity_for(t, t0)
		expected = [
			0.002477334, 0.012983397, 0.033868674, 0.066036836, 0.109266607, 0.162323125,
			0.223110865, 0.288906173, 0.356657324, 0.423315258, 0.486147183, 0.542986769,
			0.592387201, 0.633663169, 0.666829382, 0.692461148, 0.711512658, 0.725129716,
			0.734486750, 0.740666413, 0.744587561, 0.746977121, 0.748375105, 0.749159944,
			0.749582574, 0.749800770, 0.749908724, 0.749959884, 0.749983097, 0.749993176
			]
		np.testing.assert_almost_equal(expected, actual)

	def test_transmissivity_for_more_than_30(self):
		Tx = [
			46.2588491, 80.6461228, 33.5890683, 69.9042314, 54.6926692, 
			97.4384195, 31.4749322, 12.0405097, 23.1082306, 1.1380555, 
			40.5300796, 41.4641429, 70.5162717, 99.8110393, 41.2326243, 
			36.5860489, 53.4723774, 60.9559414, 35.9943330, 93.6501645, 
			54.9905339, 63.7299080, 66.0398007, 71.2566516, 25.9549594, 
			99.2520931, 21.4034391, 40.2447548, 19.3154582, 33.7060243, 
			45.6838799, 46.9516682, 57.7073056, 91.3036710, 82.4169178, 
			90.9056868, 56.7100943, 92.0847119, 11.5346327, 95.3842562, 
			46.7856538, 0.7008730, 44.9744849, 41.6352529, 82.4571707, 
			75.8087031, 50.3661508, 23.5074744, 98.7072302, 30.7741084, 
			34.1617853, 10.4476219, 89.9145703, 67.6689764, 59.1578692, 
			67.2801530, 80.2606012, 15.7446927, 75.0941560, 73.4376241, 
			91.0548289, 59.9767058, 35.4500582, 64.1528030, 53.5566714, 
			9.7064029, 98.9482709, 19.4787032, 8.7683162, 47.9101952, 
			39.3767255, 51.7442199, 75.4133389, 57.4099936, 65.3060789, 
			54.6122941, 35.7266688, 87.4787808, 11.5124674, 85.3414383, 
			58.3367066, 77.2392933, 45.4198671, 53.2450882, 81.3933256, 
			59.5011619, 98.9657781, 34.3346119, 88.2492577, 0.9912411, 
			47.7996362, 30.1008731, 63.9080943, 37.1401706, 31.7511938, 
			81.0564829, 9.4335429, 60.2436814, 6.3428760, 89.6784725
			]
		Tn = [0] * 100
		actual = self.transmissivity_for(Tx, Tn)
		expected = [
			1.016370e-01, 3.184937e-01, 4.899040e-02, 2.433514e-01, 1.467045e-01,
			4.359085e-01, 4.211612e-02, 4.306454e-03, 2.036563e-02, 1.501792e-05,
			7.545140e-02, 7.945940e-02, 2.475357e-01, 4.517449e-01, 7.845629e-02,
			5.985835e-02, 1.634706e-01, 1.929738e-01, 5.438313e-02, 3.576360e-01,
			1.277270e-01, 1.562835e-01, 1.164655e-01, 1.450571e-01, 8.751211e-03,
			1.855487e-01, 6.586621e-03, 3.355517e-02, 7.905463e-03, 2.407584e-02,
			4.038087e-02, 4.371389e-02, 8.441174e-02, 1.718655e-01, 1.837853e-01,
			2.451436e-01, 1.157686e-01, 2.834706e-01, 2.474959e-03, 2.687802e-01,
			6.770490e-02, 2.193063e-06, 5.220169e-02, 3.301812e-02, 1.293315e-01,
			8.650576e-02, 3.153980e-02, 5.779063e-03, 1.822102e-01, 1.461806e-02,
			2.823111e-02, 1.347789e-03, 2.753405e-01, 1.568186e-01, 1.461671e-01,
			1.980625e-01, 2.269190e-01, 4.622582e-03, 1.613768e-01, 1.663627e-01,
			2.805365e-01, 1.269349e-01, 2.781936e-02, 1.632794e-01, 8.501158e-02,
			1.320838e-03, 2.103131e-01, 6.247593e-03, 9.941479e-04, 5.048380e-02,
			3.321148e-02, 5.714559e-02, 1.222316e-01, 6.208415e-02, 1.177297e-01,
			9.720416e-02, 4.257352e-02, 2.636569e-01, 2.866397e-03, 3.060865e-01,
			1.018286e-01, 2.732732e-01, 7.325775e-02, 1.060217e-01, 2.164876e-01,
			1.143863e-01, 3.220721e-01, 3.244452e-02, 2.602528e-01, 6.695455e-06,
			6.990345e-02, 2.379897e-02, 1.337605e-01, 3.899722e-02, 2.699237e-02,
			2.201721e-01, 1.491898e-03, 1.175562e-01, 5.758003e-04, 2.683893e-01,
			]
		np.testing.assert_almost_equal(expected, actual)

	def transmissivity_for(self, Tx, Tn):
		return snow_melt.transmissivity(np.array(Tx), np.array(Tn))
