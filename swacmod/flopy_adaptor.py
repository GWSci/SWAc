import flopy

def mf_simulation():
	return flopy.mf6.MFSimulation(verbosity_level=0)
