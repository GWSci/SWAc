import flopy

def mf_simulation():
	return flopy.mf6.MFSimulation(verbosity_level=0)

def mf_model(sim, path):
	return flopy.mf6.mfmodel.MFModel(sim, modelname=path)
