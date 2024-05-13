import flopy
import numpy as np

def mf_simulation():
	return flopy.mf6.MFSimulation(verbosity_level=0)

def mf_model(sim, path):
	return flopy.mf6.mfmodel.MFModel(sim, modelname=path)

def mf_gwf_disv(model):
	return flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(model)

def mf_gwf_disu(model, nodes, njag):
	return flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(model,
                                                       nodes=nodes,
                                                       ja=np.zeros((njag),
                                                                   dtype=int),
                                                       nja=njag, ihc=[1],
                                                       iac=[1])

def mf_gwf_disu_with_area(model, nodes, njag, area):
	return flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(model,
                                                       nodes=nodes,
                                                       ja=np.zeros((njag),
                                                                   dtype=int),
                                                       nja=njag, 
                                                       ihc=[1],
                                                       iac=[1],
                                                       area=area)
