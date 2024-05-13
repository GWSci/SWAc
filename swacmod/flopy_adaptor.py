import flopy
import numpy as np

def mf_simulation():
	return flopy.mf6.MFSimulation(verbosity_level=0)

def mf_model(sim, path):
	return flopy.mf6.mfmodel.MFModel(sim, modelname=path)

def mf_gwf_disv(model):
	return flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(model)

def mf_gwf_disu(model, nodes, njag, area=None):
	return flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(
		model,
		nodes=nodes,
		ja=np.zeros((njag), dtype=int),
		nja=njag,
		ihc=[1],
		iac=[1],
		area=area)

def mf_tdis(sim, nper):
	return flopy.mf6.modflow.mftdis.ModflowTdis(
		sim,
		loading_package=False,
		time_units=None,
		start_date_time=None,
		nper=nper,
		filename=None,
		pname=None,
		parent_file=None)

def modflow_model(path, structured):
	return flopy.modflow.Modflow(modelname=path, version='mfusg', structured=structured)
