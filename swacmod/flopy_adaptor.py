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

def modflow_model(path, version, structured):
	return flopy.modflow.Modflow(modelname=path, version=version, structured=structured)

def modflow_disu(model, nodes, nper, njag, lenx):
	return flopy.modflow.ModflowDisU(
		model,
		nodes=nodes,
		nper=nper,
		iac=[njag] + (nodes - 1) * [0],
		ja=np.zeros((njag), dtype=int),
		njag=njag,
		idsymrd=1,
		cl1=np.zeros((lenx)),
		cl2=np.zeros((lenx)),
		fahl=np.zeros((lenx)))

def make_empty_modflow_gwf_rch_stress_period_data(model, nodes, nper):
	return flopy.mf6.ModflowGwfrch.stress_period_data.empty(
		model,
		maxbound=nodes,
		nseg=1,
		stress_periods=range(nper))

def mf_gwf_rch(model, nodes, spd):
	return flopy.mf6.modflow.mfgwfrch.ModflowGwfrch(model,
                                                       fixed_cell=False,
                                                       print_input=None,
                                                       print_flows=None,
                                                       save_flows=None,
                                                       timeseries=None,
                                                       observations=None,
                                                       maxbound=nodes,
                                                       stress_period_data=spd,
                                                       filename=None,
                                                       pname=None,
                                                       parent_file=None)

def modflow_sfr2_get_empty_segment_data(nss):
	return flopy.modflow.ModflowSfr2.get_empty_segment_data(nss)

def modflow_sfr2_get_empty_reach_data(nstrm):
	return flopy.modflow.ModflowSfr2.get_empty_reach_data(nstrm, structured=False)

def mf_str2(model, nstrm, nss, istcb1, istcb2, rd, seg_data):
	return flopy.modflow.mfsfr2.ModflowSfr2(
		model,
		nstrm=nstrm,
		nss=nss,
		nsfrpar=0,
		nparseg=0,
		const=None,
		dleak=0.0001,
		ipakcb=istcb1,
		istcb2=istcb2,
		isfropt=1,
		nstrail=10,
		isuzn=1,
		nsfrsets=30,
		irtflg=0,
		numtim=2,
		weight=0.75,
		flwtol=0.0001,
		reach_data=rd,
		segment_data=seg_data,
		channel_geometry_data=None,
		channel_flow_data=None,
		dataset_5=None,
		irdflag=0,
		iptflag=0,
		reachinput=True,
		transroute=False,
		tabfiles=False,
		tabfiles_dict=None,
		extension='sfr',
		unit_number=None,
		filenames=None)

def mf_gwf_sfr(model, nss, rd, cd, sd):
	return flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr(
		model,
		loading_package=False,
		auxiliary=None,
		boundnames=None,
		print_input=None,
		print_stage=None,
		print_flows=None,
		save_flows=None,
		stage_filerecord=None,
		budget_filerecord=None,
		timeseries=None,
		observations=None,
		mover=None,
		maximum_iterations=None,
		unit_conversion=86400.0,
		nreaches=nss,
		packagedata=rd,
		connectiondata=cd,
		diversions=None,
		perioddata=sd,
		filename=None,
		pname=None,
		parent_file=None)
