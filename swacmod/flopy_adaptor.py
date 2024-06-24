import flopy
import numpy as np

def make_mfusg_evt(path, nodes, nper, nevtopt, ievtcb, evt_dic, surf, exdp, ievt):
	m = modflow_model(path, "mfusg", True)
	modflow_dis(m, 1, nodes, 1, nper)
	evt_out = modflow_evt(m, nevtopt, ievtcb, evt_dic, surf, exdp, ievt)
	return evt_out

def make_model_with_disu_and_empty_spd_for_evt_out(path, nper, nodes):
	sim = _mf_simulation()
	m = _mf_model(sim, path)
	njag = nodes + 2
	_mf_gwf_disu(m, nodes, njag)

	_mf_tdis(sim, nper)

	spd = make_empty_modflow_gwf_evt_stress_period_data(m, nodes, nper)
	return m, spd

def make_model_with_disv_and_empty_spd_for_rch_out(path, nper, nodes):
	sim = _mf_simulation()
	m = _mf_model(sim, path)
	_mf_gwf_disv(m)

	_mf_tdis(sim, nper)

	spd = _make_empty_modflow_gwf_rch_stress_period_data(m, nodes, nper)
	return m, spd

def make_model_with_disu_and_empty_spd_for_rch_out(path, nper, nodes):
	sim = _mf_simulation()
	m = _mf_model(sim, path)
	njag = nodes + 2
	_mf_gwf_disu(m, nodes, njag, area=1.0)

	_mf_tdis(sim, nper)

	spd = _make_empty_modflow_gwf_rch_stress_period_data(m, nodes, nper)
	return m, spd

def make_model_for_sfr_mfusg(path, nodes, nper, njag, lenx, nss, nstrm):
	m = modflow_model(path, "mfusg", False)

	modflow_disu(m, nodes, nper, njag, lenx)

	m.dis = m.disu
	sd = modflow_sfr2_get_empty_segment_data(nss)
	rd = modflow_sfr2_get_empty_reach_data(nstrm)
	return m, sd, rd

def make_model_for_sfr_mf6_disv(path, nper):
	sim = _mf_simulation()
	m = _mf_model(sim, path)
	_mf_gwf_disv(m)
	_mf_tdis(sim, nper)
	return m

def make_model_for_sfr_mf6_disu(path, nodes, nper):
	sim = _mf_simulation()
	m = _mf_model(sim, path)
	njag = nodes + 2
	_mf_gwf_disu(m, nodes, njag)
	_mf_tdis(sim, nper)
	return m

def _mf_simulation():
	return flopy.mf6.MFSimulation(verbosity_level=0)

def _mf_model(sim, path):
	return flopy.mf6.mfmodel.MFModel(sim, modelname=path)

def _mf_gwf_disv(model):
	return flopy.mf6.modflow.mfgwfdisv.ModflowGwfdisv(model)

def _mf_gwf_disu(model, nodes, njag, area=None):
	return flopy.mf6.modflow.mfgwfdisu.ModflowGwfdisu(
		model,
		nodes=nodes,
		ja=np.zeros((njag), dtype=int),
		nja=njag,
		ihc=[1],
		iac=[1],
		area=area)

def _mf_tdis(sim, nper):
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

def _make_empty_modflow_gwf_rch_stress_period_data(model, nodes, nper):
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

def write_mf_gwf_rch(rch):
	rch.write()

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

def modflow_bas(model):
	return flopy.modflow.ModflowBas(model, ifrefm=False)

def modflow_str_get_empty(ncells, nss):
	return flopy.modflow.ModflowStr.get_empty(ncells=ncells, nss=nss)

def modflow_dis(model, nlay, nrow, ncol, nper):
	return flopy.modflow.ModflowDis(
		model,
		nlay=nlay,
		nrow=nrow,
		ncol=ncol,
		nper=nper)

def modflow_str(model, nstrm, istcb1, istcb2, reach_data, segment_data):
    return flopy.modflow.ModflowStr(
		model,
		mxacts=nstrm,
		nss=nstrm,
		ntrib=8,
		ipakcb=istcb1,
		istcb2=istcb2,
		stress_period_data=reach_data,
		segment_data=segment_data,
		irdflg={0:2, 1:2})

def make_empty_modflow_gwf_evt_stress_period_data(model, nodes, nper):
	return flopy.mf6.ModflowGwfevt.stress_period_data.empty(
		model,
		maxbound=nodes,
		nseg=1,
		stress_periods=range(nper),
		aux_vars=['dummy'])

def modflow_evt(model, nevtopt, ievtcb, evt_dic, surf, exdp, ievt):
	return flopy.modflow.ModflowEvt(
		model,
		nevtop=nevtopt,
		ipakcb=ievtcb,
		evtr=evt_dic,
		surf={0: surf},
		exdp={0: exdp},
		ievt={0: ievt})

def modflow_gwf_evt(model, nodes, spd):
	return flopy.mf6.ModflowGwfevt(
		model,
		fixed_cell=False,
		print_input=None,
		print_flows=None,
		save_flows=None,
		timeseries=None,
		observations=None,
		maxbound=nodes,
		nseg=1,
		stress_period_data=spd,
		surf_rate_specified=False,
		filename=None,
		pname=None,
		parent_file=None)
