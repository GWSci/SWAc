import numpy as np

class NitrateBlackboard:
	def __init__(self):
		self.a = None
		self.ae = None
		self.cell_area_m_sq = None
		self.days = None
		self.dSMD_array_mm_per_day = None
		self.her_array_mm_per_day = None
		self.historical_nitrate_reaching_water_table_array_tons_per_day = None
		self.infiltration_recharge = None
		self.interflow_to_rivers = None
		self.interflow_volume = None
		self.length = None
		self.logging = None
		self.M_soil_in_kg = None
		self.M_soil_tot_kg = None
		self.m0_array_kg_per_day = None
		self.m1_array_kg_per_day = None
		self.m1a_array_kg_per_day = None
		self.m2_array_kg_per_day = None
		self.m3_array_kg_per_day = None
		self.m4_array_kg_per_day = None
		self.macropore_att_mm_per_day = None
		self.macropore_dir_mm_per_day = None
		self.mass_balance_error_kg = None
		self.mean_hydraulic_conductivity = None
		self.mean_velocity_of_unsaturated_transport = None
		self.mi_array_kg_per_day = None
		self.nitrate_depth_to_water = None
		self.nitrate_loading = None
		self.nitrate_reaching_water_table_array_from_this_run_kg_per_day = None
		self.nitrate_reaching_water_table_array_from_this_run_tons_per_day = None
		self.nitrate_reaching_water_table_array_tons_per_day = None
		self.node = None
		self.p_non = None
		self.p_smd = None
		self.perc_through_root_mm_per_day = None
		self.Pherperc = None
		self.Pro = None
		self.proportion_0 = None
		self.proportion_100 = None
		self.proportion_reaching_water_table_array_per_day = None
		self.Psmd = None
		self.Psoilperc = None
		self.rainfall_ts = None
		self.runoff_recharge_mm_per_day = None
		self.smd = None
		self.TAW_array_mm = None
		self.time_switcher = None
		self.total_NO3_to_receptors_kg = None
		self.μ = None
		self.σ = None

	def initialise_blackboard(blackboard, data, output, node, logging):
		blackboard.length = output["rainfall_ts"].size

		blackboard.node = node
		blackboard.logging = logging
		blackboard.proportion_0 = np.zeros(blackboard.length)

		blackboard.time_switcher = data["time_switcher"]
		blackboard.days = data["series"]["date"]
		blackboard.proportion_100 = data["proportion_100"]

		params = data["params"]
		blackboard.nitrate_depth_to_water = params["nitrate_depth_to_water"][blackboard.node]
		blackboard.cell_area_m_sq = params["node_areas"][blackboard.node]
		blackboard.nitrate_loading = params["nitrate_loading"][blackboard.node]
		blackboard.a = params["nitrate_calibration_a"]
		blackboard.μ = params["nitrate_calibration_mu"]
		blackboard.σ = params["nitrate_calibration_sigma"]
		blackboard.mean_hydraulic_conductivity = params["nitrate_calibration_mean_hydraulic_conductivity"]
		blackboard.mean_velocity_of_unsaturated_transport = params["nitrate_calibration_mean_velocity_of_unsaturated_transport"]

		blackboard.perc_through_root_mm_per_day = output["perc_through_root"]
		blackboard.TAW_array_mm = output["tawtew"]
		blackboard.smd = output["smd"]
		blackboard.p_smd = output["p_smd"]
		blackboard.runoff_recharge_mm_per_day = output["runoff_recharge"]
		blackboard.macropore_att_mm_per_day = output["macropore_att"]
		blackboard.macropore_dir_mm_per_day = output["macropore_dir"]
		blackboard.interflow_volume = output["interflow_volume"]
		blackboard.infiltration_recharge = output["infiltration_recharge"]
		blackboard.interflow_to_rivers = output["interflow_to_rivers"]
		blackboard.rainfall_ts = output["rainfall_ts"]
		blackboard.ae = output["ae"]
		blackboard.historical_nitrate_reaching_water_table_array_tons_per_day = output["historical_nitrate_reaching_water_table_array_tons_per_day"]
