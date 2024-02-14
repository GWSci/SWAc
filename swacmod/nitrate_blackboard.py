import numpy as np

class NitrateBlackboard:
	def __init__(self):
		self.a = None
		self.ae = None
		self.alpha = None
		self.cell_area_m_sq = None
		self.days = None
		self.dSMD_array_mm_per_day = None
		self.effective_porosity = None
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
		self.proportion_reaching_water_table_array_per_day = None
		self.p_non_her = None
		self.Psmd = None
		self.Psoilperc = None
		self.rainfall_ts = None
		self.runoff_mm_per_day = None
		self.runoff_recharge_mm_per_day = None
		self.smd = None
		self.TAW_array_mm = None
		self.time_switcher = None
		self.total_NO3_to_receptors_kg = None
		self.μ = None
		self.σ = None

	def initialise_blackboard(self, data, output, node, logging):
		self.length = output["rainfall_ts"].size

		self.node = node
		self.logging = logging

		self.time_switcher = data["time_switcher"]
		self.days = data["series"]["date"]

		params = data["params"]
		self.nitrate_depth_to_water = params["nitrate_depth_to_water"][self.node]
		self.cell_area_m_sq = params["node_areas"][self.node]
		self.nitrate_loading = params["nitrate_loading"][self.node]
		self.a = params["nitrate_calibration_a"]
		self.μ = params["nitrate_calibration_mu"][self.node]
		self.σ = params["nitrate_calibration_sigma"]
		self.alpha = params["nitrate_calibration_alpha"]
		self.effective_porosity = params["nitrate_calibration_effective_porosity"][self.node]

		self.perc_through_root_mm_per_day = output["perc_through_root"]
		self.TAW_array_mm = output["tawtew"]
		self.smd = output["smd"]
		self.p_smd = output["p_smd"]
		self.runoff_mm_per_day = output["rapid_runoff"]		
		self.runoff_recharge_mm_per_day = output["runoff_recharge"]
		self.macropore_att_mm_per_day = output["macropore_att"]
		self.macropore_dir_mm_per_day = output["macropore_dir"]
		self.interflow_volume = output["interflow_volume"]
		self.infiltration_recharge = output["infiltration_recharge"]
		self.interflow_to_rivers = output["interflow_to_rivers"]
		self.rainfall_ts = output["rainfall_ts"]
		self.ae = output["ae"]
		self.historical_nitrate_reaching_water_table_array_tons_per_day = output["historical_nitrate_reaching_water_table_array_tons_per_day"]
