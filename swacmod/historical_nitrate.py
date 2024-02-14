import numpy as np
import swacmod.model as m
import swacmod.nitrate_proportion_reaching_water_table as nitrate_proportion

class HistoricalNitrateBlackboard():
	def __init__(self):
		self.a = None
		self.alpha = None
		self.days = None
		self.effective_porosity = None
		self.historical_mi_array_kg_per_day = None
		self.historical_mi_array_kg_per_time_period = None
		self.historical_mass_reaching_water_table_array_kg_per_day = None
		self.historical_nitrate_reaching_water_table_array_tons_per_day = None
		self.historical_nitrate_days = None
		self.historical_time_periods = None
		self.historic_proportion_reaching_water_table_array_per_day = None
		self.nitrate_depth_to_water = None
		self.node = None
		self.proportion_0 = None
		self.time_switcher = None
		self.truncated_historical_mi_array_kg_per_day = None
		self.truncated_historical_nitrate_days = None
		self.μ = None
		self.σ = None

	def initialise_blackboard(self, data, output, node):
		total_days_upper_bound = calculate_total_days_count_upper_bound(data)

		self.a = data["params"]["nitrate_calibration_a"]
		self.alpha = data["params"]["nitrate_calibration_alpha"]
		self.days = data["series"]["date"]
		self.effective_porosity = data["params"]["nitrate_calibration_effective_porosity"]
		self.historical_mi_array_kg_per_time_period = data["params"]["historical_mi_array_kg_per_time_period"][node]
		self.historical_nitrate_days = data["series"]["historical_nitrate_days"]
		self.historical_time_periods = data["params"]["historical_time_periods"]
		self.nitrate_depth_to_water = data["params"]["nitrate_depth_to_water"][node]
		self.node = node
		self.proportion_0 = np.zeros(total_days_upper_bound)
		self.μ = data["params"]["nitrate_calibration_mu"]
		self.σ = data["params"]["nitrate_calibration_sigma"]
		return self

def calculate_total_days_count_upper_bound(data):
	date_len = 0
	if "date" in data["series"]:
		date_len = len(data["series"]["date"])
	historical_nitrate_days_len = 0
	if "historical_nitrate_days" in data["series"]:
		historical_nitrate_days_len = len(data["series"]["historical_nitrate_days"])
	return date_len + historical_nitrate_days_len

def get_historical_nitrate(data, output, node):
	if (data["params"]["historical_nitrate_process"] == "enabled"):
		blackboard = HistoricalNitrateBlackboard()
		blackboard = blackboard.initialise_blackboard(data, output, node)
		blackboard = _calculate_historical_nitrate(blackboard)
		return {
			"historical_nitrate_reaching_water_table_array_tons_per_day": blackboard.historical_nitrate_reaching_water_table_array_tons_per_day,
		}
	else:
		length = len(data["series"]["date"])
		empty_array = np.zeros(length)
		return {
			"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
		}

def _calculate_aggregate_mi_unpacking(blackboard):
	return m._calculate_aggregate_mi_unpacking(blackboard)

def _calculate_historical_nitrate(blackboard):
	blackboard.historical_mi_array_kg_per_day = _calculate_aggregate_mi_unpacking(blackboard)
	blackboard.truncated_historical_nitrate_days = _calculate_truncated_historical_nitrate_days(blackboard)
	blackboard.truncated_historical_mi_array_kg_per_day = _calculate_truncated_historical_mi_array_kg_per_day(blackboard)
	blackboard.historic_proportion_reaching_water_table_array_per_day = _calculate_historic_proportion_reaching_water_table_array_per_day(blackboard)
	blackboard.historical_mass_reaching_water_table_array_kg_per_day = _calculate_historical_mass_reaching_water_table_array_kg_per_day(blackboard)
	blackboard.historical_nitrate_reaching_water_table_array_tons_per_day = _convert_kg_to_tons_array(blackboard)
	return blackboard

def _calculate_truncated_historical_nitrate_days(blackboard):
	historical_nitrate_days = blackboard.historical_nitrate_days
	days = blackboard.days
	if len(days) == 0:
		return historical_nitrate_days
	first_new_day = days[0]
	truncated_historical_nitrate_days = [d for d in historical_nitrate_days if d < first_new_day]
	return truncated_historical_nitrate_days

def _calculate_truncated_historical_mi_array_kg_per_day(blackboard):
	truncated_length = len(blackboard.truncated_historical_nitrate_days)
	return blackboard.historical_mi_array_kg_per_day[:truncated_length]

def _calculate_historic_proportion_reaching_water_table_array_per_day(blackboard):	
	return nitrate_proportion.calculate_historic_proportion_reaching_water_table_array_per_day(blackboard)

def _calculate_historical_mass_reaching_water_table_array_kg_per_day(blackboard):
	return m.calculate_historical_mass_reaching_water_table_array_kg_per_day(blackboard)

def _convert_kg_to_tons_array(blackboard):
	return blackboard.historical_mass_reaching_water_table_array_kg_per_day / 1000.0
