import numpy as np
import swacmod.model as m
import swacmod.nitrate_proportion_reaching_water_table as nitrate_proportion

class HistoricalNitrateBlackboard():
	def __init__(self):
		self.a = None
		self.days = None
		self.historical_mi_array_kg_per_day = None
		self.historical_mass_reaching_water_table_array_kg_per_day = None
		self.historical_nitrate_reaching_water_table_array_tons_per_day = None
		self.historical_nitrate_days = None
		self.historic_proportion_reaching_water_table_array_per_day = None
		self.mean_hydraulic_conductivity = None
		self.mean_velocity_of_unsaturated_transport = None
		self.nitrate_depth_to_water = None
		self.time_switcher = None
		self.truncated_historical_mi_array_kg_per_day = None
		self.truncated_historical_nitrate_days = None
		self.μ = None
		self.σ = None

	def initialise_blackboard(self, data, output, node):
		self.a = data["params"]["nitrate_calibration_a"]
		self.days = data["series"]["date"]
		self.historical_mi_array_kg_per_day = data["series"]["historical_mi_kg_per_day"][node]
		self.historical_nitrate_days = data["series"]["historical_nitrate_days"]
		return self

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}

def _calculate_historical_nitrate(blackboard):
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
