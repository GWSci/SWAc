import numpy as np
import swacmod.nitrate as nitrate

class HistoricalNitrateBlackboard():
	def __init__(self):
		self.a = None
		self.date = None
		self.historical_mi_array_kg_per_day = None
		self.historical_nitrate_date = None
		self.mean_hydraulic_conductivity = None
		self.mean_velocity_of_unsaturated_transport = None
		self.nitrate_depth_to_water = None
		self.time_switcher = None
		self.truncated_historical_mi_array_kg_per_day = None
		self.truncated_historical_nitrate_dates = None
		self.μ = None
		self.σ = None

def get_historical_nitrate(data, output, node):
	length = len(data["series"]["date"])
	empty_array = np.zeros(length)
	return {
		"historical_nitrate_reaching_water_table_array_tons_per_day": empty_array,
	}

def _calculate_historical_nitrate(blackboard):
	blackboard.truncated_historical_nitrate_dates = _calculate_truncated_historical_nitrate_date(blackboard)
	blackboard.truncated_historical_mi_array_kg_per_day = _calculate_truncated_historical_mi_array_kg_per_day(blackboard)
	return blackboard

def _calculate_truncated_historical_nitrate_date(blackboard):
	historical_nitrate_date = blackboard.historical_nitrate_date
	date = blackboard.date
	if len(date) == 0:
		return historical_nitrate_date
	first_new_date = date[0]
	truncated_historical_nitrate_dates = [d for d in historical_nitrate_date if d < first_new_date]
	return truncated_historical_nitrate_dates

def _calculate_truncated_historical_mi_array_kg_per_day(blackboard):
	truncated_length = len(blackboard.truncated_historical_nitrate_dates)
	return blackboard.historical_mi_array_kg_per_day[:truncated_length]

def _calculate_historic_proportion_reaching_water_table_array_per_day(blackboard):	
	time_switcher = blackboard.time_switcher
	length = len(blackboard.date)
	depth_to_water_m = blackboard.nitrate_depth_to_water[0]
	if depth_to_water_m == 0.0:
		return blackboard.proportion_0
	elif depth_to_water_m == 100.0:
		return blackboard.proportion_100
	else:
		return nitrate.__calculate_proportion_reaching_water_table_array_per_day(length, blackboard.a, blackboard.μ, blackboard.σ, blackboard.mean_hydraulic_conductivity, blackboard.mean_velocity_of_unsaturated_transport, depth_to_water_m, time_switcher)
