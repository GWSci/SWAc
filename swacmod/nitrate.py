import numpy as np

# + Derive a continuous relationship between annual HER and % leached nitrate
#   for each model cell, based on the 5, 50 and 95% values given by NEAP-N (see
#   bullet below for the csv of this).
#
# + Divide by 365.25 to get the above rate in daily HER.
#
# + Calculate daily HER for each cell from rainfall minus AE.
#
# + Calculate total mass M0 leached from the cell on that day, ensuring that it
#   doesn’t exhaust the maximum leachable total for the year (when added to
#   leaching from previous days in the hydrological year since 1st October).
#
# - Calculate percolation through the root zone as a proportion of daily HER
#   and assign that as the leached mass M1 available for recharge and
#   interflow.
#
# - Calculate daily flux to the interflow store as a proportion of root zone
#   percolation and remove this from mass M1 to get remnant mass M1a available
#   for recharge.
#
# - Take the sum of daily macropore recharge and runoff recharge as a
#   proportion of HER and assign that as the leached mass M2 available for
#   recharge only. **Note** that because evapotranspiration does not act on
#   macropore and runoff recharge, we may occasionally get a total of
#   percolation and other recharge that exceeds the daily HER value. If M1 + M2
#   exceeds M0, downscale M2 accordingly to get M2a, such that M1 + M2a = M0.
#
# - Calculate the total mass of nitrate in recharge at the top of the
#   unsaturated zone for that day from: Ml = M1a + M2a. This will arrive at the
#   water table over a prolonged period, due to heterogeneity in the
#   unsaturated zone (see final bullet below).
#

def _calculate_her_mm_per_day(data, output, node):
	return output["rainfall_ts"] - output["ae"]

def _calculate_m0_kg_per_day(data, output, node, her_array_mm_per_day):
	params = data["params"]
	cell_area_m_sq = params["node_areas"][node][0]
	days = data["series"]["date"]

	nitrate_leaching = params["nitrate_leaching"][node]
	max_load_per_year_kg_per_hectare = nitrate_leaching[3]
	her_at_5_percent = nitrate_leaching[4]
	her_at_50_percent = nitrate_leaching[5]
	her_at_95_percent = nitrate_leaching[6]

	hectare_area_m_sq = 10000
	max_load_per_year_kg_per_cell = max_load_per_year_kg_per_hectare * cell_area_m_sq / hectare_area_m_sq

	m0_array_kg_per_day = _calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_array_mm_per_day)
	return m0_array_kg_per_day

def _calculate_total_mass_leached_from_cell_on_days(
		max_load_per_year_kg_per_cell,
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		days,
		her_per_day):
	length = len(days)
	result = np.zeros(length)
	remaining_for_year = max_load_per_year_kg_per_cell
	for i in range(length):
		day = days[i]
		her = her_per_day[i]
		if (day.month == 10) and (day.day == 1):
			remaining_for_year = max_load_per_year_kg_per_cell
		fraction_leached = _cumulative_fraction_leaked_per_day(her_at_5_percent,
			her_at_50_percent,
			her_at_95_percent,
			her)
		mass_leached_for_day = min(remaining_for_year, max_load_per_year_kg_per_cell * fraction_leached)
		remaining_for_year -= mass_leached_for_day
		result[i] = mass_leached_for_day
	return result

def _cumulative_fraction_leaked_per_day(
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		her_per_day):
	days_in_year = 365.25
	her_per_year = days_in_year * her_per_day
	x = her_per_year
	is_below_50_percent = her_per_year < her_at_50_percent
	upper = her_at_50_percent if is_below_50_percent else her_at_95_percent
	lower = her_at_5_percent if is_below_50_percent else her_at_50_percent
	# y = mx + c
	m = 0.45 / (upper - lower)
	c = 0.5 - (her_at_50_percent * m)
	y = (m * x) + c
	return y / days_in_year

def _cumulative_fraction_leaked_per_year(
		her_at_5_percent,
		her_at_50_percent,
		her_at_95_percent,
		her_per_year):
	x = her_per_year
	is_below_50_percent = her_per_year < her_at_50_percent
	upper = her_at_50_percent if is_below_50_percent else her_at_95_percent
	lower = her_at_5_percent if is_below_50_percent else her_at_50_percent
	# y = mx + c
	m = 0.45 / (upper - lower)
	c = 0.5 - (her_at_50_percent * m)
	y = (m * x) + c
	return y
