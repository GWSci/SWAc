# + Derive a continuous relationship between annual HER and % leached nitrate
#   for each model cell, based on the 5, 50 and 95% values given by NEAP-N (see
#   bullet below for the csv of this).
#
# + Divide by 365.25 to get the above rate in daily HER.
#
# + Calculate daily HER for each cell from rainfall minus AE.
#
# - Calculate total mass M0 leached from the cell on that day, ensuring that it
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

def _calculate_daily_HER(data, output, node):
	return output["rainfall_ts"] - output["ae"]
