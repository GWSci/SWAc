from datetime import datetime
import numpy as np

class SnowMelt:
	def SnowMelt(self, Date, precip_mm, Tmax_C, Tmin_C, lat_deg, slope, aspect, tempHt, windHt, groundAlbedo, SurfEmissiv, windSp, forest, startingSnowDepth_m, startingSnowDensity_kg_m3):
		## Constants :
		WaterDens = 1000            # kg/m3
		lambdaV = 2500              # (kJ/kg) latent heat of vaporization
		SnowHeatCap = 2.1           # kJ/kg/C
		LatHeatFreez = 333.3        # kJ/kg
		Cw = 4.2*10**3              # Heat Capacity of Water (kJ/m3/C)

		## Converted Inputs :
		Tav = (Tmax_C+Tmin_C)/2		      # degrees C
		precip_m = precip_mm*0.001        # precip in m 
		R_m = np.copy(precip_m)           # (m) depth of rain
		R_m[Tav < 0] = 0                  # ASSUMES ALL SNOW at < 0C
		NewSnowDensity = 50+3.4*(Tav+15)  # kg/m3
		NewSnowDensity[NewSnowDensity < 50] = 50
		NewSnowWatEq = np.copy(precip_m)  # m
		NewSnowWatEq[Tav >= 0] = 0        # No new snow if average temp above or equals 0 C
		NewSnow = NewSnowWatEq*WaterDens/NewSnowDensity     # m
		
		jday_string_to_day_in_year = lambda x: datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday
		JDay = Date if (Date.size == 0) else np.vectorize(jday_string_to_day_in_year)(Date)
		
		lat = lat_deg*np.pi/180	          # latitude in radians
		rh = np.log((windHt+0.001)/0.001)*np.log((tempHt+0.0002)/0.0002)/(0.41*0.41*windSp*86400)	# (day/m) Thermal Resistance	 
		if (np.isscalar(windSp)): rh = np.full(precip_mm.size, rh)									##	creates a vector of rh values
		cloudiness = EstCloudiness(Tmax_C,Tmin_C)
		AE         = AtmosphericEmissivity(Tav, cloudiness)	# (-) Atmospheric Emissivity
		
		#  New Variables	:
		SnowTemp        = np.zeros_like(precip_m, dtype=float) 		# Degrees C
		rhos            = SatVaporDensity(SnowTemp)	# 	vapor density at surface (kg/m3)
		rhoa            = SatVaporDensity(Tmin_C)		#	vapor density of atmoshpere (kg/m3) 
		SnowWaterEq     = np.zeros_like(precip_mm, dtype=float)		#  (m) Equiv depth of water
		TE              = np.full_like(precip_mm, SurfEmissiv, dtype=float)	#	(-) Terrestrial Emissivity
		DCoef           = np.zeros_like(precip_m, dtype=float)				#   Density Coefficient (-) (Simplified version)
		SnowDensity 	= np.full_like(precip_mm, 450, dtype=float)			#  (kg/m3)  Max density is 450
		SnowDepth       = np.full_like(precip_mm, False, dtype=float)		#  (m)
		SnowMelt        = np.zeros_like(precip_m, dtype=float)				#  (m)
		Albedo          = np.full_like(precip_mm, groundAlbedo, dtype=float) 	#  (-) This will change for days with snow
		
		##	Energy Terms
		H 		= np.zeros_like(precip_m)	#	Sensible Heat exchanged (kJ/m2/d)
		E 		= np.zeros_like(precip_m)	#	Vapor Energy	(kJ/m2/d)
		S 		= np.zeros_like(precip_m)	#	Solar Radiation (kJ/m2/d)
		La 		= Longwave(AE, Tav)					#	Atmospheric Longwave Radiation (kJ/m2/d)
		Lt 		= np.zeros_like(precip_m)	#	Terrestrial Longwave Radiation (kJ/m2/d)
		G 		= 173								#	Ground Condution (kJ/m2/d) 
		P 		= Cw * R_m * Tav					# 	Precipitation Heat (kJ/m2/d)
		Energy 	= np.zeros_like(precip_m)	# Net Energy (kJ/m2/d)

		##  Initial Values. 
		SnowWaterEq[0] = startingSnowDepth_m * startingSnowDensity_kg_m3 / WaterDens
		SnowDepth[0] = startingSnowDepth_m			
		Albedo[0] = ifelse(NewSnow[0] > 0, 0.98-(0.98-0.50)*np.exp(-4*NewSnow[0]*10),ifelse(startingSnowDepth_m == 0, groundAlbedo, max(groundAlbedo, 0.5+(groundAlbedo-0.85)/10)))  # If snow on the ground or new snow, assume Albedo yesterday was 0.5
		S[0] = Solar(lat, np.array([JDay[0]]), Tmax_C[0], Tmin_C[0], Albedo[0], forest, aspect, slope, True)[0]
		H[0] = 1.29*(Tav[0]-SnowTemp[0])/rh[0] 
		E[0] = lambdaV*(rhoa[0]-rhos[0])/rh[0]
		if(startingSnowDepth_m > 0): 
			TE[0] = 0.97 
		Lt[0] = Longwave(TE[0],SnowTemp[0])
		Energy[0] = S[0] + La[0] - Lt[0] + H[0] + E[0] + G + P[0]
		if ((startingSnowDepth_m+NewSnow[0])>0):
			SnowDensity[0] = min(450, (startingSnowDensity_kg_m3*startingSnowDepth_m + NewSnowDensity[0]*NewSnow[0])/(startingSnowDepth_m+NewSnow[0]))
		else:
			SnowDensity[0] = 450
		SnowMelt[0] = max(0,
		                  min((startingSnowDepth_m/10+NewSnowWatEq[0]),  # yesterday on ground + today new  
		                      (Energy[0]-SnowHeatCap*(startingSnowDepth_m/10+NewSnowWatEq[0])*WaterDens*(0-SnowTemp[0]))/(LatHeatFreez*WaterDens) ) )
		SnowDepth[0] = max(0, (startingSnowDepth_m/10 + NewSnowWatEq[0]-SnowMelt[0])*WaterDens/SnowDensity[0])
		SnowWaterEq[0] = max(0,startingSnowDepth_m/10-SnowMelt[0]+NewSnowWatEq[0])	

		# Snow Melt Loop	
		for i in range(1, precip_m.size):
			if (NewSnow[i] > 0):
				Albedo[i] = 0.98-(0.98-Albedo[i-1])*np.exp(-4*NewSnow[i]*10)
			elif (SnowDepth[i-1] < 0.1):
				Albedo[i] = max(groundAlbedo, Albedo[i-1]+(groundAlbedo-0.85)/10)
			else:
				Albedo[i] = 0.35-(0.35-0.98)*np.exp(-1* np.power((0.177+np.power((np.log((-0.3+0.98)/(Albedo[i-1]-0.3))), 2.16)), 0.46))
	
			S[i] = Solar(lat, JDay[i], Tmax_C[i], Tmin_C[i], Albedo[i-1], forest, slope, aspect, False)

			if(SnowDepth[i-1] > 0): TE[i] = 0.97 	#	(-) Terrestrial Emissivity
			if(SnowWaterEq[i-1] > 0 or NewSnowWatEq[i] > 0):
				DCoef[i] = 6.2
				if(SnowMelt[i-1] == 0):
					SnowTemp[i] = max(
						min(0,Tmin_C[i]),
						min(0,(SnowTemp[i-1]+min(-SnowTemp[i-1],Energy[i-1]/((SnowDensity[i-1]*SnowDepth[i-1]+NewSnow[i]*NewSnowDensity[i])*SnowHeatCap*1000))))
						)

			rhos[i] = SatVaporDensity(SnowTemp[i])
			H[i] = 1.29*(Tav[i]-SnowTemp[i])/rh[i] 
			E[i] = lambdaV*(rhoa[i]-rhos[i])/rh[i]
			Lt[i] = Longwave(TE[i],SnowTemp[i])
			Energy[i] = S[i] + La[i] - Lt[i] + H[i] + E[i] + G + P[i]

			k = ifelse( (Energy[i]>0), 2, 1)
		
			if ((SnowDepth[i-1]+NewSnow[i]) > 0):
				SnowDensity[i] = min(
						450, 
						((SnowDensity[i-1]+k*30*(450-SnowDensity[i-1])*np.exp(-DCoef[i]))*SnowDepth[i-1] + NewSnowDensity[i]*NewSnow[i])/(SnowDepth[i-1]+NewSnow[i])
					)
			else:
				SnowDensity[i] = 450

			SnowMelt[i] = max(
				0,	
				min(
					(SnowWaterEq[i-1]+NewSnowWatEq[i]),  # yesterday on ground + today new
					(Energy[i]-SnowHeatCap*(SnowWaterEq[i-1]+NewSnowWatEq[i])*WaterDens*(0-SnowTemp[i]))/(LatHeatFreez*WaterDens)
				)  
			)

			SnowDepth[i] = max(0,(SnowWaterEq[i-1]+NewSnowWatEq[i]-SnowMelt[i])*WaterDens/SnowDensity[i])
			SnowWaterEq[i] = max(0,SnowWaterEq[i-1]-SnowMelt[i]+NewSnowWatEq[i])	# (m) Equiv depth of water

		## Expose the calculated values for test purposes.
		self.NewSnowDensity = NewSnowDensity
		self.JDay = JDay
		self.lat = lat
		self.rh = rh
		self.AE = AE

		self.SnowTemp = SnowTemp
		self.DCoef = DCoef
		self.SnowDensity = SnowDensity
		self.Albedo = Albedo

		self.Energy = Energy

		return (Date, Tmax_C, Tmin_C, precip_mm, R_m*1000, NewSnowWatEq*1000, SnowMelt*1000, NewSnow, SnowDepth, SnowWaterEq*1000)

def EstCloudiness(Tx, Tn):
	transMin = 0.15
	transMax = 0.75
	trans = transmissivity(Tx, Tn)

	cl = 1 - (trans-transMin) / (transMax-transMin)
	cl[cl > 1] = 1
	cl[cl < 0] = 0 # TODO This line is untested. Unsure how to force this condition.
	return cl

# fraction of direct solar radiation passing through 
# the atmosphere based on the Bristow-Campbell eqn
#Tx: maximum daily temperature [C]
#Tn: minimum daily temperature [C]
def transmissivity(Tx, Tn):
	A=0.75
	C=2.4
	Len = Tx.size
	dT = (Tx-Tn)  # diurnal temperature range just difference between max and min daily temperature
	avDeltaT = np.zeros(Len)
	
	if (Len<30):
		avDeltaT = np.mean(dT)
	else:
		avDeltaT[0:14] = np.mean(dT[0:30])
		countForLast15 = min(31, dT.size)
		avDeltaT[(Len-15):Len] = np.mean(dT[-countForLast15:])
		for i in range(14, Len-15):
			avDeltaT[i] = np.mean(dT[(i-14):(i+16)])
	B = 0.036*np.exp(-0.154*avDeltaT)
	return(A*(1-np.exp(-B*np.power(dT, C))))


def AtmosphericEmissivity(airtemp, cloudiness):
	return ((0.72 + 0.005 * airtemp) * (1 - 0.84 * cloudiness) + 0.84 * cloudiness)

#	T_C	= Temperature [C]
def SatVaporDensity(T_C):
	VP = SatVaporPressure(T_C)
	return(np.round(VP/(0.462 * (T_C+273.15)), 4))

# saturated vapor pressure at a given temperature (kPa)
#T_C: temperature [C]
def SatVaporPressure(T_C):
	return(0.611 * np.exp((17.3*T_C)/(237.2+T_C)))

# daily longwave radiation based on the Sephan-Boltzman equation [kJ m-2 d-1]

#emissivity: [-]
#temp: temperature of the emitting body [C]
def Longwave(emissivity, temp):
	SBconstant = 0.00000490 #[kJ m-2 K-4 d-1]
	tempK = temp+273.15 #[degrees K]
	return(emissivity*SBconstant*np.power(tempK, 4))

# potential solar radiation at the edge of the atmospher [kJ m-2 d-1]
#lat: latitdue [rad]
#Jday: Julian date or day of the year [day]
def PotentialSolar(lat, Jday):
	dec = declination(Jday)
	return(117500*(np.arccos(-np.tan(dec)*np.tan(lat))*np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.sin(np.arccos(np.tan(dec)*np.tan(lat))))/np.pi)


# solar declination [rad]
#Jday: Julian date or day of the year [day]
def declination(Jday):
	return(0.4102*np.sin(np.pi*(Jday-80)/180))

# slopefactor: adjusts solar radiation for land slope and aspect relative to the sun, 1=level ground
#lat: latitdue [rad]
#Jday: Julian date or day of the year [day]
#slope: slope of the ground [rad]
#aspect: ground aspect [rad from north]
def slopefactor(lat, Jday, slope, aspect):
	SolAsp = np.full_like(Jday, np.pi)  # Average Solar aspect is binary - either north (0) or south (pi) for the day
	SolAsp[lat - declination(Jday) < 0] = 0   # 
	SF = np.cos(slope) - np.sin(slope)*np.cos(aspect-(np.pi-SolAsp))/np.tan(solarangle(lat,Jday))
	if (np.isscalar(SF)):
		if (SF < 0):
			SF = 0
	else:
		SF[SF < 0] = 0  ## Slope factors less than zero are completely shaded
	return( SF )

# angle of solar inclination from horizontal at solar noon [rad]
#lat: latitdue [rad]
#Jday: Julian date or day of the year [day]
def solarangle(lat, Jday):
	dec = declination(Jday)
	return(np.arcsin(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(0)))

def Solar(lat, Jday, Tx, Tn, albedo, forest, slope, aspect, printWarn):
	if (abs(lat) > np.pi/2) and printWarn:
		lat = lat * np.pi / 180
	convert = 1.0 # can convert to W/m2
	np_signif5 = np.vectorize(signif5)
	return np_signif5((1 - albedo) * (1 - forest) * transmissivity(Tx, Tn) * PotentialSolar(lat, Jday) * slopefactor(lat, Jday, slope, aspect) / convert)

def signif5(number):
	return float("{0:.5g}".format(number))
	# return float(number.item().format("0:.5g"))

def ifelse(condition, trueValue, falseValue):
	if (condition):
		return trueValue
	else:
		return falseValue

