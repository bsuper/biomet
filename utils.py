import numpy as np
import pandas as pd
import scipy.io as sio
import datetime

HALF_HOURLY_SCALE = 1800. / 1e6 # each sample is half-hourly, so multiply by 1800
CARBON_SCALE = 1800. * 10e-6 * 12

###############################################
# LITTLE HELPER METHODS
###############################################

def days_elapsed(year, doy, begin_date=datetime.date(2000, 1, 1)):
    curr_date = datetime.date(year,1,1) + datetime.timedelta(doy - 1)
    return (curr_date - begin_date).days

def create_days_elapsed_series(year, doy):
    days_elapsed_lst = []
    for i in range(year.shape[0]):
        days_elapsed_lst.append(days_elapsed(year[i], doy[i]))
    return np.array(days_elapsed_lst)

###############################################
# PROCESS VARIOUS DATA FILES
###############################################

def process_cimis(filename, interpolate_missing=True):
    """
    Process Cimis Twitchell Island Tower Data

    The file is aggrgated from CIMIS hourly files (2001/1-2016/4).
    QC flags were left out for now, and new units were adopted here.

    ETo 		(mm/day)
    Precip		(mm/day)
    Sol.Rad		(MJ/m2/day)
    VPD		(kPa)
    Air.Temp	(deg C)
    Wind.Speed	(m/s)
    Soil.Temp	(deg C)
    """
    print "Processing {0}...".format(filename)
    df = pd.read_csv(filename)
    df2 = {}
    df2['year'] = df['Date'].apply(lambda x: int(x[:4]))
    df2['doy'] = df['Jul']

    # potential evapotranspiration (PET)
    df2['PET'] = df['ETo']
    # precipitation
    df2['precip'] = df['Precip']
    # shortwave radiation
    df2['sw_in'] = df["Sol.Rad"]
    # vapor pressure deficit
    df2['VPD'] = df["VPD"]
    # air Temp
    df2['air_temp'] = df["Air.Temp"]
    # wind speed
    df2["wind_speed"] = df["Wind.Speed"]
    # soil temp
    df2["soil_temp"] = df["Soil.Temp"]

    df2 = pd.DataFrame(df2)
    if interpolate_missing:
        df2 = interpolate_missing_values(df2)
    return df2

def process_modis_reflectance_veg_index(filename, prefix="", interpolate_missing=True, method='linear', order=1):
    """
    Process MOD13Q1/MCD43A4 Reflectance and Vegetation Index Data

    MOD13Q1: 250m 16-day Relectance & Vegetation Indices
    https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13q1
    YR: year
    DOY: julian day (specified as the actual day of composite)
    NDVI: normalized difference vegetation index
    EVI: enhanced vegetation index
    bnd3.ref: band 3 reflectance (459 - 479nm) blue
    bnd7.ref: band 7 reflectance (2105 - 2155nm) MIR
    bnd2.ref: band 2 reflectance (841 - 876nm) NIR
    bnd1.ref: band 1 reflectance (620 - 670nm) red
    QC: quality flag (0 as good data)
    LSWI2: Land Surface Water Index (2130nm)

    MCD43A4: 500m 8-day Nadir BRDF-Adjusted Relectance & Vegetation Indices
    YR: year
    DOY: julian day (specified as the center day of composite period)
    bnd1.ref: band 1 reflectance (620 - 670nm) red
    bnd2.ref: band 2 reflectance (841 - 876nm) NIR
    bnd3.ref: band 3 reflectance (459 - 479nm) blue
    bnd4.ref: band 4 reflectance (545 - 565nm) green
    bnd5.ref: band 5 reflectance (1230 - 1250nm)
    bnd6.ref: band 6 reflectance (1628 - 1652nm) MIR
    bnd7.ref: band 7 reflectance (2105 - 2155nm) MIR
    LSWI1: Land Surface Water Index (1680nm)
    LSWI2: Land Surface Water Index (2130nm)
    NDVI: Normalized Difference Vegetation Index
    EVI: Enhanced Vegetation Index

    LANDSAT
    YR
    DOY
    NDVI
    EVI
    LSWI2
    """
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    df = pd.read_csv(filename)
    # select only good data rows
    if "QC" in df.columns:
        df = df.loc[df["QC"] == 0]
    df2 = {}
    df2['doy'] = df['DOY']
    df2["year"] = df["YR"]
    df2[prefix + "ndvi"] = df["NDVI"]
    df2[prefix + "evi"] = df["EVI"]
    df2[prefix + "lswi2"] = df["LSWI2"]
    # LANDSAT doesn't have these
    # df2[prefix + "bnd1"] = df["bnd1.ref"]
    # df2[prefix + "bnd2"] = df["bnd2.ref"]
    # df2[prefix + "bnd3"] = df["bnd3.ref"]
    # df2[prefix + "bnd7"] = df["bnd7.ref"]
    df2 = pd.DataFrame(df2)
    df2 = df2.sort_values(["year", "doy"]).reset_index(drop=True)
    if interpolate_missing:
        df2 = interpolate_missing_values(df2, method=method, order=order)
    return df2

def process_modis_lst_emissivity(filename, prefix="", interpolate_missing=True):
    """
    Processes MODIS MOD11A2

    MOD11A2: 1km 8-day Land Surface Temperature & Emissivity
    https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod11a2
    YR: year
    DOY: julian day (specified as the center day of composite period)
    LST.day: daytime land surface temperature (deg C)
    LST.night: nighttime land surface temperature (deg C)
    QC.day: quality flag for LST.day (0 as good data)
    QC.night: quality flag for LST.night (0 as good data)
    """
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    df = pd.read_csv(filename)
    df.loc[df["QC.day"] != 0, "LST.day"] = np.nan
    df.loc[df["QC.night"] != 0, "LST.night"] = np.nan
    del df["QC.day"]
    del df["QC.night"]
    df.rename(columns={"YR": "year", "DOY": "doy", "LST.day": prefix + "LST.day",
                        "LST.night": prefix + "LST.night"}, inplace=True)
    if interpolate_missing:
        df = interpolate_missing_values(df)
    return df

def process_tower(filename, prefix="", interpolate_missing=True):
    """
    Process Tower Data
    """
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    data = sio.loadmat(filename)
    d = data['data'][0][0]
    df = {}
    df["doy"] = d['DOY'][:,0]
    df["year"] = d['year'][:,0]
    # CO2 flux (umol m-2 s-1) => gC / m^2 / day
    df[prefix + "co2_gf"] = d['wc_gf'][:,0] * CARBON_SCALE
    # CH4 flux (nmol m-2 s-1) => mgC / m^2 / day
    df[prefix + "ch4_gf"] = d['wm_gf'][:,0] * CARBON_SCALE
    # ER => gC / m^2 / day
    df[prefix + "er"] = d['er_ANNnight'][:,0] * CARBON_SCALE
    # gross primary productivity => gC / m^2 / day
    df[prefix + "gpp"] = d['gpp_ANNnight'][:,0] * CARBON_SCALE
    # latent heat flux (W m-2) => Mj / m^2 / day
    df[prefix + "le"] = d['LE_gf'][:,0] * HALF_HOURLY_SCALE
    # sensible heat flux (W m-2) => Mj / m^2 / day
    df[prefix + "h"] = d['H_gf'][:,0] * HALF_HOURLY_SCALE
    # net radiation => Mj / m^2 / day
    df[prefix + "RNET"] = d["RNET"][:,0] * HALF_HOURLY_SCALE

    df = pd.DataFrame(df)
    grouped = df.groupby(["year", "doy"]).aggregate(np.sum).reset_index()
    if interpolate_missing:
        grouped = interpolate_missing_values(grouped)
    return grouped

def process_tower_lwi(filename, prefix="", interpolate_missing=True):
    """
    Process Tower LWI data

    [2016-04-14_daily_TOWER_LW.csv]
    The data stream is derived from West Pond (Tw1) and Shermand Island (Snd) sites.

    LW_IN.wp	(MJ/m2/day)
    LW_IN.si	(MJ/m2/day)
    LW_IN		(MJ/m2/day)
    """
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    df = pd.read_csv(filename)
    del df["Time.id"]
    df.rename(columns={"YEAR": "year", "DOY": "doy", "LW_IN.wp": prefix + "LW_IN.wp",
                        "LW_IN.si": prefix + "LW_IN.si", "LW_IN": prefix + "LW_IN"}, inplace=True)
    if interpolate_missing:
        df = interpolate_missing_values(df)
    return df

###############################################
# INTERPOLATE MISSING DATA
###############################################

def interpolate_missing_values(df, method='spline', order=2):
    min_year, min_doy = df["year"].min(), df[df["year"] == df["year"].min()]["doy"].min()
    max_year, max_doy = df["year"].max(), df[df["year"] == df["year"].max()]["doy"].max()
    missing_rows = []
    year, doy = min_year, min_doy
    while year <= max_year:
        if year == max_year and doy > max_doy:
            break
        row_exists = sum((df["year"] == year) & (df["doy"] == doy)) == 1
        if not row_exists:
            new_row = dict(zip(df.columns, [np.nan] * len(df.columns)))
            new_row["year"] = year
            new_row["doy"] = doy
            missing_rows.append(new_row)
        if (year % 4 == 0 and doy == 366) or (year % 4 != 0 and doy == 365):
            year += 1
            doy = 1
        else:
            doy += 1
    df = pd.concat((df, pd.DataFrame(missing_rows)))
    return df.sort_values(["year", "doy"]).reset_index(drop=True).interpolate(method=method, order=order)

###############################################
# MERGE DATAFRAMES TOGETHER
###############################################

def merge_dataframes(dfs):
    df = reduce(lambda left, right: pd.merge(left, right, on=['year', 'doy']), dfs)
    return df
