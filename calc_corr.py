import pandas as pd
from glob import glob
from datetime import datetime
import seaborn as sns
import shapely.geometry
import numpy as np
from scipy.signal import argrelextrema
import os

year = 2017

base = 'calcs/pts/rgb/galv_plume/click/'
Files = glob(base + str(year) + '*.npz')

# stats will hold numbers for correlation
statsname = base + str(year) + '.csv'
if os.path.exists(statsname):
    stats = pd.read_csv(statsname, parse_dates=True, index_col=0)
else:
    stats = pd.DataFrame()

# load pts down Galv channel
ptsch = np.load(base + 'channel_pts.npz')['pts']
ptch = ptsch[-1]  # point at end and center of channel
## find line that goes along Galv channel (across-shelf)
# slope intercept form: y = mx+b
# another form: mx-y+b=0
m = (ptsch[-1,1] - ptsch[0,1])/(ptsch[-1,0] - ptsch[0,0])  # slope of line out galv channel
b = ptch[1] - m*ptch[0]  # y intercept
theta = np.arctan2(m,1)  # radians, angle from +x direction to along-channel (ebb) direction
## find line that is perpendicular and goes through ptch (along-shelf)
mperp = -1/m
bperp = ptch[1] - mperp*ptch[0]

def cart2chan(pts):
    '''convert from cartesian to channel coordinates.

    Shift by dx and dy to get origin at channel entrance,
    then rotate so that positive x' is along-channel/offshore/ebb tide
    and positive y' is across-channel/along-shore/upcoast.
    '''

    # shift origin to ptch (entrance to channel)
    pts -= ptch

    # rotate to be in x',y' coordinates
    if pts.ndim == 2:
        pts = pts[:,0]*np.cos(-theta) - pts[:,1]*np.sin(-theta), \
              pts[:,0]*np.sin(-theta) + pts[:,1]*np.cos(-theta)
    elif pts.ndim == 1:
        pts = pts[0]*np.cos(-theta) - pts[1]*np.sin(-theta), \
              pts[0]*np.sin(-theta) + pts[1]*np.cos(-theta)
    return pts




# File = Files[0]
for i, File in enumerate(Files):

    # check if this file already in stats, in which case move to next loop
    if datetime.strptime(File.split('/')[-1], '%Y-%m-%dT%H%M-galv.npz') in stats.index:
        continue

    # get pandas timestamp object from filename
    date = pd.Timestamp(File.split('/')[-1].split('.')[0][:-5])
    # date = datetime.strptime(File.split('/')[-1].split('.')[0], '%Y-%m-%dT%H%M-galv')
    # pre-date
    d1 = date - pd.Timedelta('28 days')


    # load plume points and calculate some metrics
    pts = np.load(File)['pts']
    # rotate to be in channel coordinates
    pts = cart2chan(pts)

    # make shapely 2d shape
    xy = zip(pts[0], pts[1])
    poly = shapely.geometry.Polygon(xy)

    # find centroid
    p = {}
    key = 'plume: centroid [along channel/offshore] [m]'
    p[key] = poly.centroid.x
    key = 'plume: centroid [across channel/upcoast] [m]'
    p[key] = poly.centroid.y
    # find area
    key = 'plume: area [m^2]'
    p[key] = poly.area
    # find extent acrosscoast/along channel
    key = 'plume: extent [along channel/offshore] [m]'
    p[key] = pts[0].max()
    # find max extent alongcoast/across channel. - is downcoast
    key = 'plume: extent [across channel/upcoast] [m]'
    p[key] = pts[1][np.argmax(abs(pts[1]))]

    if i == 0:  # adding columns first time
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=p)], axis=1)
    else:  # adding rows subsequently
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=p)], axis=0)

    ## get recent data
    # All in UTC
    baseurl = 'http://localhost'  # 'http://pong.tamu.edu'
    # Wind
    cols = ['East [m/s]', 'North [m/s]', 'Speed [m/s]', 'Dir from [deg T]']
    wind = pd.DataFrame(np.nan, columns=cols, index=pd.date_range(start=d1.normalize(), end=(date + pd.Timedelta('1 day')).normalize() - pd.Timedelta('15 min'), freq='15T'))

    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771341&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
    try:
        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
        windtemp = windtemp.resample('15min', base=0).mean()
        # save good values
        for col in cols:  # save column by column so it aligns properly
            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
    except:
        pass
    finally:
        if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771486&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
            try:
                windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                windtemp = windtemp.resample('15min', base=0).mean()
                # save good values
                for col in cols:  # save column by column so it aligns properly
                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
            except:
                pass
            finally:
                if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771013&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                    try:
                        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Conductivity [mS/cm]','Salinity', 'Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                        windtemp = windtemp.resample('15min', base=0).mean()
                        # save good values
                        for col in cols:  # save column by column so it aligns properly
                            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                    except:
                        pass
                    finally:
                        if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=42035&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                            try:
                                windtemp = pd.read_table(url, parse_dates=True, index_col=0).drop(['Gust [m/s]', 'AtmPr [MB]', 'AirT [deg C]', 'Dew pt [deg C]', 'WaterT [deg C]', 'RelH [%]', 'Wave Ht [m]', 'Wave Pd [s]'], axis=1)
                                windtemp = windtemp.resample('60min', base=0).interpolate().resample('15min', base=0).interpolate()
                                # save good values
                                for col in cols:  # save column by column so it aligns properly
                                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                            except:
                                pass
                            finally:
                                if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=B&table=met&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                    try:
                                        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['AirT [deg C]', 'AtmPr [MB]', 'Gust [m/s]', 'Comp [deg M]', 'Tx', 'Ty', 'PAR ', 'RelH [%]'], axis=1)
                                        windtemp = windtemp.resample('15min', base=0).interpolate()
                                        # save good values
                                        for col in cols:  # save column by column so it aligns properly
                                            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                                    except:
                                        pass
                                    finally:
                                        if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770971&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                            try:
                                                windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                                                windtemp = windtemp.resample('15min', base=0).mean()
                                                # save good values
                                                for col in cols:  # save column by column so it aligns properly
                                                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                                            except:
                                                pass
                                            finally:
                                                if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771972&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                                    try:
                                                        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                                                        windtemp = windtemp.resample('15min', base=0).mean()
                                                        # save good values
                                                        for col in cols:  # save column by column so it aligns properly
                                                            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                                                    except:
                                                        pass
                                                    finally:
                                                        if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                                            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770808&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                                            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                                            try:
                                                                windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                                                                windtemp = windtemp.resample('15min', base=0).mean()
                                                                # save good values
                                                                for col in cols:  # save column by column so it aligns properly
                                                                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                                                            except:
                                                                pass
                                                            finally:
                                                                if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                                                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770613&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                                                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                                                    try:
                                                                        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['AirT [deg C]', 'AtmPr [MB]', 'Conductivity [mS/cm]', 'Gust [m/s]', 'RelH [%]', 'Salinity', 'Water Level [m]', 'WaterT [deg C]'], axis=1)
                                                                        windtemp = windtemp.resample('15min', base=0).mean()
                                                                        # save good values
                                                                        for col in cols:  # save column by column so it aligns properly
                                                                            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()],col] = windtemp.loc[~windtemp['East [m/s]'].isnull(),col]
                                                                    except:
                                                                        pass
                                                                    finally:
                                                                        if (wind['East [m/s]'].isnull().sum() > 15) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                                                            raise Exception("some wind data missing")  # indices where need data still


    # tide
    try:  # g06010 or model
        try:  # g06010
            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=g06010&table=ports&Datatype=download&units=M&tz=UTC&model=False&datepicker='
            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
            tide = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
            tide = tide.resample('15min', base=0).interpolate()
            # don't want more than 2 hours of nan's in there
            if (tide.index[-1] - tide.index[0] < pd.Timedelta('7 days')) or (tide.isnull().sum() > 8):
                raise Exception("tide data too short in length")  # indices where need data still
        except:  # model
            base = 'https://tidesandcurrents.noaa.gov/noaacurrents/DownloadPredictions?fmt=csv&i=30min&d='
            suffix = '&r=2&tz=GMT&u=2&id=g06010_1&t=24hr&i=30min&threshold=leEq&thresholdvalue='
            # url to download data file starting the day before this sat data, week of data
            url2 = base + (date - pd.Timedelta('6 days')).strftime('%Y-%m-%d') + suffix
            tide = pd.read_csv(url2, parse_dates=True, index_col=0, names=['dates [UTC]', 'Along [cm/s]'], header=0)
            tide = tide.resample('15min', base=0).interpolate()
    except:  # derivative of water level
        # print('g06010 model failed')
        url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771341&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
        url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
        tide = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
        # calculate derivative
        tide['Along [cm/s]'] = (tide['Water Level [m]'].rolling(window=15, center=True).mean().diff().rolling(window=15, center=True).mean()*7e3).shift(periods=50)
        tide = tide.resample('15min', base=0).mean()
    # except:  # try BOLI depths
    #     url = 'https://waterdatafortexas.org/coastal/api/stations/BOLI/data/water_depth_nonvented?output_format=csv&binning=hour'
    #     tide = pd.read_csv(url, parse_dates=True, index_col=0, comment='#', header=0, names=['dates [UTC]', 'depth [m]'])
    #     # calculate derivative
    #     tide['Along [cm/s]'] = (tide['depth [m]'].rolling(window=5, center=True).mean().diff().rolling(window=5, center=True).mean()*500).shift(periods=-4)

    df = pd.concat([wind, tide], axis=1)

    # river discharge
    url = 'https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&cb_00065=on&format=rdb&site_no=08067070&period=&begin_date=' + (d1 - pd.Timedelta('1 day')).strftime('%Y-%m-%d') + '&end_date=' + date.strftime('%Y-%m-%d')
    river = pd.read_table(url, parse_dates=True, comment='#', header=1, usecols=[2,4], index_col=0, na_values=-999,
                     names=['Dates [UTC]', 'Trinity flow rate [m^3/s]']).tz_localize('US/Central', ambiguous='NaT').tz_convert('UTC').tz_localize(None)
    river = river*0.3048**3  # to m^3/s
    df = pd.concat([df, river.resample('15min', base=0).mean()], axis=1)

    # stop data at the satellite image time
    df = df[d1.strftime('%Y-%m-%d 00:00'):date.strftime('%Y-%m-%d %H:%M')]
    ##


    # store numbers in stats for this image
    dt = 15*60  # 15 minute time step, in seconds


    # run river metrics for 28 days max
    r = {}
    for nday in np.arange(1,14):
        for delay in np.arange(0,14):  # number of days delay
            key = 'river: int ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m^3]'
            # start date is sat image date minus delay minus nday
            dst = date - pd.Timedelta(str(delay) + ' days') - pd.Timedelta(str(nday) + 'days')
            # end date is sat image date minus delay
            dend = date - pd.Timedelta(str(delay) + ' days')
            # print(dst)
            r[key] = df['Trinity flow rate [m^3/s]'][dst:dend].sum()*dt
    if i == 0:  # adding columns first time
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=r)], axis=1)
    else:  # adding rows subsequently
        for key in r.keys():
            stats[key].loc[date] = r[key]
        # stats = pd.concat([stats, pd.DataFrame(index=[date], data=r)], axis=0)

    # run wind metrics for 28 days max
    w = {}
    for nday in np.arange(1,14):
        for delay in np.arange(0,14):  # number of days delay
            # start date is sat image date minus delay minus nday
            dst = date - pd.Timedelta(str(delay) + ' days') - pd.Timedelta(str(nday) + 'days')
            # end date is sat image date minus delay
            den = date - pd.Timedelta(str(delay) + ' days')

            # mean east/north, std east/north, mean/std speed, mean/std dir
            keye = 'wind: east mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[keye] = np.nanmean(df['East [m/s]'][dst:den])
            keyn = 'wind: north mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[keyn] = np.nanmean(df['North [m/s]'][dst:den])
            key = 'wind: speed mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[key] = np.nanmean(df['Speed [m/s]'][dst:den])
            key = 'wind: dir mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[key] = np.arctan2(w[keyn], w[keye])

    if i == 0:  # adding columns first time
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=w)], axis=1)
    else:  # adding rows subsequently
        for key in w.keys():
            stats[key].loc[date] = w[key]

    # run tidal metrics
    # calculate time since previous local maximum and time since +/- switch
    # integrate from previous local max and since last +/- switch
    # time since previous local maximum only makes sense if after time since +/- switch
    # +/- switch catches whether it switched from flood to ebb (typical) or opposite
    t = {}

    # index of start of present tide
    ist = np.where(np.sign(df['Along [cm/s]']) == -np.sign(df['Along [cm/s]'][-1]))[0][-1]
    # index of most recent local maximum
    imax = argrelextrema(df['Along [cm/s]'].values, np.greater)[0][-1]

    # ALSO SAVE TIME INTO CYCLE OR SOMETHING?
    # case ebb tide
    if df['Along [cm/s]'][-1] < 0:
        #  case no small ebb tide (just count from start of ebb)
        if imax < ist:
            for hoursum in np.arange(1,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after start of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before ebb tide start
                    if dst < df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = np.nansum(df['Along [cm/s]'][dst:den])*dt/100.

        # case yes small ebb tide before sat image on this ebb tide
        elif imax > ist:
            for hoursum in np.arange(1,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after mini ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before local max
                    if dst < df.index[imax]:
                        t[key] = np.nan
                    else:
                        t[key] = np.nansum(df['Along [cm/s]'][dst:den])*dt/100.

    # case flood tide. time lag from sat image date but nan out if calc goes into flood
    elif df['Along [cm/s]'][-1] > 0:
        # index of start of previous tide
        istp = np.where(np.sign(df['Along [cm/s]'][:ist]) == -np.sign(df['Along [cm/s]'][ist]))[0][-1]
        #  case no small ebb tide (just count from start of ebb)
        if imax < istp:
            for hoursum in np.arange(1,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after start and before end of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before start of ebb tide
                    # or if end is during flood tide (after end of ebb)
                    if dst < df.index[istp] or den > df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = np.nansum(df['Along [cm/s]'][dst:den])*dt/100.

        # case yes small ebb tide before sat image on previous ebb tide
        elif imax > istp:
            for hoursum in np.arange(1,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after mini ebb tide and before end of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before local max mini ebb tide
                    # or if end is during flood tide (after end of ebb)
                    if dst < df.index[imax] or den > df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = np.nansum(df['Along [cm/s]'][dst:den])*dt/100.

    if i == 0:  # adding columns first time
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=t)], axis=1)
    else:  # adding rows subsequently
        for key in t.keys():
            if key not in stats.keys():
                stats[key] = np.nan
                stats[key].loc[date] = t[key]
            else:
                stats[key].loc[date] = t[key]

    stats.to_csv(statsname)



# correlate some things
#
import scipy.stats
# keys for plume metrics
pkeys = [key for key  in stats.keys() if 'plume' in key]
# keys for tide metrics
tkeys = [key for key  in stats.keys() if 'tide' in key]
# keys for river metrics
rkeys = [key for key  in stats.keys() if 'river' in key]
# keys for wind metrics
wkeys = [key for key  in stats.keys() if 'wind' in key]
# keys for all mechanisms
mkeys = tkeys + rkeys + wkeys

r = pd.DataFrame(index=pkeys, columns=mkeys); p = pd.DataFrame(index=pkeys, columns=mkeys)
for pkey in pkeys:
    for mkey in mkeys:
        # indices where not nans
        inds = np.where(~stats[mkey].isnull())[0]
        rtemp, ptemp = scipy.stats.pearsonr(stats[pkey].iloc[inds], stats[mkey].iloc[inds])

        r[mkey].loc[pkey] = rtemp
        p[mkey].loc[pkey] = ptemp


# pull out largest correlations and plot them
for pkey in r.index:
    # indices of max correlation coefficients, in order
    inds = np.argsort(abs(r.loc[pkey].values))[::-1]
    for ind in inds:
        mkey = r.columns[ind]
        if (abs(r.loc[pkey,mkey]) < 0.5) or (p.loc[pkey,mkey] > 0.1) or (stats[mkey].isnull().sum() > len(stats)/2):
            break
        print(r.loc[pkey,mkey], p.loc[pkey,mkey])
        plt.figure()
        sns.regplot(x=mkey, y=pkey, data=stats, scatter_kws={'s': 50});

# plot plume characteristics
fig, axes = plt.subplots(1,3, sharey=True, figsize=(14,4))
stats[pkeys[0]].plot(kind='hist', bins=20, ax=axes[0])#, label='area [m$^2$]', legend=True)
axes[0].set_xlabel('area [m$^2$]')

stats[pkeys[1]].plot(kind='hist', bins=20, ax=axes[1], alpha=0.7, label='centroid loc [upcoast]', legend=True)
stats[pkeys[2]].plot(kind='hist', bins=20, ax=axes[1], alpha=0.7, label='centroid loc [offshore]', legend=True)
axes[1].set_xlabel('distance [m]')

stats[pkeys[3]].plot(kind='hist', bins=20, ax=axes[2], alpha=0.7, label='extent [upcoast]', legend=True)
stats[pkeys[4]].plot(kind='hist', bins=20, ax=axes[2], alpha=0.7, label='extent [offshore]', legend=True)
axes[2].set_xlabel('distance [m]')

fig.savefig('figures/rgb/galv_plume/click/plume_metrics.pdf', bbox_inches='tight')

# Plot example conditions
from matplotlib.dates import date2num
dst = tide.index[0]; den = tide.index[-1]
idx = date2num(pd.to_datetime(wind[dst:den].index).to_pydatetime())
ddt = 1
width=.1

fig, axes = plt.subplots(3,1, sharex=True, figsize=(14,6))
axes[0].plot(idx, tide, lw=2, color='k')
axes[0].set_ylabel('Along-channel\nspeed [cm$\cdot$s$^{-1}$]')
axes[0].grid(which='major', lw=1.5, color='k', alpha=0.1)

axes[1].quiver(idx, np.zeros(len(wind[dst:den])),
               wind[dst:den]['East [m/s]'], wind[dst:den]['North [m/s]'],
               headaxislength=0, headlength=0, width=width, units='y',
               scale_units='y', scale=1)
axes[1].set_ylabel('Wind [m$\cdot$s$^{-1}$]')
axes[1].set_ylim(-10,10)
axes[1].grid(which='major', lw=1.5, color='k', alpha=0.1)

axes[2].plot(idx, river[dst:den], lw=2, color='k')
axes[2].set_xlim(idx[0], idx[-1])
axes[2].set_ylabel('Trinity river\ndischarge [m$^3$]')
import matplotlib as mpl
# minor = mpl.dates.HourLocator(byhour=np.arange(0,24,6))
# ax.xaxis.set_minor_locator(minor)
major = mpl.dates.HourLocator(byhour=np.arange(0,24,24))
axes[2].xaxis.set_major_locator(major)
axes[2].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %d'))
axes[2].grid(which='major', lw=1.5, color='k', alpha=0.1)

fig.savefig('figures/rgb/galv_plume/click/example_mechanisms.pdf', bbox_inches='tight')
