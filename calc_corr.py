import pandas as pd
from glob import glob
from datetime import datetime
import seaborn as sns
import shapely.geometry
import numpy as np
from scipy.signal import argrelextrema

year = 2017

base = 'calcs/pts/rgb/galv_plume/click/'
Files = glob(base + str(year) + '*.npz')

# stats will hold numbers for correlation
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
    windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
    windtemp = windtemp.resample('15min', base=0).mean()
    # save good values
    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
    if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
        url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771486&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
        url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
        windtemp = windtemp.resample('15min', base=0).mean()
        # save good values
        wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
        if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771013&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
            windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Conductivity [mS/cm]','Salinity', 'Gust [m/s]', 'AtmPr [MB]', 'Water Level [m]', 'AirT [deg C]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
            windtemp = windtemp.resample('15min', base=0).mean()
            # save good values
            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
            if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=42035&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                windtemp = pd.read_table(url, parse_dates=True, index_col=0).drop(['Gust [m/s]', 'AtmPr [MB]', 'AirT [deg C]', 'Dew pt [deg C]', 'WaterT [deg C]', 'RelH [%]', 'Wave Ht [m]', 'Wave Pd [s]'], axis=1)
                windtemp = windtemp.resample('60min', base=0).interpolate().resample('15min', base=0).interpolate()
                # save good values
                wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=B&table=met&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                    windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['AirT [deg C]', 'AtmPr [MB]', 'Gust [m/s]', 'Comp [deg M]', 'Tx', 'Ty', 'PAR ', 'RelH [%]'], axis=1)
                    windtemp = windtemp.resample('15min', base=0).interpolate()
                    # save good values
                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                    if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                        url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770971&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                        url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                        windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                        windtemp = windtemp.resample('15min', base=0).mean()
                        # save good values
                        wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                        if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8771972&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                            windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                            windtemp = windtemp.resample('15min', base=0).mean()
                            # save good values
                            wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                            if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770808&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['Water Level [m]', 'Gust [m/s]', 'AirT [deg C]', 'AtmPr [MB]', 'RelH [%]', 'WaterT [deg C]'], axis=1)
                                windtemp = windtemp.resample('15min', base=0).mean()
                                # save good values
                                wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                                if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                    url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=8770613&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                                    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                                    windtemp = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999).drop(['AirT [deg C]', 'AtmPr [MB]', 'Conductivity [mS/cm]', 'Gust [m/s]', 'RelH [%]', 'Salinity', 'Water Level [m]', 'WaterT [deg C]'], axis=1)
                                    windtemp = windtemp.resample('15min', base=0).mean()
                                    # save good values
                                    wind.loc[windtemp.index[~windtemp['East [m/s]'].isnull()]] = windtemp[~windtemp['East [m/s]'].isnull()]
                                    if (wind['East [m/s]'].isnull().sum() > 3) or (wind.index[-1] < date) or (wind.index[0] > d1):
                                        raise Exception("some wind data missing")  # indices where need data still


    # tide
    try:  # g06010 or model
        try:  # g06010
            url = baseurl + '/tabswebsite/subpages/tabsquery.php?Buoyname=g06010&table=ports&Datatype=download&units=M&tz=UTC&model=False&datepicker='
            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
            tide = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
            tide = tide.resample('15min', base=0).interpolate()
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
                     names=['Dates [UTC]', 'Trinity flow rate [m^3/s]']).tz_localize('US/Central', ambiguous='infer').tz_convert('UTC').tz_localize(None)
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
            w[keye] = df['East [m/s]'][dst:den].mean()
            keyn = 'wind: north mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[keyn] = df['North [m/s]'][dst:den].mean()
            key = 'wind: speed mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
            w[key] = df['Speed [m/s]'][dst:den].mean()
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
            for hoursum in np.arange(0,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after start of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before ebb tide start
                    if dst < df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = df['Along [cm/s]'][dst:den].sum()*dt/100.

        # case yes small ebb tide before sat image on this ebb tide
        elif imax > ist:
            for hoursum in np.arange(0,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after mini ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before local max
                    if dst < df.index[imax]:
                        t[key] = np.nan
                    else:
                        t[key] = df['Along [cm/s]'][dst:den].sum()*dt/100.

    # case flood tide. time lag from sat image date but nan out if calc goes into flood
    elif df['Along [cm/s]'][-1] > 0:
        # index of start of previous tide
        istp = np.where(np.sign(df['Along [cm/s]'][:ist]) == -np.sign(df['Along [cm/s]'][ist]))[0][-1]
        #  case no small ebb tide (just count from start of ebb)
        if imax < istp:
            for hoursum in np.arange(0,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after start and before end of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before start of ebb tide
                    # or if end is during flood tide (after end of ebb)
                    if dst < df.index[istp] or den > df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = df['Along [cm/s]'][dst:den].sum()*dt/100.

        # case yes small ebb tide before sat image on previous ebb tide
        elif imax > istp:
            for hoursum in np.arange(0,7):  # number of hours to sum over
                for hourdelay in np.arange(0,7):  # number of hours before sat image to lag sum
                    key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after mini ebb tide and before end of ebb tide [m]'
                    dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                    den = date - pd.Timedelta(str(hourdelay) + ' hours')
                    # remove if calc start is before local max mini ebb tide
                    # or if end is during flood tide (after end of ebb)
                    if dst < df.index[imax] or den > df.index[ist]:
                        t[key] = np.nan
                    else:
                        t[key] = df['Along [cm/s]'][dst:den].sum()*dt/100.

    if i == 0:  # adding columns first time
        stats = pd.concat([stats, pd.DataFrame(index=[date], data=t)], axis=1)
    else:  # adding rows subsequently
        for key in t.keys():
            if key not in stats.keys():
                stats[key] = t[key]
            else:
                stats[key].loc[date] = t[key]

    stats.to_csv('/'.join(File.split('/')[:-1] + [str(year)]) + '.csv')



# # correlate some things
# sns.regplot(x=, y='', data=df, ax=ax, scatter_kws={'s': 50});
