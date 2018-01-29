import pandas as pd
from glob import glob
from datetime import datetime
import seaborn as sns

base = 'calcs/pts/rgb/galv_plume/click/'
Files = glob(base + '*.npz')

# stats will hold numbers for correlation
stats = pd.DataFrame()


File = Files[0]

# get pandas timestamp object from filename
date = pd.Timestamp(File.split('/')[-1].split('.')[0][:-5])
# date = datetime.strptime(File.split('/')[-1].split('.')[0], '%Y-%m-%dT%H%M-galv')
# pre-date
d1 = date - pd.Timedelta('28 days')

# load pts down Galv channel
ptsch = np.load(base + 'channel_pts.npz')['pts']
ptch = ptsch[-1]  # point at end and center of channel
## find line that goes along Galv channel (across-shelf)
# slope intercept form: y = mx+b
# another form: mx-y+b=0
m = (ptsch[-1,1] - ptsch[0,1])/(ptsch[-1,0] - ptsch[0,0])  # slope of line out galv channel
b = ptch[1] - m*ptch[0]  # y intercept
## find line that is perpendicular and goes through ptch (along-shelf)
mperp = -1/m
bperp = ptch[1] - mperp*ptch[0]


# load plume points and calculate some metrics
pts = np.load(File)['pts']

# find center of points [x0, y0]
pt0 = [pts[:,0].mean(), pts[:,1].mean()]
# measure of size of clicked area in meters (?)
std0 = np.sqrt(pts[:,0].std()**2 + pts[:,1].std()**2)
# calculated as distance between pt0 and the line:
# # along-coast distance from along-channel line
# alongcoast = abs(m*pt0[0] - pt0[1] + b)/np.sqrt(m**2 + 1)
# across-coast distance from across-channel line
# # which will always be positive (across-shelf away from the coast)
# acrosscoast = abs(mperp*pt0[0] - pt0[1] + bperp)/np.sqrt(mperp**2 + 1)
# # angle to get along coast distance
# phi = np.arccos(acrosscoast/dist)
# np.tan(phi)*acrosscoast

# distance between jetties and pt0, also hypotheneuse
dist = np.sqrt((ptch[0] - pt0[0])**2 + (ptch[1] - pt0[1])**2)
# angle to get across coast distance
phi = np.arccos(alongcoast/dist)

#
# https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
alongcoast = (pt0[0] - ptch[0])*(0 - ptch[1]) - (pt0[1] - ptch[1])*(acrosscoast - ptch[0])

# CHANGE ALL COORDS TO RELATIVE TO JETTIES, THEN CALCULATE METRICS
# area, extent

# theta = np.arctan2(m,1)  # radians, angle from +x direction to along-channel (ebb) direction
# vecch = [np.cos(theta), np.sin(theta)]  # points along channel in ebb direction
# # rotate from pt0 in cartesian to channel coords
# vecrot0 = pt0[0]*np.cos(theta) - pt0[1]*np.sin(theta), \
#          pt0[0]*np.sin(theta) + pt0[1]*np.cos(theta)


# # across-shelf (along-channel, ebb positive) distance from jetties
# vec = pt0 - ptch  # m, vector from center of end of channel to center of points
# vecrot0 = vec[0]*np.cos(theta) - vec[1]*np.sin(theta), \
#          vec[0]*np.sin(theta) + vec[1]*np.cos(theta)
# ptrot0 = vecrot0 + ptch
# m(x2-xch)=(y2-ych)
# y = mx+b
# mx - y + b = 0

## get recent data
# All in UTC
# Wind
try:
    try:
        try:
            try:  # 8771341
                url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=8771341&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                wind = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
                wind = wind.resample('15min', base=0).mean()
                if wind['East [m/s]'].isnull().all():
                    raise Exception("some data present with buoy 8771341 but not wind data")
                if wind['East [m/s]'].isnull().any():
                    raise Exception("some wind data missing")
            except:  # 8771486
                url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=8771486&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
                url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
                wind = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
                wind = wind.resample('15min', base=0).mean()
                if wind['East [m/s]'].isnull().all():
                    raise Exception("some data present with buoy 8771486 but not wind data")
        except:  # 8771013
            url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=8771013&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
            url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
            wind = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
            wind = wind.resample('15min', base=0).mean()
            if wind['East [m/s]'].isnull().all():
                raise Exception("some data present with buoy 8771013 but not wind data")
    except:  # 42035
        url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=42035&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
        url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
        wind = pd.read_table(url, parse_dates=True, index_col=0)#, na_values=-999)
        wind = wind.resample('60min', base=0).interpolate().resample('15min', base=0).interpolate()
        if wind['East [m/s]'].isnull().all():
            raise Exception("some data present with buoy 42035 but not wind data")
except:  # B
    url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=B&table=met&Datatype=download&units=M&tz=UTC&model=False&datepicker='
    url += d1.strftime('%Y-%m-%d') + '+-+' + date.strftime('%Y-%m-%d')
    wind = pd.read_table(url, parse_dates=True, index_col=0, na_values=-999)
    wind = wind.resample('15min', base=0).interpolate()
    if wind['East [m/s]'].isnull().all():
        raise Exception("some data present with buoy B but not wind data")

# tide
try:  # g06010 or model
    try:  # g06010
        url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=g06010&table=ports&Datatype=download&units=M&tz=UTC&model=False&datepicker='
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
    url = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?Buoyname=8771341&table=nos&Datatype=download&units=M&tz=UTC&model=False&datepicker='
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
        r[key] = df['Trinity flow rate [m^3/s]'][dst:dend].sum()*dt}

stats.append(pd.DataFrame(index=[date], data=r))


# run wind metrics for 28 days max
w = {}
for nday in np.arange(1,14):
    for delay in np.arange(0,14):  # number of days delay
        # start date is sat image date minus delay minus nday
        dst = date - pd.Timedelta(str(delay) + ' days') - pd.Timedelta(str(nday) + 'days')
        # end date is sat image date minus delay
        dend = date - pd.Timedelta(str(delay) + ' days')

        # mean east/north, std east/north, mean/std speed, mean/std dir
        keye = 'wind: east mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
        w[keye] = df['East [m/s]'][dst:den].mean()
        keyn = 'wind: north mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
        w[keyn] = df['North [m/s]'][dst:den].mean()
        key = 'wind: speed mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
        w[key] = df['Speed [m/s]'][dst:den].mean()
        key = 'wind: dir mean ' + str(nday) + ' days up to ' + str(delay) + ' days before date [m/s]'
        w[key] = np.arctan2(w[keyn], w[keye])

stats = pd.concat([stats, pd.DataFrame(index=[date], data=w)], axis=1)

# stats.append(pd.DataFrame(index=[date], data=w))



# run tidal metrics
# calculate time since previous local maximum and time since +/- switch
# integrate from previous local max and since last +/- switch
# time since previous local maximum only makes sense if after time since +/- switch
# +/- switch catches whether it switched from flood to ebb (typical) or opposite
t = {}
from scipy.signal import argrelextrema

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
                key = 'tide: int over ' + str(hoursum) + ' hours, delayed ' + str(hourdelay) + ' hours before sat image but after mini ebb tide and before end of ebb tide[m]'
                dst = date - pd.Timedelta(str(hourdelay) + ' hours') - pd.Timedelta(str(hoursum) + ' hours')
                den = date - pd.Timedelta(str(hourdelay) + ' hours')
                # remove if calc start is before local max mini ebb tide
                # or if end is during flood tide (after end of ebb)
                if dst < df.index[imax] or den > df.index[ist]:
                    t[key] = np.nan
                else:
                    t[key] = df['Along [cm/s]'][dst:den].sum()*dt/100.

stats = pd.concat([stats, pd.DataFrame(index=[date], data=t)], axis=1)
# stats.append(pd.DataFrame(index=[date], data=t))




# # correlate some things
# sns.regplot(x=, y='', data=df, ax=ax, scatter_kws={'s': 50});
