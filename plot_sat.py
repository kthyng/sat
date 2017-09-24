'''
Plot satellite data in the northwestern Gulf of Mexico.

Usage:
plot_sat.py [-h] year "var" "area"

Example usage:
run plot_sat 2014 "ci" "wgom" "wgom"
run plot_sat 2017 "ci" "gcoos" "txla"
run plot_sat 2017 "ci" "wgom" "txla" --plotsource 'yes'
run plot_sat 2017 "rgb" "galv" "galv_plume" --plotshipping 'yes' --plottide 'yes' --scale 5

** not CI for GCOOS before 2016 -- GCOOS is appearing lighter for CI in 2016 and 2017 than in WGOM
'''

import matplotlib as mpl
mpl.use("Agg") # set matplotlib to use the backend that does not require a windowing system
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
# try:
#     from StringIO import StringIO
# except ImportError:
#     from io import StringIO
import io
import cmocean.cm as cmo
try:
    from bs4 import BeautifulSoup
except:
    from BeautifulSoup import BeautifulSoup
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.path import Path
import pdb
from datetime import datetime, timedelta
import argparse
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
import os
import cartopy
ccrs = cartopy.crs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import xarray as xr
import cartopy.io.shapereader as shpreader
import pandas as pd
from matplotlib.dates import date2num

# Input arguments: year and what to plot
parser = argparse.ArgumentParser()
parser.add_argument('year', type=int, help='What year to plot')
parser.add_argument('var', type=str, help='What field to plot: "sst" (sea surface temp) or "oci" (chlorophyll-a with good correction algorithm) or "ci" (chlorophyll-a with no sun glint) or "rgb" (color) or "CHL" (chlorophyll-a)')
parser.add_argument('area', type=str, help='Area getting data from to plot: "gcoos" (full Gulf of Mexico) or "wgom" (western Gulf of Mexico) or "galv"')
parser.add_argument('figarea', type=str, help='What area in Gulf to plot data in: "wgom" (western Gulf of Mexico) or "txla" (TXLA domain) or "galv_plume"')
parser.add_argument('--plotbathy', default=None, type=str, help='Plot bathymetry from NWGOM numerical model or not (any string input will cause to plot)')
parser.add_argument('--plotdomain', default=None, type=str, help='Plot numerical domain from NWGOM numerical model or not (any string input will cause to plot)')
parser.add_argument('--plotshipping', default=None, type=str, help='Plot shipping lanes or not (any string input will cause to plot)')
parser.add_argument('--plottide', default=None, type=str, help='Plot tides or not (any string input will cause to plot)')
parser.add_argument('--plotsource', default=None, type=str, help='Plot sat data link or not (any string input will cause to plot)')
parser.add_argument('--scale', default=None, type=int, help='To plot reference scale, input the reference km value as as int.')
args = parser.parse_args()

mpl.rcParams.update({'font.size': 11})

# any input for plotbathy will cause bathy to plot
if args.plotbathy is not None:
    plotbathy = True
else:
    plotbathy = False
if args.plotdomain is not None:
    plotdomain = True
else:
    plotdomain = False
if args.plotshipping is not None:
    plotshipping = True
else:
    plotshipping = False
if args.plottide is not None:
    plottide = True
else:
    plottide = False
if args.plotsource is not None:
    plotsource = True
else:
    plotsource = False
if args.scale is not None:
    plotscale = True
    scale = args.scale
else:
    plotscale = False

# loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_hindcast_agg'
loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/txla_nesting6_grid/txla_grd_v4_new.nc'
grid = xr.open_dataset(loc)
hlevs = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]  # isobath contour depths

merc = ccrs.Mercator(central_longitude=-85.0)
pc = ccrs.PlateCarree()
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

if plotshipping:
    fname = 'data/fairway/fairway.shp'
    shape_feature = cartopy.feature.ShapelyFeature(shpreader.Reader(fname).geometries(),
                                    cartopy.crs.PlateCarree(), facecolor='none')

# data locations
# Galveston Bay Entrance, North Jetty, TX - Station ID: 8771341
# https://tidesandcurrents.noaa.gov/stationhome.html?id=8771341
wndbc = [-(94+43.5/60), 29+21.4/60]
# TABS B
wtabs = [-(94+53.944/60), 28+58.939/60]
# currents: g06010
# https://tidesandcurrents.noaa.gov/cdata/StationInfo?id=g06010#metadata
tndbc = [-(94+44.450/60), 29+20.533/60]

# Satellite data is in equidistant cylindrical projection which is just lon/lat
# get this info from "[X] Region and Data Description" on page
if args.area == 'gcoos':
    dataextent = [-98, -79, 18, 31]  # extend of data
    figextent = [-98, -88, 18, 30.5]  # extent of fig
    lon = np.linspace(-98, -79, 2090)
    lat = np.linspace(18, 31, 1430)
elif args.area == 'wgom':
    dataextent = [-98, -90, 18, 30]
    lon = np.linspace(-98, -90, 880)
    lat = np.linspace(18, 30, 1320)
elif args.area == 'galv':
    dataextent = [-96.5, -93.5, 27.8, 29.8]
    lon = np.linspace(-96.5, -93.5, 880)
    lat = np.linspace(27.8, 29.8, 1320)
LON, LAT = np.meshgrid(lon, lat[::-1])

lj, li = grid.lon_rho.shape
lj, li = lj-1, li-1

# in what area to actually plot the data
if args.figarea == 'wgom':
    figextent = [-98, -90, 18, 30]
    figsize = (4.9, 7)
    top, right, left, bottom =.96, .98, .15, .01
    caxpos = [0.19, 0.93, 0.25, 0.02]  # colorbar axis position
    datex, datey = 0.02, 0.02  # location of date on figure
    datax, datay = 0.55, 0.005  # location of data note on figure
elif args.figarea == 'txla':
    figextent = [-98, -87.5, 22.8, 30.5]
    figsize = (7, 7)
    top, right, left, bottom =.96, .98, .15, .01
    caxpos = [0.19, 0.79, 0.24, 0.02]  # colorbar axis position
    datex, datey = 0.01, 0.82  # location of date on figure
    datax, datay = 0.41, 0.97  # location of data note on figure
    # tuples of indices for bottom, left, top, right of domain to check for values before plotting
    ibottom = (0, np.arange(0, li))
    ileft = (np.arange(0, lj), -1)
    itop = (-1, np.arange(li, 0, -1))
    iright = (np.arange(lj, 0, -1), 0)
elif args.figarea == 'galv_plume':
    figextent = [-94.95, -94.4, 29.1, 29.6]
    figsize = (7, 7)
    top, right, left, bottom =.96, .98, .15, .01
    caxpos = [0.19, 0.79, 0.24, 0.02]  # colorbar axis position
    datex, datey = 0.73, 0.95  # location of date on figure
    datax, datay = 0.41, 0.97  # location of data note on figure
    scalex, scaley = 0.07, 0.02  # location of scale
    # tuples of indices for bottom, left, top, right of domain to check for values before plotting
    # (I think these *are* named properly)
    itop = (165, np.arange(275, 280) )
    iright = (np.arange(165, 155, -1), 280)
    ibottom = (160, np.arange(280, 275, -1))
    ileft = (np.arange(155, 165), 275)


if args.var == 'sst':
    cmap = cmo.thermal
    cmin = 10; cmax = 35; dc = 5
    ticks = np.arange(cmin, cmax+dc, dc)
elif args.var == 'oci':
    cmap = cmo.algae
    cmin = 0.1; cmax = 5; dc = 5
elif args.var == 'ci':
    cmap = cmo.algae
    # # larger range
    # cmin = 0.02; cmax = 0.4; dc = 5
    # ticks = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    # # more of colormap used
    # cmin = 0.035; cmax = 0.15; dc = 5
    # ticks = np.array([0.035, 0.05, 0.75, 0.1, 0.15])
    # # a little darker for light end
    # cmin = 0.01; cmax = 0.15; dc = 5
    # ticks = np.array([0.01, 0.02, 0.05, 0.75, 0.1, 0.15])
    #
    cmin = 0.025; cmax = 0.15; dc = 5
    ticks = np.array([0.025, 0.05, 0.75, 0.1, 0.15])
    # cmin = 0.005; cmax = 0.5; dc = 5
    # cmin = 0.002; cmax = 0.5; dc = 5
    # ticks = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    # ticks = np.array([0.01, 0.02, 0.05, 0.2])

url = 'http://optics.marine.usf.edu/subscription/modis/' + args.area.upper() + '/' + str(args.year) + '/daily/'
soup = BeautifulSoup(requests.get(url).text, "lxml")

# indices to be within the box of the domain and not covered by land and in numerical domain
# for full numerical domain:
lon = np.concatenate((grid.lon_rho[ibottom], grid.lon_rho[ileft], grid.lon_rho[itop], grid.lon_rho[iright]))
lat = np.concatenate((grid.lat_rho[ibottom], grid.lat_rho[ileft], grid.lat_rho[itop], grid.lat_rho[iright]))
verts = np.vstack((lon, lat)).T
path = Path(verts)
ptstotal = path.contains_points(np.vstack((LON.flat, LAT.flat)).T).sum()  # how many points possible within model domain
# true/false array of sat data points contained within area of interest
ipts = path.contains_points(np.vstack((LON.flat, LAT.flat)).T)

# numerical domain
nlon = np.concatenate((grid.lon_rho[::-1, 0], grid.lon_rho[0, :], grid.lon_rho[:, -1]))
nlat = np.concatenate((grid.lat_rho[::-1, 0], grid.lat_rho[0, :], grid.lat_rho[:, -1]))
# verts = np.vstack((lon, lat)).T

# x = np.concatenate((grid.x_rho[0, :], grid.x_rho[:, -1], grid.x_rho[-1, ::-1], grid.x_rho[::-1, 0]))
# y = np.concatenate((grid.y_rho[0, :], grid.y_rho[:, -1], grid.y_rho[-1, ::-1], grid.y_rho[::-1, 0]))
# verts = np.vstack((x, y)).T
# path = Path(verts)
# ptstotal = path.contains_points(np.vstack((X.flat, Y.flat)).T).sum()  # how many points possible within model domain

def scale_bar(ax, length, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    """

    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(merc)
    #Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * 500, sbcx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sbcy, sbcy], transform=merc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbcx, sbcy, str(length) + ' km', transform=merc,
            horizontalalignment='center', verticalalignment='bottom')

if not os.path.exists('figures/'):  # make sure directory exists
    os.makedirs('figures/')
if not os.path.exists('figures/' + args.var):  # make sure directory exists
    os.makedirs('figures/' + args.var)
if not os.path.exists('figures/' + args.var + '/' + args.figarea):  # make sure directory exists
    os.makedirs('figures/' + args.var + '/' + args.figarea)

for row in soup.findAll('a')[5:]:  # loop through each day
    # print row
    soup_dir = BeautifulSoup(requests.get(url + row.string).text, "lxml")  # open up page for a day
    # print soup_dir
    for File in soup_dir.findAll('a')[5:]:  # find all files for this day
        # print File
        # search for the image file we want, might be more than one for a day
        if args.area == 'gcoos':
            if args.var == 'ci':
                fname = '.1KM.' + args.area.upper() + '.PASS.L3D_RRC.' + args.var.upper() + '.png'
            else:
                fname = '.1KM.' + args.area.upper() + '.PASS.L3D.' + args.var.upper() + '.png'
        elif args.area == 'wgom':
            fname = '.1KM.' + args.area.upper() + '.PASS.L3D_RRC.' + args.var.upper() + '.png'
        elif args.area == 'galv':
            fname = '.QKM.' + args.area.upper() + '.PASS.L3D_RRC.' + args.var.upper() + '.png'


        if fname in File.string:
            image_loc = url + row.string + File.string  # save file address
            day = row.string.split('/')[0]
            time = File.string.split('.')[0][-4:]
            whichsat = File.string[0]  # which satellite
            date = datetime(args.year, 1, 1) + timedelta(days=int(day) - 1) \
                    + timedelta(hours=int(time[:2])) + timedelta(minutes=int(time[2:]))
            filename = 'figures/' + args.var + '/' + args.figarea + '/' + date.isoformat()[0:13] + date.isoformat()[14:16] + '-' + args.area + '.png'
            if os.path.exists(filename):
                continue
            # open and load in image
            response = requests.get(image_loc)
            img = Image.open(io.BytesIO(response.content))
            # img = Image.open(StringIO(response.content))
            foo = np.asarray(img)
            # if not date == datetime(2015, 12, 3, 17, 20):
            #     continue
            # different processing if RGB vs. embedded scalar values
            if args.var != 'rgb':
                # mask out bad areas
                foo_mask = np.ma.masked_where(foo > 236, foo)
                # Only plot image if subdomain we're interested in is there
                mask = ~foo_mask.mask
                # count active sat data points with model domain to have enough shown
                # also make sure not all zeros
                if not ((path.contains_points(np.vstack((LON[mask], LAT[mask])).T).sum() > (ptstotal/4.)) \
                     and ((foo == 0).sum() < 500000)):

                    continue
                else:

                    # Temperature data: map data that goes from 0 to 236 to be from 10 to 32 instead
                    if args.var == 'sst':
                        foo_mask = (foo_mask/236.)*22 + 10
                    # Chlorophyll: map data onto 0.01 to 5 logscale
                    elif args.var == 'oci':
                        # http://stackoverflow.com/questions/19472747/convert-linear-scale-to-logarithmic/19472811#19472811
                        bb = np.log(5/0.01)/(1-0.0)  # mapping from 0 to 1 (linear) to 0.01 to 5 (logscale)
                        aa = 5/np.exp(bb*1)
                        foo_mask = aa*np.exp(bb*(foo_mask/236.))  # now in logscale
                    # Color index for no sun glint: map data onto 0.002 to 0.5 logscale which does not connect directly with chlorophyll numbers
                    elif args.var == 'ci':
                        # http://stackoverflow.com/questions/19472747/convert-linear-scale-to-logarithmic/19472811#19472811
                        bb = np.log(0.5/0.005)/(1-0.0)  # mapping from 0 to 1 (linear) to 0.005 to 5 (logscale)
                        aa = 0.5/np.exp(bb*1)
                        foo_mask = aa*np.exp(bb*(foo_mask/236.))  # now in logscale
            else:
                # check for too much black or white in designated important region (defined by ibottom, etc)
                # check for black in one channel within important region
                if (foo[:,:,0].flat[ipts] == 0).sum() > (ptstotal*(1./4)):
                    continue
                # check for white in one channel
                if (foo[:,:,0].flat[ipts] >= 180).sum() > (ptstotal*(1./4)):
                    continue

            # plot
            fig = plt.figure(figsize=figsize)#, dpi=200)  # (9.5, 10))
            fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom)
            ax = plt.axes(projection=merc)
            gl = ax.gridlines(linewidth=0.2, color='gray', alpha=0.5, linestyle='-', draw_labels=True)
            # the following two make the labels look like lat/lon format
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # gl.xlocator = mticker.FixedLocator([-105, -95, -85, -75, -65])  # control where the ticks are
            # gl.xlabel_style = {'size': 15, 'color': 'gray'}  # control how the tick labels look
            # gl.ylabel_style = {'color': 'red', 'weight': 'bold'}
            gl.xlabels_bottom = False  # turn off labels where you don't want them
            gl.ylabels_right = False
            ax.add_feature(land_10m, facecolor='0.8')
            ax.coastlines(resolution='10m')  # coastline resolution options are '110m', '50m', '10m'
            ax.add_feature(states_provinces, edgecolor='0.2')
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='0.2')
            ax.set_extent(figextent, pc)

            if args.var == 'sst':
                mappable = ax.pcolormesh(LON, LAT, foo_mask, cmap=cmap, transform=pc)
            elif (args.var == 'oci') or (args.var == 'ci'):
                mappable = ax.imshow(foo_mask.filled(0), origin='upper', cmap=cmap, transform=pc, extent=dataextent, norm=LogNorm(vmin=cmin, vmax=cmax))
                # mappable = ax.pcolormesh(LON, LAT, foo_mask, cmap=cmap, norm=LogNorm(vmin=cmin, vmax=cmax), transform=pc)
            elif args.var == 'rgb':
                mappable = ax.imshow(foo, origin='upper', transform=pc, extent=dataextent)

            # isobaths
            if plotbathy:
                if args.figarea == 'txla':  # don't have data outside numerical domain
                    ax.contour(grid.lon_rho, grid.lat_rho, grid.h, hlevs, colors='0.6', transform=pc, linewidths=0.5)
                    ax.contour(grid.lon_rho, grid.lat_rho, grid.h, [100], colors='k', transform=pc, linewidths=0.5)  # 100 m isobath in black

            # plot numerical domain
            if plotdomain:
                ax.plot(nlon, nlat, ':k', transform=pc, lw=0.75, alpha=0.7)

            # plot shipping lanes
            if plotshipping:
                # shipping lanes
                ax.add_feature(shape_feature, edgecolor='k', linewidth=0.5, linestyle='--')

            # plot tide for context
            if plottide:

                # plot tide location
                ax.plot(tndbc[0], tndbc[1], '^', color='k', markersize=5, transform=pc)

                # tidal prediction (only goes back and forward in time 2 years)
                # base = 'https://tidesandcurrents.noaa.gov/noaacurrents/DownloadPredictions?fmt=csv&i=30min&d='
                # suffix = '&r=2&tz=GMT&u=2&id=g06010_1&t=24hr&i=30min&threshold=leEq&thresholdvalue='
                # url to download data file starting the day before this sat data, week of data
                # url2 = base + dtstart.strftime('%Y-%m-%d') + suffix
                ##
                dtstart = date - timedelta(days=1)
                dtend = dtstart + timedelta(days=2)
                # tidal data
                base = 'https://tidesandcurrents.noaa.gov/cdata/DataPlot?id=g06010&bin=0&bdate='
                suffix = '&unit=0&timeZone=UTC&view=csv'
                url2 = base + dtstart.strftime('%Y%m%d') + '&edate=' + dtend.strftime('%Y%m%d') + suffix
                try:
                    df = pd.read_csv(url2, parse_dates=True, index_col=0)
                except:  # sometimes there is no data
                    continue
                df = df[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                # angle needs to be in math convention for trig and between 0 and 360
                theta = 90 - df[' Dir (true)']
                theta[theta<0] += 360
                # tidal data needs to be converted into along-channel direction
                # along-channel flood direction (from website for data), converted from compass to math angle
                diralong = 259
                diralong = 90 - diralong + 360
                # direbb, dirflood = 77, 282  # deg true, from tidal prediction website
                # direbb, dirflood = 90 - direbb, 90 - dirflood + 360  # change from compass to math angles
                # import pdb; pdb.set_trace()
                # first convert to east/west, north/south
                # all speeds in cm/s
                east = df[' Speed (cm/sec)']*np.cos(np.deg2rad(theta))
                north = df[' Speed (cm/sec)']*np.sin(np.deg2rad(theta))
                # then convert to along-channel (mean ebb and mean flood)
                df['along'] = east*np.cos(diralong) - north*np.sin(diralong)

                # wind
                # NDBC data buoy 8771341
                windname = '8771341'
                base = 'https://tidesandcurrents.noaa.gov/cgi-bin/newdata.cgi?type=met&id=8771341&begin='
                suffix = '&units=metric&timezone=GMT&mode=csv&interval=h'
                url2 = base + dtstart.strftime('%Y%m%d') + '&end=' + dtend.strftime('%Y%m%d') + suffix
                dfw = pd.read_csv(url2, parse_dates=True, index_col=0)
                dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                # angle needs to be in math convention for trig and between 0 and 360
                # also have to switch wind from direction from to direction to with 180 switch
                theta = 90 - (dfw[' DIR'] - 180)
                theta[theta<0] += 360
                # add to original dataframe
                dfw['East [m/s]'] = dfw[' WINDSPEED']*np.cos(np.deg2rad(theta))
                dfw['North [m/s]'] = dfw[' WINDSPEED']*np.sin(np.deg2rad(theta))
                df = pd.merge(df, dfw[['East [m/s]', 'North [m/s]']], how='outer', left_index=True, right_index=True)
                # if more than half of the wind entries are nan, use TABS buoys instead
                # if date == datetime(2017, 3, 2, 18, 55):
                #     import pdb; pdb.set_trace()
                if len(dfw[' DIR']) == 0 or np.isnan(dfw[' DIR']).sum() > 0.5*len(dfw[' DIR']):
                    try:
                        windname = '42035'
                        base = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?tz=UTC&units=M&Buoyname=42035&table=ndbc&datepicker='
                        suffix = '&Datatype=download&model=False'
                        url2 = base + dtstart.strftime('%Y-%m-%d') + '+-+' + dtend.strftime('%Y-%m-%d') + suffix
                        dfw = pd.read_csv(url2, parse_dates=True, index_col=0, delimiter='\t')
                        dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                        # interpolate to hourly like 8771341
                        dfw = dfw.resample('60T').interpolate()
                        # add to original dataframe
                        df = pd.merge(df, dfw[['East [m/s]', 'North [m/s]']], how='outer', left_index=True, right_index=True)

                        # throw assertion if still too many nans
                        assert not np.isnan(dfw['East [m/s]']).sum() > 0.5*len(dfw['East [m/s]'])
                    except:
                        # backup: TABS buoy B
                        windname = 'B'
                        base = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?tz=UTC&units=M&Buoyname=B&table=met&datepicker='
                        suffix = '&Datatype=download&model=False'
                        url2 = base + dtstart.strftime('%Y-%m-%d') + '+-+' + dtend.strftime('%Y-%m-%d') + suffix
                        dfw = pd.read_csv(url2, parse_dates=True, index_col=0, delimiter='\t')
                        dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                        # interpolate to hourly like 8771341
                        dfw = dfw.resample('60T').interpolate()
                        # add to original dataframe
                        df = pd.merge(df, dfw[['East [m/s]', 'North [m/s]']], how='outer', left_index=True, right_index=True)
                df.idx = date2num(df.index.to_pydatetime())  # in units of days


                axwind = fig.add_axes([0.51, 0.83, 0.24, 0.08])
                ddt = 1
                # import pdb; pdb.set_trace()
                axwind.quiver(df.idx[::ddt], np.zeros(len(df[::ddt])), df[::ddt]['East [m/s]'], df[::ddt]['North [m/s]'], headaxislength=0,
                          headlength=0, width=0.2, units='y', scale_units='y', scale=1, color='k')
                axwind.text(0.75, 0.02, windname, fontsize=6, transform=axwind.transAxes)
                axwind.get_yaxis().set_ticks(np.arange(-10,15,5))
                axwind.get_yaxis().set_ticklabels(['', '-5', '', '5', ''])
                axwind.set_ylim(-10.1,10.1)
                axwind.axvline(x=date2num(date), ymin=0, ymax=1)
                [s.set_visible(False) for s in axwind.spines.values()]
                axwind.get_xaxis().set_ticks([])
                [t.set_visible(False) for t in axwind.get_xticklines()]

                axtide = fig.add_axes([0.51, 0.74, 0.24, 0.08], sharex=axwind)
                axtide.plot(df.idx, df['along']/100, color='k')
                axtide.get_yaxis().set_ticks(np.arange(-1,1.5,0.5))
                axtide.get_yaxis().set_ticklabels(['', '-0.5', '', '0.5', ''])
                axtide.set_ylim(-1.3, 1.3)
                plt.xticks(rotation=70)
                axtide.axhline(y=0.0, xmin=0, xmax=1, color='k', linestyle=':', linewidth=0.5)
                axtide.axvline(x=date2num(date), ymin=0, ymax=1)
                [s.set_visible(False) for s in axtide.spines.values()]
                hours = mpl.dates.HourLocator(byhour=np.arange(0,24,12))
                axtide.xaxis.set_minor_locator(hours)
                days = mpl.dates.DayLocator()
                axtide.xaxis.set_major_locator(days)
                axtide.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d'))
                # import pdb; pdb.set_trace()



            # data source
            if plotsource:
                ax.text(datax, datay, 'data from optics.marine.usf.edu/', fontsize=8, transform=ax.transAxes, color='0.3')

            # Date and time
            ax.text(datex, datey, date.strftime('%Y %b %d %H:%M'), fontsize=11, color='0.2', transform=ax.transAxes)#,
                    # bbox=dict(facecolor='0.8', edgecolor='0.8', boxstyle='round'))

            # scale
            # https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
            if plotscale:
                scale_bar(ax, scale, location=(scalex, scaley))

            # Colorbar in upper left corner
            if args.var != 'rgb':
                cax = fig.add_axes(caxpos)  # colorbar axes
                cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
                if args.var == 'sst':
                    cb.set_label(r'Surface temperature [$^\circ\!$C]', fontsize=14, color='0.2')
                    cb.set_ticks(ticks)
                elif args.var == 'oci':
                    cb.set_label(r'Chlorophyll-a [mg$\,$m$^{-3}$]', fontsize=14, color='0.2')
                elif args.var == 'ci':
                    cb.set_label('Color index', fontsize=9, color='0.2')
                    if len(ticks) < 7:
                        dd=1
                    else:
                        dd=2
                    cb.set_ticks(ticks[::dd])
                    cb.set_ticklabels(ticks[::dd])
                cb.ax.tick_params(labelsize=9, length=2, color='0.2', labelcolor='0.2')
                # # box behind to hide lines
                # ax.add_patch(patches.Rectangle((0.005, 0.925), 0.42, 0.0625, transform=ax.transAxes, color='0.8', zorder=3))
                # ax.add_patch(patches.Rectangle((0.1, 0.895), 0.24, 0.029, transform=ax.transAxes, color='0.8', zorder=3))
                # change colorbar tick color http://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
                cbtick = plt.getp(cb.ax.axes, 'yticklabels')
                plt.setp(cbtick, color='0.2')

            fig.savefig(filename, bbox_inches='tight')
            # import pdb; pdb.set_trace()
            plt.close(fig)
