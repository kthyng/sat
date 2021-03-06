'''
Plot satellite data in the northwestern Gulf of Mexico.

Usage:
plot_sat.py [-h] year "var" "area"

Example usage:
run plot_sat 2014 "ci" "wgom" "wgom"
run plot_sat 2017 "ci" "gcoos" "txla"
run plot_sat 2017 "ci" "wgom" "txla" --plotsource 'yes' --plotbathy 'yes' --plotsource 'yes' --scale 50
run plot_sat 2017 "rgb" "galv" "galv_plume" --plotshipping 'yes' --plottide 'yes' --scale 5
run plot_sat 2017 "rgb" "galv" "galv_bay" --plotsource 'yes' --plottide 'yes' --scale 5  --dpi 100
run plot_sat 2017 "rgb" "galv" "galv_plume" --plotsource 'yes' --plottide 'yes' --scale 5  --dpi 100 --click 'yes'
run plot_sat 2018 "rgb" "galv" "galv_plume" --plotsource 'yes' --plottide 'yes' --scale 5  --dpi 100 --plotlocs 'yes'
run plot_sat 2017 "ci" "wgom" "TX" --plotsource 'yes' --plotbathy 'yes' --plotsource 'yes' --scale 50
run plot_sat 2016 "rgb" "galv" "galv_plume" --plotsource 'yes' --plottide 'yes' --scale 5  --dpi 300 --figtype 'tiff'
** not CI for GCOOS before 2016 -- GCOOS is appearing lighter for CI in 2016 and 2017 than in WGOM
'''

import numpy as np
from PIL import Image
import requests
import io
import cmocean.cm as cmo
try:
    from bs4 import BeautifulSoup
except:
    from BeautifulSoup import BeautifulSoup
import pdb
from datetime import datetime, timedelta
import argparse
import os
import cartopy
ccrs = cartopy.crs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import xarray as xr
import cartopy.io.shapereader as shpreader
import pandas as pd
import logging

# Input arguments: year and what to plot
parser = argparse.ArgumentParser()
parser.add_argument('year', type=int, help='What year to plot')
parser.add_argument('var', type=str, help='What field to plot: "sst" (sea surface temp) or "oci" (chlorophyll-a with good correction algorithm) or "ci" (chlorophyll-a with no sun glint) or "rgb" (color) or "CHL" (chlorophyll-a)')
parser.add_argument('area', type=str, help='Area getting data from to plot: "gcoos" (full Gulf of Mexico) or "wgom" (western Gulf of Mexico) or "galv"')
parser.add_argument('figarea', type=str, help='What area in Gulf to plot data in: "wgom" (western Gulf of Mexico) or "txla" (TXLA domain) or "galv_plume" or "galv_bay" or "TX"')
parser.add_argument('--plotbathy', default=None, type=str, help='Plot bathymetry from NWGOM numerical model or not (any string input will cause to plot)')
parser.add_argument('--plotdomain', default=None, type=str, help='Plot numerical domain from NWGOM numerical model or not (any string input will cause to plot)')
parser.add_argument('--plotshipping', default=None, type=str, help='Plot shipping lanes or not (any string input will cause to plot)')
parser.add_argument('--plottide', default=None, type=str, help='Plot tides or not (any string input will cause to plot)')
parser.add_argument('--plotsource', default=None, type=str, help='Plot sat data link or not (any string input will cause to plot)')
parser.add_argument('--plotlocs', default=None, type=str, help='Plot input locations, currently CTD sampling.')
parser.add_argument('--plotblobs', default=None, type=str, help='Plot blobs for detection.')
parser.add_argument('--scale', default=None, type=int, help='To plot reference scale, input the reference km value as as int.')
parser.add_argument('--dpi', default=100, type=int, help='dpi as int.')
parser.add_argument('--click', default=None, type=str, help='Pause each image to possibly click points.')
parser.add_argument('--figtype', default=None, type=str, help='Option in case you do not want a png for your image save file. Default "png", options: "tiff", "eps", "jpg"')
args = parser.parse_args()


logging.basicConfig(filename='%s.log' % args.year, level=logging.INFO)
logging.info('\n\n\n')

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
if args.plotlocs is not None:
    plotlocs = True
else:
    plotlocs = False
if args.plotblobs is not None:
    plotblobs = True
else:
    plotblobs = False
if args.click is not None:
    click = True
else:
    click = False
if args.figtype is not None:
    figtype = args.figtype
else:
    figtype = 'png'
dpi = args.dpi


if not click:  # if we want to click, can't use this backend since need GUI
    import matplotlib as mpl
    mpl.use("Agg") # set matplotlib to use the backend that does not require a windowing system
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
from matplotlib.dates import date2num
mpl.rcParams.update({'font.size': 11})

loc = '/Volumes/GoogleDrive/My Drive/projects/grid.nc'
# loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/forecast_his_archive_agg.nc'
# loc = '../grid.nc'
# loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_hindcast_agg'
# loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/txla_nesting6_grid/txla_grd_v4_new.nc'

# KMT: replace grid work
# grid = xr.open_dataset(loc)
# hlevs = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]  # isobath contour depths

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
    lon = np.linspace(-96.5, -93.5, 1320)
    lat = np.linspace(27.8, 29.8, 880)
LON, LAT = np.meshgrid(lon, lat[::-1])

# lj, li = grid.lon_rho.shape
# lj, li = lj-1, li-1
# KMT: FIND li, lj

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
    scalex, scaley = 0.07, 0.02  # location of scale
    # tuples of indices for bottom, left, top, right of domain to check for values before plotting
    ibottom = (0, np.arange(0, li))
    ileft = (np.arange(0, lj), -1)
    itop = (-1, np.arange(li, 0, -1))
    iright = (np.arange(lj, 0, -1), 0)
elif args.figarea == 'TX':
    figextent = [-97.8, -94, 26, 29.9]
    figsize = (7, 7)
    top, right, left, bottom =.96, .98, .15, .01
    caxpos = [0.25, 0.85, 0.35, 0.02]  # colorbar axis position
    datex, datey = 0.1, 0.75  # location of date on figure
    datax, datay = 0.41, 0.97  # location of data note on figure
    scalex, scaley = 0.07, 0.02  # location of scale
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
    datex, datey = 0.45, 0.9  # location of date on figure
    datax, datay = 0.65, 0.01  # location of data note on figure
    scalex, scaley = 0.07, 0.02  # location of scale
    # tuples of indices for bottom, left, top, right of domain to check for values before plotting
    # (I think these *are* named properly)
    itop = (165, np.arange(275, 280) )
    iright = (np.arange(165, 155, -1), 280)
    ibottom = (155, np.arange(280, 275, -1))
    ileft = (np.arange(155, 165), 275)
    windextent = [0.65, 0.81, 0.32, 0.1]  # for wind box overlay
    tideextent = [0.65, 0.7, 0.32, 0.1]  # for tide box overlay
    overlayfont = 11
elif args.figarea == 'galv_bay':
    figextent = [-95.3, -94.455, 28.92, 29.8]
    figsize = (7, 7)
    top, right, left, bottom =.96, .98, .15, .01
    caxpos = [0.19, 0.79, 0.24, 0.02]  # colorbar axis position
    datex, datey = 0.025, 0.96  # location of date on figure
    datax, datay = 0.675, 0.983  # location of data note on figure
    scalex, scaley = 0.06, 0.12  # location of scale
    # tuples of indices for bottom, left, top, right of domain to check for values before plotting
    # (I think these *are* named properly)
    itop = (165, np.arange(275, 280) )
    iright = (np.arange(165, 155, -1), 280)
    ibottom = (155, np.arange(280, 275, -1))
    ileft = (np.arange(155, 165), 275)
    windextent = [0.22, 0.54, 0.26, 0.093]  # for wind box overlay
    tideextent = [0.22, 0.44, 0.26, 0.093]  # for tide box overlay
    overlayfont = 10

# instead of numerical grid, use fig area to decide about where need image to be present


if args.var == 'sst':
    cmap = cmo.thermal
    # # normally:
    # cmin = 10; cmax = 35; dc = 5
    # ticks = np.arange(cmin, cmax+dc, dc)
    # just post harvey:
    cmin = 25; cmax = 35; dc = 2
    ticks = np.arange(cmin, cmax+dc, dc)
    # # for small summer range (later post harvey)
    # cmin = 28; cmax = 30; dc = .5
    # ticks = np.arange(cmin, cmax+dc, dc)
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
    # cmin = 0.01; cmax = 0.1; dc = 5
    # ticks = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
    # cmin = 0.01; cmax = 0.05; dc = 5
    # ticks = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    # # used the following for small signal Harvey Rapid Response, 9/2017
    # cmin = 0.015; cmax = 0.05; dc = 5
    # ticks = np.array([0.015, 0.03, 0.04, 0.05])
    #
    # Use this for general txla CI
    cmin = 0.025; cmax = 0.15; dc = 5
    ticks = np.array([0.025, 0.05, 0.075, 0.1, 0.15])
    # cmin = 0.005; cmax = 0.5; dc = 5
    # cmin = 0.002; cmax = 0.5; dc = 5
    # ticks = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    # ticks = np.array([0.01, 0.02, 0.05, 0.2])

url = 'http://optics.marine.usf.edu/subscription/modis/' + args.area.upper() + '/' + str(args.year) + '/daily/'
soup = BeautifulSoup(requests.get(url).text, "lxml")

# indices to be within the box of the domain and not covered by land and in numerical domain
# for full numerical domain:
# KMT: replace this
# lon = np.concatenate((grid.lon_rho[ibottom], grid.lon_rho[ileft], grid.lon_rho[itop], grid.lon_rho[iright]))
# lat = np.concatenate((grid.lat_rho[ibottom], grid.lat_rho[ileft], grid.lat_rho[itop], grid.lat_rho[iright]))
# verts = np.vstack((lon, lat)).T
# path = Path(verts)
# # true/false array of sat data points contained within area of interest
# ipts = path.contains_points(np.vstack((LON.flat, LAT.flat)).T)
# ptstotal = ipts.sum()  # how many points possible within area of interest

# # numerical domain
# nlon = np.concatenate((grid.lon_rho[::-1, 0], grid.lon_rho[0, :], grid.lon_rho[:, -1]))
# nlat = np.concatenate((grid.lat_rho[::-1, 0], grid.lat_rho[0, :], grid.lat_rho[:, -1]))
# # verts = np.vstack((lon, lat)).T

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

# if not os.path.exists('figures/'):  # make sure directory exists
os.makedirs('figures/', exist_ok=True)
# if not os.path.exists('figures/' + args.var):  # make sure directory exists
os.makedirs('figures/%s' % args.var, exist_ok=True)
# if not os.path.exists('figures/' + args.var + '/' + args.figarea):  # make sure directory exists
os.makedirs('figures/%s/%s' % (args.var, args.figarea), exist_ok=True)
os.makedirs('figures/%s/%s/bad' % (args.var, args.figarea), exist_ok=True)  # for bad data
# foos = []
filebase = 'figures/' + args.var + '/' + args.figarea + '/'
if click:
    filebase += 'click/'
    clickbase = '/'.join(['calcs', 'pts'] + filebase.split('/')[1:])
#     if not os.path.exists(filebase):  # make sure directory exists
    os.makedirs(filebase, exist_ok=True)
    for i in range(len(clickbase.split('/')[:-1])):
        loc = '/'.join(clickbase.split('/')[:i+1])
#         if not os.path.exists(loc):  # make sure directory exists
        os.makedirs(loc, exist_ok=True)

for row in soup.findAll('a')[5:]:  # loop through each day
    # print row
    soup_dir = BeautifulSoup(requests.get(url + row.string).text, "lxml")  # open up page for a day
    # print soup_dir
    for File in soup_dir.findAll('a')[5:]:  # find all files for this day
        # print File
        # search for the image file we want, might be more than one for a day
#         if args.area == 'gcoos':
# #             if args.var == 'ci':
# #                 fname = '.1KM.' + args.area.upper() + '.PASS.L3D_RRC.' + args.var.upper() + '.png'
# #             else:
#             fname = '.1KM.' + args.area.upper() + '.PASS.L3D.' + args.var.upper() + '.png'
#         elif args.area == 'wgom':
# #             if args.var == 'sst':
#             fname = '.1KM.' + args.area.upper() + '.PASS.L3D.' + args.var.upper() + '.png'
# #             else:
# #                 fname = '.1KM.' + args.area.upper() + '.PASS.L3D_RRC.' + args.var.upper() + '.png'
#         elif args.area == 'galv':
#             fname = '.QKM.' + args.area.upper() + '.PASS.L3D.' + args.var.upper() + '.png'
        fname = '.QKM.' + args.area.upper() + '.PASS.L3D.' + args.var.upper() + '.png'

#         if '.RGB.' in File.string:
#             import pdb; pdb.set_trace()

        if fname in File.string:
            image_loc = url + row.string + File.string  # save file address
            day = row.string.split('/')[0]
            time = File.string.split('.')[0][-4:]
            whichsat = File.string[0]  # which satellite
            date = datetime(args.year, 1, 1) + timedelta(days=int(day) - 1) \
                    + timedelta(hours=int(time[:2])) + timedelta(minutes=int(time[2:]))
            filename = filebase + date.isoformat()[0:13] + date.isoformat()[14:16] + '-' + args.area
            if plotlocs:
                filename += '-locs'
            if dpi != 100:
                filename += '-dpi%i' % dpi

#             print(date)
            logging.info(date)
            # if not date > datetime(2016,1,22):
            #     continue
            # import pdb; pdb.set_trace()
            if os.path.exists('%s.%s' % (filename, figtype)):
                logging.info('%s: file already exists' % filename)
                continue
            # file might already exist in "bad" data directory
            filenamebad = '/'.join(filename.split('/')[:-1] + ['bad',filename.split('/')[-1]])
            if os.path.exists('%s.%s' % (filenamebad, figtype)):
                logging.info('%s: file already exists' % filenamebad)
                continue
            # open and load in image
            response = requests.get(image_loc)
            img = Image.open(io.BytesIO(response.content))
            # img = Image.open(StringIO(response.content))
            foo = np.asarray(img)
            # if not date == datetime(2017, 1, 7, 16, 20):
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
                # if date == datetime(2017, 1, 31, 18, 45):
                # # if date == datetime(2017, 9, 27, 19, 40):
                #     import pdb; pdb.set_trace()
                    # continue
                # check for too much black or white in designated important region (defined by ibottom, etc)
                # check for black in one channel within important region
                # KMT: need replacement here
#                 if (foo[:,:,0].flat[ipts] == 0).sum() > (ptstotal*(1./4)):
#                     print(filename + ': too much black in image')
#                     continue
#                 # check for white in one channel
#                 if (foo[:,:,0].flat[ipts] >= 180).sum() > (ptstotal*(1./4)):
#                     print(filename + ': too much white in image')
#                     continue
                ds = xr.DataArray(data=foo[::-1], coords={'lon': lon, 'lat': lat}, dims=['lat','lon','colors']).astype(int)
                foo = ds.sel(lon=slice(*figextent[:2]), lat=slice(*figextent[2:]))
                # check for black in one channel within important region
                # KMT: need replacement here
                # check for bad images: too much black, white, or grey in data in figview
#                 import pdb; pdb.set_trace()
                red, green, blue = foo[:,:,0].copy(deep=True), foo[:,:,1].copy(deep=True), foo[:,:,2].copy(deep=True)
                if ((red>150).sum() > red.size*0.05) \
                    and ((green>150).sum() > red.size*0.05) \
                    and ((blue>150).sum() > red.size*0.05):
#                     print('white')
                    # kludgey way of putting file in a subdir
                    logging.info('%s: Too much white in data' % filenamebad)
                    filename = filenamebad
                if (red==112).sum() > red.size/2:
#                     print('gray')
                    # kludgey way of putting file in a subdir
                    logging.info('%s: Too much gray in data' % filenamebad)
                    filename = filenamebad
#                     continue
# #
#                 try:
#                     toomany = foo[:,:,0].size/2
#                     red = foo[:,:,0]
#                     assert (foo[:,:,0] >= 220).sum() < toomany
#                     assert (foo[:,:,0] == 180).sum() < toomany
#                     assert (foo[:,:,0] <= 30).sum() < toomany  # white
#                     assert (foo[:,:,0] == 112).sum() < toomany  # gray
#                 except:
#                     continue
#                 if (foo[:,:,0] >= 220).sum() > foo[:,:,0].size/4:
#                     print(filename + ': too much black in image')
#                     continue
#                 # check for white in one channel
# #                 if (foo[:,:,0] == 180).sum() > foo[:,:,0].size/4:
# #                     print(filename + ': too much white in image')
# #                     continue
#                 if (foo[:,:,0] <= 30).sum() > foo[:,:,0].size/4:
#                     print(filename + ': too much white in image')
#                     continue
#                 if (foo[:,:,0] == 112).sum() > foo[:,:,0].size/4:
#                     print(filename + ': too much gray in image')
#                     continue
            # foos.append(foo)
            # plot
            fig = plt.figure(figsize=figsize)#, dpi=200)  # (9.5, 10))
            fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom)
            ax = plt.axes(projection=merc)
            gl = ax.gridlines(linewidth=0.2, color='gray', alpha=0.5, linestyle='-', draw_labels=True)
            # the following two make the labels look like lat/lon format
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.bottom_labels = False  # turn off labels where you don't want them
            gl.right_labels = False
            ax.add_feature(land_10m, facecolor='0.8')
            ax.coastlines(resolution='10m')  # coastline resolution options are '110m', '50m', '10m'
            ax.add_feature(states_provinces, edgecolor='0.2')
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='0.2')
            ax.set_extent(figextent, pc)

            if args.var == 'sst':
                mappable = ax.pcolormesh(LON, LAT, foo_mask, cmap=cmap, transform=pc, vmin=cmin, vmax=cmax)
            elif (args.var == 'oci') or (args.var == 'ci'):
                mappable = ax.imshow(foo_mask.filled(0), origin='upper', cmap=cmap, transform=pc, extent=dataextent, norm=LogNorm(vmin=cmin, vmax=cmax))
                # mappable = ax.pcolormesh(LON, LAT, foo_mask, cmap=cmap, norm=LogNorm(vmin=cmin, vmax=cmax), transform=pc)
            elif args.var == 'rgb':
                mappable = foo.plot.imshow(ax=ax, transform=pc, extent=dataextent)
#                 mappable = ax.imshow(foo, origin='lower', transform=pc, extent=dataextent)
#                 mappable = ax.imshow(foo, origin='upper', transform=pc, extent=dataextent)

            # isobaths
            if plotbathy:
                if args.figarea in ['txla', 'TX']:  # don't have data outside numerical domain
                    ax.contour(grid.lon_rho, grid.lat_rho, grid.h, hlevs, colors='0.6', transform=pc, linewidths=0.5)
                    ax.contour(grid.lon_rho, grid.lat_rho, grid.h, [100], colors='k', transform=pc, linewidths=0.5)  # 100 m isobath in black

            # plot numerical domain
            if plotdomain:
                ax.plot(nlon, nlat, ':k', transform=pc, lw=0.75, alpha=0.7)

            # overlay kml points for CTD locs
            if plotlocs:
                pts = np.loadtxt('kmlpts.txt')
                ax.plot(pts[:,0], pts[:,1], 'yo', transform=pc, markersize=4)

            # plot shipping lanes
            if plotshipping:
                # shipping lanes
                ax.add_feature(shape_feature, edgecolor='k', linewidth=0.5, linestyle='--')

            # plot tide for context
            if plottide:

                # plot tide location
                ax.plot(tndbc[0], tndbc[1], '^', color='k', markersize=5, transform=pc)

                ##
                dtstart = date - timedelta(days=1)
                dtend = dtstart + timedelta(days=2)
                # tidal data
                base = 'https://tidesandcurrents.noaa.gov/cdata/DataPlot?id=g06010&bin=0&bdate='
                suffix = '&unit=0&timeZone=UTC&view=csv'
                url2 = base + dtstart.strftime('%Y%m%d') + '&edate=' + dtend.strftime('%Y%m%d') + suffix
                try:
                    tidename = 'g06010'
                    df = pd.read_csv(url2, parse_dates=True, index_col=0)
                    df = df[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                    # angle needs to be in math convention for trig and between 0 and 360
                    theta = 90 - df[' Dir (true)']
                    theta[theta<0] += 360
                    # tidal data needs to be converted into along-channel direction
                    # along-channel flood direction (from website for data), converted from compass to math angle
                    diralong = 259
                    diralong = 90 - diralong + 360
                    # first convert to east/west, north/south
                    # all speeds in cm/s
                    east = df[' Speed (cm/sec)']*np.cos(np.deg2rad(theta))
                    north = df[' Speed (cm/sec)']*np.sin(np.deg2rad(theta))
                    # then convert to along-channel (mean ebb and mean flood)
                    # this is overwriting speed (magnitude) with speed (alongchannel)
                    df[' Speed (cm/sec)'] = east*np.cos(diralong) - north*np.sin(diralong)
                    # import pdb; pdb.set_trace()
                    assert df[' Speed (cm/sec)'].size > 2*24*60./6  # at least two days work of data to use

                except:  # sometimes there is no data
                    try:
                        print(filename + ': no tidal data, trying model')
                        tidename = 'model'
                        # tidal prediction (only goes back and forward in time 2 years)
                        base = 'https://tidesandcurrents.noaa.gov/noaacurrents/DownloadPredictions?fmt=csv&i=30min&d='
                        suffix = '&r=2&tz=GMT&u=2&id=g06010_1&t=24hr&i=30min&threshold=leEq&thresholdvalue='
                        # url to download data file starting the day before this sat data, week of data
                        url2 = base + dtstart.strftime('%Y-%m-%d') + suffix
                        df = pd.read_csv(url2, parse_dates=True, index_col=0)
                        df = df[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                    except:  # sometimes there is no data
                        print(filename + ': no tidal model, using derivative of tidal height data')

                        # NDBC data buoy 8771341 for tidal heights
                        tidename = '8771341 d/dt'
                        base = 'https://tidesandcurrents.noaa.gov/api/datagetter?product=water_level&application=NOS.COOPS.TAC.WL&station=8771341&begin_date='
                        suffix = '&datum=MLLW&units=metric&time_zone=GMT&format=csv'
                        # base = 'https://tidesandcurrents.noaa.gov/cgi-bin/newdata.cgi?type=met&id=8771341&begin='
                        # suffix = '&units=metric&timezone=GMT&mode=csv&interval=h'
                        url2 = base + (dtstart - timedelta(days=1)).strftime('%Y%m%d') + '&end_date=' + dtend.strftime('%Y%m%d') + suffix
                        df = pd.read_csv(url2, parse_dates=True, index_col=0)
                        # import pdb; pdb.set_trace()
                        # ad hoc calculation of tidal speed from the heights
                        df[' Speed (cm/sec)'] = (df[' Water Level'].rolling(window=15, center=True).mean().diff().rolling(window=15, center=True).mean()*7e3).shift(periods=50)
                        df = df[dtstart.strftime('%Y%m%d'):dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day

                        # continue

                # wind
                # NDBC data buoy 8771341
                windname = '8771341'
                base = 'https://tidesandcurrents.noaa.gov/cgi-bin/newdata.cgi?type=met&id=8771341&begin='
                suffix = '&units=metric&timezone=GMT&mode=csv&interval=h'
                url2 = base + dtstart.strftime('%Y%m%d') + '&end=' + dtend.strftime('%Y%m%d') + suffix
                dfw = pd.read_csv(url2, parse_dates=True, index_col=0)
                dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                if not (len(dfw[' DIR']) == 0 or np.isnan(dfw[' DIR']).sum() > 0.5*len(dfw[' DIR'])):
                    # angle needs to be in math convention for trig and between 0 and 360
                    # also have to switch wind from direction from to direction to with 180 switch
                    theta = 90 - (dfw[' DIR'] - 180)
                    theta[theta<0] += 360
                    dfw['East [m/s]'] = dfw[' WINDSPEED']*np.cos(np.deg2rad(theta))
                    dfw['North [m/s]'] = dfw[' WINDSPEED']*np.sin(np.deg2rad(theta))
                    # if date == datetime(2017, 3, 2, 18, 55):
                    #     import pdb; pdb.set_trace()
                # if more than half of the wind entries are nan, use TABS buoys instead
                else:
                    try:
                        windname = '42035'
                        base = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?tz=UTC&units=M&Buoyname=42035&table=ndbc&datepicker='
                        suffix = '&Datatype=download&model=False'
                        url2 = base + dtstart.strftime('%Y-%m-%d') + '+-+' + dtend.strftime('%Y-%m-%d') + suffix
                        dfw = pd.read_csv(url2, parse_dates=True, index_col=0, delimiter='\t')
                        dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                        # interpolate to hourly like 8771341
                        dfw = dfw.resample('60T').interpolate()

                        # throw assertion if still too many nans
                        assert not np.isnan(dfw['East [m/s]']).sum() > 0.5*len(dfw['East [m/s]'])
                    except:
                        # backup: TABS buoy B
                        windname = 'B'
                        base = 'http://pong.tamu.edu/tabswebsite/subpages/tabsquery.php?tz=UTC&units=M&Buoyname=B&table=met&datepicker='
                        suffix = '&Datatype=download&model=False'
                        url2 = base + dtstart.strftime('%Y-%m-%d') + '+-+' + dtend.strftime('%Y-%m-%d') + suffix
                        try:
                            dfw = pd.read_csv(url2, parse_dates=True, index_col=0, delimiter='\t')
                            dfw = dfw[:dtend.strftime('%Y%m%d') + ' 12:00'] # go until noon the following day
                            # interpolate to hourly like 8771341
                            dfw = dfw.resample('60T').interpolate()
                        except:
                            continue

                # add to original dataframe
                df = pd.merge(df, dfw[['East [m/s]', 'North [m/s]']], how='outer', left_index=True, right_index=True)
                df.idx = date2num(df.index.to_pydatetime())  # in units of days


                axwind = fig.add_axes(windextent)
                ddt = 1
                # import pdb; pdb.set_trace()
                axwind.quiver(df.idx[::ddt], np.zeros(len(df[::ddt])), df[::ddt]['East [m/s]'], df[::ddt]['North [m/s]'], headaxislength=0,
                          headlength=0, width=0.2, units='y', scale_units='y', scale=1, color='k')
                axwind.text(0.01, 0.02, windname, fontsize=7, transform=axwind.transAxes)
                axwind.text(0.01, 0.85, 'm/s', fontsize=7, transform=axwind.transAxes)
                axwind.get_yaxis().set_ticks(np.arange(-10,15,5))
                axwind.get_yaxis().set_ticklabels(['', '-5', '', '5', ''], fontsize=overlayfont)
                axwind.set_ylim(-10.5,10.5)
                axwind.axvline(x=date2num(date), ymin=0, ymax=1)
                [s.set_visible(False) for s in axwind.spines.values()]
                axwind.get_xaxis().set_ticks([])
                axwind.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
                [t.set_visible(False) for t in axwind.get_xticklines()]
                axwind.grid(color='0.2', linestyle='-', linewidth=0.1, which='both')

                axtide = fig.add_axes(tideextent, sharex=axwind)
                axtide.plot(df.idx, df[' Speed (cm/sec)']/100, color='k')
                axtide.text(0.01, 0.04, tidename, fontsize=7, transform=axtide.transAxes)
                axtide.text(0.01, 0.85, 'm/s', fontsize=7, transform=axtide.transAxes)
                # label ebb and flood
                axtide.text(0.88, 0.85, 'flood', fontsize=7, transform=axtide.transAxes)
                axtide.text(0.88, 0.02, 'ebb', fontsize=7, transform=axtide.transAxes)
                axtide.get_yaxis().set_ticks(np.arange(-1,1.5,0.5))
                axtide.get_yaxis().set_ticklabels(['', '-0.5', '', '0.5', ''], fontsize=overlayfont)
                axtide.set_ylim(-1.4, 1.4)
                plt.xticks(rotation=70)
                axtide.axhline(y=0.0, xmin=0, xmax=1, color='k', linestyle=':', linewidth=0.5)
                axtide.axvline(x=date2num(date), ymin=0, ymax=1)
                [s.set_visible(False) for s in axtide.spines.values()]
                hours = mpl.dates.HourLocator(byhour=np.arange(0,24,3))
                axtide.xaxis.set_minor_locator(hours)
                days = mpl.dates.DayLocator()
                axtide.xaxis.set_major_locator(days)
                axtide.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d'))
                axtide.autoscale(enable=True, axis='x', tight=True)
                axtide.grid(color='0.2', linestyle='-', linewidth=0.1, which='both')
                # axtide.set_xlim(dtstart.strftime('%Y-%m-%d'), dtend.strftime('%Y%m%d') + ' 12:00')



            # data source
            if plotsource:
                ax.text(datax, datay, 'data from optics.marine.usf.edu', fontsize=8, transform=ax.transAxes, color='0.3')

            # Date and time
            if args.figarea != 'galv_plume':
                ax.text(datex, datey, date.strftime('%Y %b %d %H:%M'), fontsize=14, color='0.2', transform=ax.transAxes)#,
            else:
                ax.text(datex, datey, date.strftime('%Y\n%b %d\n%H:%M'), fontsize=14, color='0.2', transform=ax.transAxes)#,
                    # bbox=dict(facecolor='0.8', edgecolor='0.8', boxstyle='round'))

            # scale
            # https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
            if plotscale:
                scale_bar(ax, scale, location=(scalex, scaley))

            # if plotblobs:
                # from skimage.feature import blob_dog, blob_log, blob_doh
                # from skimage.color import rgb2gray
                # y1, y2 = 150, 310
                # x1, x2 = 690, 920
                #
                # image = rgb2gray(foo)[y1:y2, x1:x2]
                # blobs_log = blob_log(image, min_sigma=10, max_sigma=40, threshold=.05)
                # blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
                # for blob in blobs_log:
                #     y, x, r = blob
                #     if (60 < y) and (y < 100) and (90 < x) and (x < 150):
                #         # import pdb; pdb.set_trace()
                #         # c = plt.Circle((x+x1, y+y1), r, color='y', linewidth=2, fill=False, transform=pc)
                #         # ax.add_patch(c)
                #         ax.scatter()


            # Colorbar in upper left corner
            if args.var != 'rgb':
                cax = fig.add_axes(caxpos)  # colorbar axes
                cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
                # import pdb; pdb.set_trace()
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

            if click:
                # Delete cancels last input
                # accumulates points until enter key is pushed
                pts = plt.ginput(timeout=0, n=-1)
                if len(pts) > 0:
                    x, y = zip(*pts)
                    ax.plot(x, y, 'y', lw=2)
                    clickname = clickbase + filename.split('/')[-1]
                    np.savez(clickname + '.npz', pts=pts)
                else:
                    # for clicking, don't save if no plume clicked
                    plt.close(fig)
                    continue

            fig.savefig('%s.%s' % (filename, figtype), bbox_inches='tight', dpi=dpi)
            # import pdb; pdb.set_trace()
            plt.close(fig)
# np.savez('notebooks/foos-' + str(args.year) + '.npz', foos=foos)
