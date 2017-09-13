'''
Plot satellite data in the northwestern Gulf of Mexico.

Usage:
plot_sat.py [-h] year "var" "area"

Example usage:
run plot_sat 2014 "ci" "wgom" "wgom"
run plot_sat 2017 "ci" "gcoos" "txla"
'''

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
# import tracpy
# import tracpy.plotting
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


# Input arguments: year and what to plot
parser = argparse.ArgumentParser()
parser.add_argument('year', type=int, help='What year to plot')
parser.add_argument('var', type=str, help='What field to plot: "sst" (sea surface temp) or "oci" (chlorophyll-a with good correction algorithm) or "ci" (chlorophyll-a with no sun glint) or "rgb" (color) or "CHL" (chlorophyll-a)')
parser.add_argument('area', type=str, help='Area getting data from to plot: "gcoos" (full Gulf of Mexico) or "wgom" (western Gulf of Mexico) or "galveston"')
parser.add_argument('figarea', type=str, help='What area in Gulf to plot data in: "wgom" (western Gulf of Mexico) or "txla" (TXLA domain)')
args = parser.parse_args()

mpl.rcParams.update({'font.size': 11})

# grid_filename = '../../grid.nc'
# grid_filename = '/atch/raid1/zhangxq/Projects/txla_nesting6/txla_grd_v4_new.nc'
# grid = tracpy.inout.readgrid(grid_filename, usebasemap=True, llcrnrlat=22.85, llcrnrlon=-97.9, urcrnrlat=30.5)
# proj = tracpy.tools.make_proj(setup='nwgom', usebasemap=True, llcrnrlat=22.85, llcrnrlon=-97.9, urcrnrlat=30.5)
# grid = tracpy.inout.readgrid(grid_filename, proj)
loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_hindcast_agg'
grid = xr.open_dataset(loc)
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
hlevs = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]  # isobath contour depths


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
elif args.area == 'galveston':
    dataextent = [-96.5, -93.5, 27.8, 29.8]
    lon = np.linspace(-96.5, -93.5, 880)
    lat = np.linspace(27.8, 29.8, 1320)
LON, LAT = np.meshgrid(lon, lat[::-1])

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


if args.var == 'sst':
    cmap = cmo.thermal
    cmin = 10; cmax = 35; dc = 5
    ticks = np.arange(cmin, cmax+dc, dc)
elif args.var == 'oci':
    cmap = cmo.algae
    cmin = 0.1; cmax = 5; dc = 5
elif args.var == 'ci':
    cmap = cmo.algae
    # cmin = 0.01; cmax = 0.2; dc = 5
    cmin = 0.02; cmax = 0.4; dc = 5
    # cmin = 0.005; cmax = 0.5; dc = 5
    # cmin = 0.01; cmax = 0.1; dc = 5
    # cmin = 0.002; cmax = 0.5; dc = 5
    # ticks = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ticks = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    # ticks = np.array([0.01, 0.02, 0.05, 0.2])

url = 'http://optics.marine.usf.edu/subscription/modis/' + args.area.upper() + '/' + str(args.year) + '/daily/'
soup = BeautifulSoup(requests.get(url).text, "lxml")

# indices to be within the box of the domain and not covered by land and in numerical domain
lon = np.concatenate((grid.lon_rho[0, :], grid.lon_rho[:, -1], grid.lon_rho[-1, ::-1], grid.lon_rho[::-1, 0]))
lat = np.concatenate((grid.lat_rho[0, :], grid.lat_rho[:, -1], grid.lat_rho[-1, ::-1], grid.lat_rho[::-1, 0]))
verts = np.vstack((lon, lat)).T
path = Path(verts)
ptstotal = path.contains_points(np.vstack((LON.flat, LAT.flat)).T).sum()  # how many points possible within model domain
# numerical domain
nlon = np.concatenate((grid.lon_rho[::-1, 0], grid.lon_rho[0, :], grid.lon_rho[:, -1]))
nlat = np.concatenate((grid.lat_rho[::-1, 0], grid.lat_rho[0, :], grid.lat_rho[:, -1]))
# verts = np.vstack((lon, lat)).T

# x = np.concatenate((grid.x_rho[0, :], grid.x_rho[:, -1], grid.x_rho[-1, ::-1], grid.x_rho[::-1, 0]))
# y = np.concatenate((grid.y_rho[0, :], grid.y_rho[:, -1], grid.y_rho[-1, ::-1], grid.y_rho[::-1, 0]))
# verts = np.vstack((x, y)).T
# path = Path(verts)
# ptstotal = path.contains_points(np.vstack((X.flat, Y.flat)).T).sum()  # how many points possible within model domain

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
            # mask out bad areas
            foo_mask = np.ma.masked_where(foo > 236, foo)
            # Only plot image if subdomain we're interested in is there
            mask = ~foo_mask.mask
            # count active sat data points with model domain to have enough shown
            # also make sure not all zeros
            if (path.contains_points(np.vstack((LON[mask], LAT[mask])).T).sum() > (ptstotal/4.)) \
                 and ((foo == 0).sum() < 500000):

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

                # plot
                fig = plt.figure(figsize=figsize, dpi=200)  # (9.5, 10))
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

                # isobaths
                if args.figarea == 'txla':  # don't have data outside numerical domain
                    ax.contour(grid.lon_rho, grid.lat_rho, grid.h, hlevs, colors='0.6', transform=pc, linewidths=0.5)

                # plot numerical domain
                ax.plot(nlon, nlat, ':k', transform=pc, lw=0.75, alpha=0.7)

                # data source
                ax.text(datax, datay, 'data from optics.marine.usf.edu/', fontsize=8, transform=ax.transAxes, color='0.3')

                # Date and time
                ax.text(datex, datey, date.strftime('%Y %b %d %H:%M'), fontsize=11, color='0.2', transform=ax.transAxes)#,
                        # bbox=dict(facecolor='0.8', edgecolor='0.8', boxstyle='round'))

                # Colorbar in upper left corner
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
