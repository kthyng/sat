'''
Plot satellite data in the northwestern Gulf of Mexico.

Usage:
plot_sat.py [-h] year var area
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from StringIO import StringIO
from cmocean import cm
import tracpy
import tracpy.plotting
from bs4 import BeautifulSoup
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.path import Path
import pdb
from datetime import datetime, timedelta
import argparse
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
import os


# Input arguments: year and what to plot
parser = argparse.ArgumentParser()
parser.add_argument('year', type=int, help='What year to plot')
parser.add_argument('var', type=str, help='What field to plot: "sst" (sea surface temp) or "oci" (chlorophyll-a with good correction algorithm) or "ci" (chlorophyll-a with no sun glint)')
parser.add_argument('area', type=str, help='Area getting data from to plot: "gcoos" (full Gulf of Mexico) or "wgom" (western Gulf of Mexico)')
args = parser.parse_args()

if (args.var == 'ci') and (args.area == 'gcoos'):
    print 'Color index is not available for the full Gulf of Mexico. Choose "wgom" instead.'

mpl.rcParams.update({'font.size': 14})
mpl.rcParams['font.sans-serif'] = 'Arev Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.cal'] = 'cursive'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.tt'] = 'monospace'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.bf'] = 'sans:bold'
mpl.rcParams['mathtext.sf'] = 'sans'
mpl.rcParams['mathtext.fallback_to_cm'] = 'True'

# grid_filename = '../../grid.nc'
grid_filename = '/atch/raid1/zhangxq/Projects/txla_nesting6/txla_grd_v4_new.nc'
grid = tracpy.inout.readgrid(grid_filename, usebasemap=True, llcrnrlat=22.85, llcrnrlon=-97.9, urcrnrlat=30.5)
# proj = tracpy.tools.make_proj(setup='nwgom', usebasemap=True)
# grid = tracpy.inout.readgrid(grid_filename, proj, llcrnrlat=22.85, llcrnrlon=-97.9, urcrnrlat=30.5)

# grid = octant.grid.CGrid_geo(dsgrid['lon_vert'].data, dsgrid['lat_vert'].data, proj)

# Satellite data is in equidistant cylindrical projection which is just lon/lat
if args.area == 'gcoos':
    lon = np.linspace(-98, -79, 2090)
    lat = np.linspace(18, 31, 1430)
elif args.area == 'wgom':
    lon = np.linspace(-98, -90, 880)
    lat = np.linspace(18, 30, 1320)
LON, LAT = np.meshgrid(lon, lat[::-1])
X, Y = grid['basemap'](LON, LAT)

if args.var == 'sst':
    cmap = cm.temp
    cmin = 10; cmax = 35; dc = 5
    ticks = np.arange(cmin, cmax+dc, dc)
elif args.var == 'oci':
    cmap = cm.chl
    cmin = 0.1; cmax = 5; dc = 5
elif args.var == 'ci':
    cmap = cm.chl
    cmin = 0.01; cmax = 0.1; dc = 5
    # cmin = 0.002; cmax = 0.5; dc = 5

url = 'http://optics.marine.usf.edu/subscription/modis/' + args.area.upper() + '/' + str(args.year) + '/daily/'
soup = BeautifulSoup(requests.get(url).text, "lxml")

# indices to be within the box of the domain and not covered by land and in numerical domain
x = np.concatenate((grid['xr'][0, :], grid['xr'][:, -1], grid['xr'][-1, ::-1], grid['xr'][::-1, 0]))
y = np.concatenate((grid['yr'][0, :], grid['yr'][:, -1], grid['yr'][-1, ::-1], grid['yr'][::-1, 0]))
verts = np.vstack((x, y)).T
path = Path(verts)
ptstotal = path.contains_points(np.vstack((X.flat, Y.flat)).T).sum()  # how many points possible within model domain

if not os.path.exists('figures/'):  # make sure directory exists
    os.makedirs('figures/')
if not os.path.exists('figures/' + args.var):  # make sure directory exists
    os.makedirs('figures/' + args.var)

for row in soup.findAll('a')[5:]:  # loop through each day
    # print row
    soup_dir = BeautifulSoup(requests.get(url + row.string).text, "lxml")  # open up page for a day
    # print soup_dir
    for File in soup_dir.findAll('a')[5:]:  # find all files for this day
        # print File
        # search for the image file we want, might be more than one for a day
        if args.area == 'gcoos':
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
            filename = 'figures/' + args.var + '/' + date.isoformat()[0:13] + date.isoformat()[14:16] + '-' + args.area + '.png'
            if os.path.exists(filename):
                continue

            # if not ('2015-08-26' in date.isoformat()):
            #     continue

            # print date
            # open and load in image
            response = requests.get(image_loc)
            img = Image.open(StringIO(response.content))
            foo = np.asarray(img)
            # mask out bad areas
            foo_mask = np.ma.masked_where(foo > 236, foo)
            # Only plot image if subdomain we're interested in is there
            mask = ~foo_mask.mask
            # count active sat data points with model domain to have enough shown
            # also make sure not all zeros
            if (path.contains_points(np.vstack((X[mask], Y[mask])).T).sum() > (ptstotal/4.)) \
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
                    bb = np.log(0.5/0.002)/(1-0.0)  # mapping from 0 to 1 (linear) to 0.01 to 5 (logscale)
                    aa = 0.5/np.exp(bb*1)
                    foo_mask = aa*np.exp(bb*(foo_mask/236.))  # now in logscale

                # plot
                fig = plt.figure(figsize=(10.1, 8.4), dpi=100)
                ax = fig.add_axes([0.06, 0.00, 0.93, 0.97])
                ax.set_frame_on(False)  # kind of like it without the box
                tracpy.plotting.background(grid=grid, ax=ax, mers=np.arange(-97, -87), merslabels=[0, 0, 1, 0], pars=np.arange(23, 32), col='0.2', halpha=0.5)
                if args.var == 'sst':
                    mappable = ax.pcolormesh(X, Y, foo_mask, cmap=cmap)
                elif (args.var == 'oci') or (args.var == 'ci'):
                    mappable = ax.pcolormesh(X, Y, foo_mask, cmap=cmap, norm=LogNorm(vmin=cmin, vmax=cmax))

                # data source
                ax.text(0.45, 0.98, 'satellite data from http://optics.marine.usf.edu/', fontsize=10, transform=ax.transAxes, color='0.3')

                # Date and time
                ax.text(0.02, 0.8, date.strftime('%Y %b %d %H:%M'), fontsize=18, color='0.2', transform=ax.transAxes,
                        bbox=dict(facecolor='0.8', edgecolor='0.8', boxstyle='round'))

                # Colorbar in upper left corner
                cax = fig.add_axes([0.09, 0.92, 0.35, 0.025])  # colorbar axes
                cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
                if args.var == 'sst':
                    cb.set_label(r'Surface temperature [$^\circ\!$C]', fontsize=14, color='0.2')
                    cb.set_ticks(ticks)
                elif args.var == 'oci':
                    cb.set_label(r'Chlorophyll-a [mg$\,$m$^{-3}$]', fontsize=14, color='0.2')
                elif args.var == 'ci':
                    cb.set_label('Color index', fontsize=14, color='0.2')
                cb.ax.tick_params(labelsize=14, length=2, color='0.2', labelcolor='0.2')
                # box behind to hide lines
                ax.add_patch(patches.Rectangle((0.005, 0.925), 0.42, 0.0625, transform=ax.transAxes, color='0.8', zorder=3))
                ax.add_patch(patches.Rectangle((0.1, 0.895), 0.24, 0.029, transform=ax.transAxes, color='0.8', zorder=3))
                # change colorbar tick color http://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
                cbtick = plt.getp(cb.ax.axes, 'yticklabels')
                plt.setp(cbtick, color='0.2')
                fig.savefig(filename, bbox_inches='tight')
                plt.close(fig)
