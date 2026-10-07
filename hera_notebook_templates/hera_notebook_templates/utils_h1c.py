# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pyuvdata import UVCal, UVData, utils
import os
import sys
import glob
import uvtools as uvt
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.coordinates import SkyCoord as sc
import pandas
import warnings 
import copy
import math
from uvtools import dspec
import hera_qm 
from hera_mc import cm_active
from matplotlib.lines import Line2D
from matplotlib import colors
import json
from hera_notebook_templates.data import DATA_PATH
from astropy.io import fits
import csv
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import healpy
import yaml
warnings.filterwarnings('ignore')



def read_a_priori_ant_flags(a_priori_flags_yaml, ant_indices_only=False, by_ant_pol=False, ant_pols=None):
    '''Parse an a priori flag YAML file for a priori antenna flags.
    Parameters
    ----------
    a_priori_flags_yaml : str
        Path to YAML file with a priori antenna flags
    ant_indices_only : bool
        If True, ignore polarizations and flag entire antennas when they appear, e.g. (1, 'Jee') --> 1.
    by_ant_pol : bool
        If True, expand all integer antenna indices into per-antpol entries using ant_pols
    ant_pols : list of str
        List of antenna polarizations strings e.g. 'Jee'. If not empty, strings in
        the YAML must be in here or an error is raised. Required if by_ant_pol is True.
    Returns
    -------
    a_priori_antenna_flags : list
         List of a priori antenna flags, either integers or ant-pol tuples e.g. (0, 'Jee')
    '''

    if ant_indices_only and by_ant_pol:
        raise ValueError("ant_indices_only and by_ant_pol can't both be True.")
    apaf = []
    apf = yaml.safe_load(open(a_priori_flags_yaml, 'r'))

    # Load antenna flags
    if 'ex_ants' in apf:
        for ant in apf['ex_ants']:
            # flag antenna number
            if type(ant) == int:
                apaf.append(ant)
            # flag single antpol
            elif (type(ant) == list) and (len(ant) == 2) and (type(ant[0]) == int) and (type(ant[1]) == str):
                # check that antpol string is valid if ant_pols is not empty
                if (ant_pols is not None) and (ant[1] not in ant_pols):
                    raise ValueError(f'{ant[1]} is not a valid ant_pol in {ant_pols}.')
                if ant_indices_only:
                    apaf.append(ant[0])
                else:
                    apaf += [tuple(ant)]
            else:
                raise TypeError(f'ex_ants entires must be integers or a list of one int and one str. {ant} is not.')

        # Expand all integer antenna flags into antpol pairs
        if by_ant_pol:
            if ant_pols is None:
                raise ValueError('If by_ant_pol is True, then ant_pols must be specified.')
            apapf = []
            for ant in apaf:
                if type(ant) == int:
                    apapf += [(ant, pol) for pol in ant_pols]
                else:  # then it's already and antpol tuple
                    apapf.append(ant)
            return sorted(set(apapf))

    return list(set(apaf))

def load_data(data_path,JD):
    HHfiles = sorted(glob.glob("{0}/zen.{1}.*.sum.uvh5".format(data_path,JD)))
    Nfiles = len(HHfiles)
    hhfile_bases = map(os.path.basename, HHfiles)
    sep = '.'
    x = sep.join(HHfiles[0].split('.')[-4:-2])
    y = sep.join(HHfiles[-1].split('.')[-4:-2])
    print(f'{len(HHfiles)} sum files found between JDs {x} and {y}')

    # choose one for single-file plots
    hhfile1 = HHfiles[len(HHfiles)//2]
    
    # Load data
    uvd_hh = UVData()

    unread = True
    while unread is True:
        try:
            uvd_hh.read(hhfile1, skip_bad_files=True)
        except:
            hhfile += 1
            continue
        unread = False
    uvd_xx1 = uvd_hh.select(polarizations = -5, inplace = False)
    uvd_xx1.ants = np.unique(np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array]))
    # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'

    uvd_yy1 = uvd_hh.select(polarizations = -6, inplace = False)
    uvd_yy1.ants = np.unique(np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array]))

   
    return HHfiles, uvd_xx1, uvd_yy1

def plot_sky_map(uvd,ra_pad=20,dec_pad=30,clip=True,fwhm=11,nx=300,ny=200,sources=[]):
    map_path = f'{DATA_PATH}/haslam408_dsds_Remazeilles2014.fits'
    hdulist = fits.open(map_path)

    # Set up the HEALPix projection
    nside = hdulist[1].header['NSIDE']
    order = hdulist[1].header['ORDERING']
    hp = HEALPix(nside=nside, order=order, frame=Galactic())
    
    #Get RA/DEC coords of observation
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit='m')
    time_array = uvd.time_array
    obstime_start = Time(time_array[0],format='jd',location=loc)
    obstime_end = Time(time_array[-1],format='jd',location=loc)
    zenith_start = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_start,location=loc)
    zenith_start = zenith_start.transform_to('icrs')
    zenith_end = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_end,location=loc)
    zenith_end = zenith_end.transform_to('icrs')
    lst_start = obstime_start.sidereal_time('mean').hour
    lst_end = obstime_end.sidereal_time('mean').hour
    start_coords = [zenith_start.ra.degree,zenith_start.dec.degree]
    if start_coords[0] > 180:
        start_coords[0] = start_coords[0] - 360
    end_coords = [zenith_end.ra.degree,zenith_end.dec.degree]
    if end_coords[0] > 180:
        end_coords[0] = end_coords[0] - 360
    
    # Sample a 300x200 grid in RA/Dec
    ra_range = [zenith_start.ra.degree-ra_pad, zenith_end.ra.degree+ra_pad]
    dec_range = [zenith_start.dec.degree-ra_pad, zenith_end.dec.degree+ra_pad]
    if clip == True:
        ra = np.linspace(ra_range[0],ra_range[1], nx)
        dec = np.linspace(dec_range[0],dec_range[1], ny)
    else:
        ra = np.linspace(-180,180,nx)
        dec = np.linspace(-90,zenith_start.dec.degree+90,ny)
    ra_grid, dec_grid = np.meshgrid(ra * u.deg, dec * u.deg)
    
    #Create alpha grid
    alphas = np.ones(ra_grid.shape)
    alphas = np.multiply(alphas,0.5)
    ra_min = np.argmin(np.abs(np.subtract(ra,start_coords[0]-fwhm/2)))
    ra_max = np.argmin(np.abs(np.subtract(ra,end_coords[0]+fwhm/2)))
    dec_min = np.argmin(np.abs(np.subtract(dec,start_coords[1]-fwhm/2)))
    dec_max = np.argmin(np.abs(np.subtract(dec,end_coords[1]+fwhm/2)))
    alphas[dec_min:dec_max, ra_min:ra_max] = 1

    # Set up Astropy coordinate objects
    coords = sc(ra_grid.ravel(), dec_grid.ravel(), frame='icrs')

    # Interpolate values
    temperature = healpy.read_map(map_path)
    tmap = hp.interpolate_bilinear_skycoord(coords, temperature)
    tmap = tmap.reshape((ny, nx))
    tmap = np.flip(tmap,axis=1)
    alphas = np.flip(alphas,axis=1)

    # Make a plot of the interpolated temperatures
    plt.figure(figsize=(12, 7))
    im = plt.imshow(tmap, extent=[ra[-1], ra[0], dec[0], dec[-1]], 
                    cmap=plt.cm.viridis, aspect='auto', vmin=10,vmax=40,alpha=alphas,origin='lower')
    plt.xlabel('RA (ICRS)')
    plt.ylabel('DEC (ICRS)')
    plt.hlines(y=start_coords[1]-fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    plt.hlines(y=start_coords[1]+fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    plt.vlines(x=start_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    plt.vlines(x=end_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    plt.annotate(np.around(lst_start,2),xy=(start_coords[0],dec[-1]),xytext=(0,8),
                 fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    plt.annotate(np.around(lst_end,2),xy=(end_coords[0],dec[-1]),xytext=(0,8),
                 fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    plt.annotate('LST (hours)',xy=(np.average([start_coords[0],end_coords[0]]),dec[-1]),
                xytext=(0,22),fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    for s in sources:
        if s[1] > dec[0] and s[1] < dec[-1]:
            if s[0] > 180:
                s = (s[0]-360,s[1],s[2])
            if s[2] == 'LMC' or s[2] == 'SMC':
                plt.annotate(s[2],xy=(s[0],s[1]),xycoords='data',fontsize=8,xytext=(20,-20),
                             textcoords='offset points',arrowprops=dict(facecolor='black', shrink=2,width=1,
                                                                        headwidth=4))
            else:
                plt.scatter(s[0],s[1],c='k',s=6)
                if len(s[2]) > 0:
                    plt.annotate(s[2],xy=(s[0]+3,s[1]-4),xycoords='data',fontsize=6)
    plt.show()
    plt.close()
    hdulist.close()

def plot_inspect_ants(uvd1,jd,badAnts=[],flaggedAnts={},tempAnts={},crossedAnts=[],use_ants='auto'):
#     status_use = ['RF_ok','digital_ok','calibration_maintenance','calibration_ok','calibration_triage']
    if use_ants == 'auto':
        use_ants = uvd1.get_ants()
    h = cm_active.get_active(at_date=jd, float_format="jd")
    inspectAnts = []
    for ant in use_ants:
#         status = h.apriori[f'HH{ant}:A'].status
        if ant in badAnts or ant in flaggedAnts.keys() or ant in crossedAnts:
            inspectAnts.append(ant)
    inspectAnts = np.unique(inspectAnts)
    inspectTitles = {}
    for ant in inspectAnts:
        inspectTitles[ant] = 'Flagged by: '
        if ant in badAnts:
            inspectTitles[ant] = f'{inspectTitles[ant]} correlation matrix,'
        if ant in flaggedAnts.keys():
            inspectTitles[ant] = f'{inspectTitles[ant]} ant_metrics,'
        if ant in crossedAnts:
            inspectTitles[ant] = f'{inspectTitles[ant]} cross matrix,'
        try:
            for k in tempAnts.keys():
                if ant in tempAnts[k]:
                    inspectTitles[ant] = f'{inspectTitles[ant]} template - {k},'
        except:
            continue
        if inspectTitles[ant][-1] == ',':
            inspectTitles[ant] = inspectTitles[ant][:-1]
    print('Antennas that require further inspection are:')
    print(inspectAnts)
    
    for ant in inspectAnts:
        auto_waterfall_lineplot(uvd1,ant,jd,title=inspectTitles[ant])
        
    return inspectAnts
    
def auto_waterfall_lineplot(uv, ant, jd, pols=['xx','yy'], colorbar_min=1e6, colorbar_max=1e8, title=''):
    freq = uv.freq_array[0]*1e-6
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2,0.7,1])
    it = 0
    pol_dirs = ['NN','EE']
    for p,pol in enumerate(pols):
        waterfall= plt.subplot(gs[it])
        jd_ax=plt.gca()
        times= np.unique(uv.time_array)
        d = np.abs(uv.get_data((ant,ant, pol)))
        if len(np.nonzero(d)[0])==0:
            print('#########################################')
            print(f'Data for antenna {ant} is entirely zeros')
            print('#########################################')
            plt.close()
            return
        im = plt.imshow(d,norm=colors.LogNorm(), 
                    aspect='auto')
        waterfall.set_title(f'{pol_dirs[p]} pol')
        freqs = uv.freq_array[0, :] / 1000000
        xticks = np.arange(0, len(freqs), 120)
        plt.xticks(xticks, labels =np.around(freqs[xticks],2))
        if p == 0:
            jd_ax.set_ylabel('JD')
            jd_yticks = [int(i) for i in np.linspace(0,len(times)-1,8)]
            jd_labels = np.around(times[jd_yticks],2)
            jd_ax.set_yticks(jd_yticks)
            jd_ax.set_yticklabels(jd_labels)
            jd_ax.autoscale(False)
        if p == 1:
            lst_ax = jd_ax.twinx()
            lst_ax.set_ylabel('LST (hours)')
            lsts = uv.lst_array*3.819719
            inds = np.unique(lsts,return_index=True)[1]
            lsts = [lsts[ind] for ind in sorted(inds)]
            lst_yticks = [int(i) for i in np.linspace(0,len(lsts)-1,8)]
            lst_labels = np.around([lsts[i] for i in lst_yticks],2)
            lst_ax.set_yticks(lst_yticks)
            lst_ax.set_yticklabels(lst_labels)
            lst_ax.set_ylim(jd_ax.get_ylim())
            lst_ax.autoscale(False)
            jd_ax.set_yticks([])
        line= plt.subplot(gs[it+2])
        averaged_data= np.abs(np.average(uv.get_data((ant,ant,pol)),0))
        plt.plot(freq,averaged_data)
        line.set_yscale('log')
        if p == 0:
            line.set_ylabel('Night Average')
        else:
            line.set_yticks([])
        line.set_xlim(freq[0],freq[-1])
        line.set_xticks([])
        
        line2 = plt.subplot(gs[it+4])
        dat = uv.get_data((ant,ant,pol))
        dat = np.abs(dat[len(dat)//2,:])
        plt.plot(freq,dat)
        line2.set_yscale('log')
        line2.set_xlabel('Frequency (MHz)')
        if p == 0:
            line2.set_ylabel('Single Slice')
        else:
            line2.set_yticks([])
        line2.set_xlim(freq[0],freq[-1])
        
        plt.setp(waterfall.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        cbar = plt.colorbar(im, pad= 0.2, orientation = 'horizontal')
        cbar.set_label('Power')
        it=1
    fig.suptitle(f'{ant}', fontsize=10,y=0.96)
    plt.annotate(title, xy=(0.5,0.94), ha='center',xycoords='figure fraction')
    plt.show()
    plt.close()

def plot_autos(uvdx, uvdy):
    ants = uvdx.get_ants()
    sorted_ants = sorted(ants)
    freqs = (uvdx.freq_array[0])*10**(-6)
    times = uvdx.time_array
    lsts = uvdx.lst_array  
    
    Nants = len(ants)
    Nside = 6
    Yside = Nants//6 + 1
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime

    xlim = (np.min(freqs), np.max(freqs))
    ylim = (0, 20)

    fig, axes = plt.subplots(Yside, Nside, figsize=(16,Yside*3))

    ptitle = 1.92/(Yside*3)
    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10,y=1+ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=1, wspace=0.05, hspace=0.3)
    k = 0
    for n,a in enumerate(ants):
        j = n%Nside
        i = n//Nside
        ax = axes[i,j]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        px, = ax.plot(freqs, 10*np.log10(np.abs(uvdx.get_data((a, a,'xx'))[t_index])), color='r', alpha=0.75, linewidth=1)
        py, = ax.plot(freqs, 10*np.log10(np.abs(uvdy.get_data((a, a,'yy'))[t_index])), color='b', alpha=0.75, linewidth=1)
        ax.grid(False, which='both')
        ax.set_title(f'{a}', fontsize=10)
        if k == 0:
            ax.legend([px, py], ['NN', 'EE'])
        if i == Yside-1:
            [t.set_fontsize(10) for t in ax.get_xticklabels()]
            ax.set_xlabel('freq (MHz)', fontsize=10)
        else:
            ax.set_xticklabels([])
        if j!=0:
            ax.set_yticklabels([])
        else:
            [t.set_fontsize(10) for t in ax.get_yticklabels()]
            ax.set_ylabel(r'$10\cdot\log$(amp)', fontsize=10)
        j += 1
        k += 1
    for k in range(j,Nside):
        axes[i,k].axis('off')
    plt.show()
    plt.close()
    
def plot_wfs(uvd, pol, mean_sub=False, save=False, jd=''):
#     amps = np.abs(uvd.data_array[:, :, :, pol].reshape(uvd.Ntimes, uvd.Nants_data, uvd.Nfreqs, 1))
    ants = uvd.get_ants()
    sorted_ants = sorted(ants)
    freqs = (uvd.freq_array[0])*10**(-6)
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    polnames = ['xx','yy']
    
    Nants = len(ants)
    Nside = 6
    Yside = Nants//6 + 1
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
    ptitle = 1.92/(Yside*3)
    fig, axes = plt.subplots(Yside, Nside, figsize=(16,Yside*3))
    if pol == 0:
        fig.suptitle("North Polarization", fontsize=14, y=1+ptitle)
    else:
        fig.suptitle("East Polarization", fontsize=14, y=1+ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=.1, right=.9, top=1, wspace=0.1, hspace=0.3)
    vmin = 0
    vmax = 2
    
    k = 0
    for n,a in enumerate(ants):
        j = n%Nside
        i = n//Nside
        ax = axes[i,j]
        dat = np.log10(np.abs(uvd.get_data(a,a,polnames[pol])))
        if mean_sub == True:
            ms = np.subtract(dat, np.nanmean(dat,axis=0))
            im = ax.imshow(ms, 
                       vmin = -0.04, vmax = 0.04, aspect='auto',interpolation='nearest')
        else:
            im = ax.imshow(dat, 
                           vmin = vmin, vmax = vmax, aspect='auto',interpolation='nearest')
        ax.set_title(f'{a}', fontsize=10)
        ax.grid(False, which='both')
        if i == Yside-1:
            xticks = [int(i) for i in np.linspace(0,len(freqs)-1,3)]
            xticklabels = np.around(freqs[xticks],0)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('Freq (MHz)', fontsize=10)
            [t.set_rotation(70) for t in ax.get_xticklabels()]
        else:
            ax.set_xticklabels([])
        if j!=0:
            ax.set_yticklabels([])
        else:
            yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
            yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
            [t.set_fontsize(12) for t in ax.get_yticklabels()]
            ax.set_ylabel('Time(LST)', fontsize=10)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_ylabel('Time(LST)', fontsize=10)
        if j==Nside-1:
            pos = ax.get_position()
            cbar_ax=fig.add_axes([0.91,pos.y0,0.01,pos.height])        
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(f'Node {n}',rotation=270, labelpad=15)
        j += 1
        k += 1
    for k in range(j,Nside):
        axes[i,k].axis('off')
    plt.show()
    plt.close()

#     for i,n in enumerate(inclNodes):
#         ants = nodes[n]['ants']
#         j = 0
#         for _,a in enumerate(sorted_ants):
#             if a not in ants:
#                 continue
# #             status = h.apriori[f'HH{a}:A'].status
# #             abb = status_abbreviations[status]
#             ax = axes[i,j]
#             dat = np.log10(np.abs(uvd.get_data(a,a,polnames[pol])))
#             if mean_sub == True:
#                 ms = np.subtract(dat, np.nanmean(dat,axis=0))
#                 im = ax.imshow(ms, 
#                            vmin = -0.07, vmax = 0.07, aspect='auto',interpolation='nearest')
#             else:
#                 im = ax.imshow(dat, 
#                                vmin = vmin, vmax = vmax, aspect='auto',interpolation='nearest')
#             ax.set_title(f'{a}', fontsize=10)
#             if i == len(inclNodes)-1:
#                 xticks = [int(i) for i in np.linspace(0,len(freqs)-1,3)]
#                 xticklabels = np.around(freqs[xticks],0)
#                 ax.set_xticks(xticks)
#                 ax.set_xticklabels(xticklabels)
#                 ax.set_xlabel('Freq (MHz)', fontsize=10)
#                 [t.set_rotation(70) for t in ax.get_xticklabels()]
#             else:
#                 ax.set_xticklabels([])
#             if j != 0:
#                 ax.set_yticklabels([])
#             else:
#                 yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
#                 yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
#                 [t.set_fontsize(12) for t in ax.get_yticklabels()]
#                 ax.set_ylabel('Time(LST)', fontsize=10)
#                 ax.set_yticks(yticks)
#                 ax.set_yticklabels(yticklabels)
#                 ax.set_ylabel('Time(LST)', fontsize=10)
#             j += 1
#         for k in range(j,maxants):
#             axes[i,k].axis('off')
#         pos = ax.get_position()
#         cbar_ax=fig.add_axes([0.91,pos.y0,0.01,pos.height])        
#         cbar = fig.colorbar(im, cax=cbar_ax)
#         cbar.set_label(f'Node {n}',rotation=270, labelpad=15)
#     if save is True:
#         plt.savefig(f'{jd}_mean_subtracted_per_node_{pol}.png',bbox_inches='tight',dpi=300)
#     plt.show()
#     plt.close()
    
    
def plot_mean_subtracted_wfs(uvd, use_ants, jd, pols=['xx','yy']):
    freqs = (uvd.freq_array[0])*1e-6
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    ants = sorted(use_ants)
    Nants = len(ants) 
    pol_labels = ['NN','EE']
    
#     status_colors = {
#         'dish_maintenance' : 'salmon',
#         'dish_ok' : 'red',
#         'RF_maintenance' : 'lightskyblue',
#         'RF_ok' : 'royalblue',
#         'digital_maintenance' : 'plum',
#         'digital_ok' : 'mediumpurple',
#         'calibration_maintenance' : 'lightgreen',
#         'calibration_ok' : 'green',
#         'calibration_triage' : 'lime'}
#     status_abbreviations = {
#         'dish_maintenance' : 'dish-M',
#         'dish_ok' : 'dish-OK',
#         'RF_maintenance' : 'RF-M',
#         'RF_ok' : 'RF-OK',
#         'digital_maintenance' : 'dig-M',
#         'digital_ok' : 'dig-OK',
#         'calibration_maintenance' : 'cal-M',
#         'calibration_ok' : 'cal-OK',
#         'calibration_triage' : 'cal-Tri'}
    h = cm_active.get_active(at_date=jd, float_format="jd")
    
    fig, axes = plt.subplots(Nants, 2, figsize=(7,Nants*2.2))
    fig.suptitle('Mean Subtracted Waterfalls')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.85, top=.975, wspace=0.05, hspace=0.2)

    for i,ant in enumerate(ants):
#         status = h.apriori[f'HH{ant}:A'].status
#         abb = status_abbreviations[status]
#         color = status_colors[status]
        for j,pol in enumerate(pols):
            ax = axes[i,j]
            dat = np.log10(np.abs(uvd.get_data(ant,ant,pol)))
            ms = np.subtract(dat, np.nanmean(dat,axis=0))
            im = ax.imshow(ms, 
                           vmin = -0.07, vmax = 0.07, aspect='auto',interpolation='nearest')
            ax.set_title(f'{ant} - {pol_labels[j]}', fontsize=10)
            if j != 0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
                yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            if i != Nants-1:
                ax.set_xticklabels([])
            else:
                xticks = [int(i) for i in np.linspace(0,len(freqs)-1,8)]
                xticklabels = np.around(freqs[xticks],0)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Frequency (MHz)', fontsize=10)
        if j == 1:
            pos = ax.get_position()
            cbar_ax=fig.add_axes([0.88,pos.y0,0.02,pos.height])
            fig.colorbar(im, cax=cbar_ax)
    fig.show()

def plot_closure(uvd, triad_length, pol):
    """Plot closure phase for an example triad.
    Parameters
    ----------
    files : list of strings
        List of data filenames
    triad_length : float {14., 29.}
        Length of the triangle segment length. Must be 14 or 29.
    pol : str {xx, yy}
        Polarization to plot.
    Returns
    -------
    None
    """


    if triad_length == 14.:
        triad_list = [[0, 11, 12], [0, 1, 12], [1, 12, 13], [1, 2, 13],
                      [2, 13, 14], [11, 23, 24], [11, 12, 24], [12, 24, 25],
                      [12, 13, 25], [13, 25, 26], [13, 14, 26], [14, 26, 27],
                      [23, 36, 37], [23, 24, 37], [24, 37, 38], [24, 25, 38],
                      [25, 38, 39], [25, 26, 39], [26, 39, 40], [26, 27, 40],
                      [27, 40, 41], [36, 37, 51], [37, 51, 52], [37, 38, 52],
                      [38, 52, 53], [38, 39, 53], [39, 53, 54], [39, 40, 54],
                      [40, 54, 55], [40, 41, 55], [51, 66, 67], [51, 52, 67],
                      [53, 54, 69], [54, 69, 70], [54, 55, 70], [55, 70, 71],
                      [65, 66, 82], [66, 82, 83], [66, 67, 83], [67, 83, 84],
                      [70, 71, 87], [120, 121, 140], [121, 140, 141], [121, 122, 141],
                      [122, 141, 142], [122, 123, 142], [123, 142, 143], [123, 124, 143]]
    else:
        triad_list = [[0, 23, 25], [0, 2, 25], [1, 24, 26], [2, 25, 27], [11, 36, 38],
                      [11, 13, 38], [12, 37, 39], [12, 14, 39], [13, 38, 40], [14, 39, 41],
                      [23, 25, 52], [24, 51, 53], [24, 26, 53], [25, 52, 54], [25, 27, 54],
                      [26, 53, 55], [36, 65, 67], [36, 38, 67], [38, 67, 69], [38, 40, 69],
                      [39, 41, 70], [40, 69, 71], [51, 82, 84], [51, 53, 84], [52, 83, 85],
                      [52, 54, 85], [54, 85, 87], [83, 85, 120], [85, 120, 122], [85, 87, 122],
                      [87, 122, 124]]


    # Look for a triad that exists in the data
    for triad in triad_list:
        bls = [[triad[0], triad[1]], [triad[1], triad[2]], [triad[2], triad[0]]]
        triad_in = True
        for bl in bls:
            inds = uvd.antpair2ind(bl[0], bl[1], ordered=False)
            if len(inds) == 0:
                triad_in = False
                break
        if triad_in:
            break

    if not triad_in:
        raise ValueError('Could not find triad in data.')

    closure_ph = np.angle(uvd.get_data(triad[0], triad[1], pol)
                          * uvd.get_data(triad[1], triad[2], pol)
                          * uvd.get_data(triad[2], triad[0], pol))
    plt.imshow(closure_ph, aspect='auto', rasterized=True,
                           interpolation='nearest', cmap = 'twilight')
    
def plotNodeAveragedSummary(uv,HHfiles,jd,use_ants,pols=['xx','yy'],mat_pols=['xx','yy'],
                            baseline_groups=[],removeBadAnts=False,plotRatios=False,plotSummary=True):
    """
    Plots a summary of baseline correlations throughout a night for each baseline group specified, separated into inter-node and intra-node baselines, for each polarization specified.
    
    Parameters
    ----------
    uv: UVData object
        UVData object containing any file from the desired night of observation.
    HHfiles: List
        A list of all files to be looked at for the desired night of observation.
    jd: String
        The JD of the night of observation
    pols: List
        A list containing the desired polarizations to look at. Options are any polarization strings accepted by pyuvdata. 
    baseline_groups: []
        A list containing the baseline types to look at, formatted as (length, N-S separation, label (str)).
    removeBadAnts: Bool
        Option to flag seemingly dead antennas and remove them from the per-baseline-group averaging. 
    
    Returns
    -------
    badAnts: List
        A list specifying the antennas flagged as dead or non-correlating.
    """
    if baseline_groups == []:
        baseline_groups = [(14,0,'14m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE'),(29,0,'29m E-W'),(29,22,'29m SW-NE'),
                       (44,0,'44m E-W'),(58.5,0,'58m E-W'),(73,0,'73m E-W'),(87.6,0,'88m E-W'),
                      (102.3,0,'102m E-W')]
    nodeMedians,lsts,badAnts=get_correlation_baseline_evolutions(uv,HHfiles,jd,use_ants,pols=pols,mat_pols=mat_pols,
                                                                bl_type=baseline_groups,removeBadAnts=removeBadAnts,
                                                                plotRatios=plotRatios)
#     pols = mat_pols
#     if plotSummary is False:
#         return badAnts
#     if len(lsts)>1:
#         fig,axs = plt.subplots(len(pols),2,figsize=(16,16))
#         maxLength = 0
#         cmap = plt.get_cmap('Blues')
#         for group in baseline_groups:
#             if group[0] > maxLength:
#                 maxLength = group[0]
#         for group in baseline_groups:
#             length = group[0]
#             data = nodeMedians[group[2]]
#             colorInd = float(length/maxLength)
#             if len(data['inter']['xx']) == 0:
#                 continue
#             for i in range(len(pols)):
#                 pol = pols[i]
#                 axs[i][0].plot(data['inter'][pol], color=cmap(colorInd), label=group[2])
#                 axs[i][1].plot(data['intra'][pol], color=cmap(colorInd), label=group[2])
#                 axs[i][0].set_ylabel('Median Correlation Metric')
#                 axs[i][0].set_title('Internode, Polarization %s' % pol)
#                 axs[i][1].set_title('Intranode, Polarization %s' % pol)
#                 xticks = np.arange(0,len(lsts),1)
#                 axs[i][0].set_xticks(xticks)
#                 axs[i][0].set_xticklabels([str(lst) for lst in lsts])
#                 axs[i][1].set_xticks(xticks)
#                 axs[i][1].set_xticklabels([str(lst) for lst in lsts])
#         axs[1][1].legend()
#         axs[1][0].set_xlabel('LST (hours)')
#         axs[1][1].set_xlabel('LST (hours)')
#         fig.tight_layout(pad=2)
#     else:
#         print('#############################################################################')
#         print('Not enough LST coverage to show metric evolution - that plot is being skipped')
#         print('#############################################################################')
#     return badAnts
    
def plotVisibilitySpectra(file,jd,use_ants='auto',badAnts=[],pols=['xx','yy']):
    """
    Plots visibility amplitude spectra for a set of redundant baselines, labeled by inter vs. intranode baselines.
    
    Parameters
    ---------
    file: String
        File to calculate the spectra from
    jd: String
        JD of the night 'file' was observed on
    badAnts: List
        A list of antennas not to include in the plot
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    """
    
    pol_labels = ['NS','EW']
    plt.subplots_adjust(wspace=0.25)
    uv = UVData()
    uv.read_uvh5(file)
    baseline_groups = get_baseline_groups(uv,use_ants="auto")
    freqs = uv.freq_array[0]/1000000
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    obstime_start = Time(uv.time_array[0],format='jd',location=loc)
    startTime = obstime_start.sidereal_time('mean').hour
    JD = int(obstime_start.jd)
    j = 0
    fig, axs = plt.subplots(len(baseline_groups),2,figsize=(12,4*len(baseline_groups)))
    for orientation in baseline_groups:
        bls = baseline_groups[orientation]
        usable = 0
        for i in range(len(bls)):
            ants = uv.baseline_to_antnums(bls[i])
            if ants[0] in badAnts or ants[1] in badAnts:
                continue
            if ants[0] in use_ants and ants[1] in use_ants:
                usable += 1
        if usable <=4:
            use_all = True
            print(f'Note: not enough baselines of orientation {orientation} - using all available baselines')
        elif usable <= 10:
            print(f'Note: only a small number of baselines of orientation {orientation} are available')
            use_all = False
        else:
            use_all = False
        for p in range(len(pols)):
            inter=False
            intra=False
            pol = pols[p]
            for i in range(len(bls)):
                ants = uv.baseline_to_antnums(bls[i])
                ant1 = ants[0]
                ant2 = ants[1]
                if (ant1 in use_ants and ant2 in use_ants) or use_all == True:
#                     key1 = 'HH%i:A' % (ant1)
#                     n1 = x[key1].get_part_from_type('node')['E<ground'][1:]
#                     key2 = 'HH%i:A' % (ant2)
#                     n2 = x[key2].get_part_from_type('node')['E<ground'][1:]
                    dat = np.mean(np.abs(uv.get_data(ant1,ant2,pol)),0)
                    auto1 = np.mean(np.abs(uv.get_data(ant1,ant1,pol)),0)
                    auto2 = np.mean(np.abs(uv.get_data(ant2,ant2,pol)),0)
                    norm = np.sqrt(np.multiply(auto1,auto2))
                    dat = np.divide(dat,norm)
                    if ant1 in badAnts or ant2 in badAnts:
                        continue
#                     if n1 == n2:
#                         if intra is False:
#                             axs[j][p].plot(freqs,dat,color='blue',label='intranode')
#                             intra=True
#                         else:
#                             axs[j][p].plot(freqs,dat,color='blue')
#                     else:
#                         if inter is False:
#                             axs[j][p].plot(freqs,dat,color='red',label='internode')
#                             inter=True
#                         else:
#                             axs[j][p].plot(freqs,dat,color='red')
                    axs[j][p].plot(freqs,dat,color='blue')
                    axs[j][p].set_yscale('log')
                    axs[j][p].set_title('%s: %s pol' % (orientation,pol_labels[p]))
                    if j == 0:
                        axs[len(baseline_groups)-1][p].set_xlabel('Frequency (MHz)')
            if p == 0:
                axs[j][p].legend()
        axs[j][0].set_ylabel('log(|Vij|)')
        axs[j][1].set_yticks([])
        j += 1
    fig.suptitle('Visibility spectra (JD: %i)' % (JD))
    fig.subplots_adjust(top=.94,wspace=0.05)
    plt.show()
    plt.close()
    
def plot_antenna_positions(uv, badAnts=[],use_ants='auto'):
    """
    Plots the positions of all antennas that have data, colored by node.
    
    Parameters
    ----------
    uv: UVData object
        Observation to extract antenna numbers and positions from
    badAnts: List
        A list of flagged or bad antennas. These will be outlined in black in the plot. 
    flaggedAnts: Dict
        A dict of antennas flagged by ant_metrics with value corresponding to color in ant_metrics plot
    """
    
    plt.figure(figsize=(12,10))
    if badAnts == None:
        badAnts = []
    all_ants = uv.antenna_numbers
#     nodes, antDict, inclNodes = generate_nodeDict(uv)
#     N = len(inclNodes)
    cmap = plt.get_cmap('tab20')
    i = 0
#     nodePos = geo_sysdef.read_nodes()
#     antPos = geo_sysdef.read_antennas()
#     ants = geo_sysdef.read_antennas()
#     nodes = geo_sysdef.read_nodes()
#     firstNode = True
    firstAnt = True
    for i,a in enumerate(all_ants):
        width = 0
        widthf = 0
        if a in badAnts:
            width = 2
        x = uv.antenna_positions[i,0]
        y = uv.antenna_positions[i,1]
        if a in use_ants:
            falpha = 0.5
        else:
            falpha = 0.1
        if firstAnt:
            if a in badAnts:
                if falpha == 0.1:
                    plt.plot(x,y,marker="h",markersize=40,alpha=falpha,color='b',
                        markeredgecolor='black',markeredgewidth=0)
                    plt.annotate(a, [x-1, y])
                    continue
                plt.plot(x,y,marker="h",markersize=40,alpha=falpha,color='b',
                    markeredgecolor='black',markeredgewidth=0)
                if a in badAnts:
                    plt.plot(x,y,marker="h",markersize=40,color='b',
                        markeredgecolor='black',markeredgewidth=width, markerfacecolor="None")
            else:
                if falpha == 0.1:
                    plt.plot(x,y,marker="h",markersize=40,alpha=falpha,color='b',
                        markeredgecolor='black',markeredgewidth=0)
                    plt.annotate(a, [x-1, y])
                    continue
                plt.plot(x,y,marker="h",markersize=40,alpha=falpha,color='b',
                    markeredgecolor='black',markeredgewidth=width)
            firstAnt = False
        else:
            plt.plot(x,y,marker="h",markersize=40,alpha=falpha,color='b',
                markeredgecolor='black',markeredgewidth=0)
            if a in badAnts and a in use_ants:
                plt.plot(x,y,marker="h",markersize=40,color='b',
                    markeredgecolor='black',markeredgewidth=width, markerfacecolor="None")
        plt.annotate(a, [x-1, y])
    plt.xlabel('East')
    plt.ylabel('North')
    plt.show()
    plt.close()
    
def plot_lst_coverage(uvd):
    """
    Plots the LST and JD coverage for a particular night.
    
    Parameters
    ----------
    uvd: UVData Object
        Object containing a whole night of data, used to extract the time array.
    """
    lsts = uvd.lst_array*3.819719
    jds = np.unique(uvd.time_array)
    alltimes = np.arange(np.floor(jds[0]),np.ceil(jds[0]),jds[2]-jds[1])
    df = jds[2]-jds[1]
    truetimes = [np.min(np.abs(jds-jd))<=df*0.6 for jd in alltimes]
    usetimes = np.tile(np.asarray(truetimes),(20,1))

    fig = plt.figure(figsize=(20,2))
    ax = fig.add_subplot()
    im = ax.imshow(usetimes, aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,interpolation='nearest')
    fig.colorbar(im)
    ax.set_yticklabels([])
    ax.set_yticks([])
    if len(alltimes) <= 15:
        xticks = [int(i) for i in np.linspace(0,len(alltimes)-1,len(alltimes))]
    else:
        xticks = [int(i) for i in np.linspace(0,len(alltimes)-1,14)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(alltimes[xticks],2))
    ax.set_xlabel('JD')
    ax.set_title('LST (hours)')
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)
    jds = alltimes[xticks]
    lstlabels = []
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit='m')
    for jd in jds:
        t = Time(jd,format='jd',location=loc)
        lstlabels.append(t.sidereal_time('mean').hour)
    ax2.set_xticklabels(np.around(lstlabels,2))
    ax2.set_label('LST (hours)')
    ax2.tick_params(labelsize=12)
    plt.show()
    plt.close()
    
    
def calcEvenOddAmpMatrix(sm,pols=['xx','yy'],nodes='auto', badThresh=0.25, plotRatios=False):
    """
    Calculates a matrix of phase correlations between antennas, where each pixel is calculated as (even/abs(even)) * (conj(odd)/abs(odd)), and then averaged across time and frequency.
    
    Paramters:
    ---------
    sm: UVData Object
        Sum observation.
    df: UVData Object
        Diff observation. Must be the same time of observation as sm. 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    nodes: String or List
        Nodes to include in matrix. Default is 'auto', which generates a list of all nodes included in the provided data files. 
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas.
    
    Returns:
    -------
    data: Dict
        Dictionary containing calculated values, formatted as data[polarization][ant1,ant2]. 
    badAnts: List
        List of antennas that were flagged as bad based on badThresh.
    """
    nants = len(sm.get_ants())
    data = {}
    antnumsAll = sorted(sm.get_ants())
    badAnts = []
    for p in range(len(pols)):
        pol = pols[p]
        data[pol] = np.empty((nants,nants))
        for i in range(len(antnumsAll)):
            thisAnt = []
            for j in range(len(antnumsAll)):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                dat = sm.get_data(ant1,ant2,pol)
                even = dat[::2,:]
#                 print('Even')
#                 print(even)
                odd = dat[1::2,:]
#                 s = sm.get_data(ant1,ant2,pol)
#                 d = df.get_data(ant1,ant2,pol)
#                 even = (s + d)/2
                even = np.divide(even,np.abs(even))
#                 print('Even norm')
#                 print(even)
#                 odd = (s - d)/2
                odd = np.divide(odd,np.abs(odd))
                product = np.multiply(even,np.conj(odd))
#                 print('product')
#                 print(product)
                data[pol][i,j] = np.abs(np.nanmean(product))
                thisAnt.append(np.abs(np.mean(product)))
            pgood = np.count_nonzero(~np.isnan(thisAnt))/len(thisAnt)
            if (np.nanmedian(thisAnt) < badThresh or pgood<0.2) and antnumsAll[i] not in badAnts:
                if pol[0]==pol[1]:
                    #Don't assign bad ants based on cross pols
                    badAnts.append(antnumsAll[i])
    if plotRatios is True:
        if len(pols) == 4:
            data['xx-xy'] = np.subtract(data['xx'],data['xy'])
            data['xx-yx'] = np.subtract(data['xx'],data['yx'])
            data['yy-xy'] = np.subtract(data['yy'],data['xy'])
            data['yy-yx'] = np.subtract(data['yy'],data['yx'])
        else:
            print('Can only calculate differences if cross pols were specified')
        polAnts = {}
        badAnts = []
        subs = ['xx-xy','xx-yx','yy-xy','yy-yx']
        for k in subs:
            for i,ant in enumerate(antnumsAll):
                dat = data[k][i,:]
                if np.nanmedian(dat) < 0:
                    if ant in polAnts.keys():
                        polAnts[ant] = polAnts[ant] + 1
                    else:
                        polAnts[ant] = 1
                    if polAnts[ant] == 4:
                        badAnts.append(ant)  
    return data, badAnts


def plotCorrMatrix(uv,data,pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False,plotRatios=False):
    """
    Plots a matrix representing the phase correlation of each baseline.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation used for calculating the correlation metric
    data: Dict
        Dictionary containing the correlation metric for each baseline and each polarization. Formatted as data[polarization]  [ant1,ant2] 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    vminIn: float
        Lower limit of colorbar. Default is 0.
    vmaxIn: float
        Upper limit of colorbar. Default is 1.
    nodes: Dict
        Dictionary containing the nodes (and their constituent antennas) to include in the matrix. Formatted as nodes[Node #][Ant List, Snap # List, Snap Location List].
    logScale: Bool
        Option to put colormap on a logarithmic scale. Default is False.
    """
#     if nodes=='auto':
#         nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.get_ants())
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(2,2,figsize=(16,16))
    dirs = ['NN','EE','NE','EN']
    cmap='plasma'
    if plotRatios is True:
        pols = ['xx-xy','yy-xy','xx-yx','yy-yx']
        dirs=pols
        vminIn=-1
        cmap='seismic'
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    antnumsAll = sorted(uv.get_ants())
    i = 0
    for p in range(len(pols)):
        if p >= 2:
            i=1
        pol = pols[p]
        nants = len(antnumsAll)
        if logScale is True:
            im = axs[i][p%2].imshow(data[pol],cmap=cmap,origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
        else:
            im = axs[i][p%2].imshow(data[pol],cmap=cmap,origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
        axs[i][p%2].set_xticks(np.arange(0,nantsTotal)+1)
        axs[i][p%2].set_xticklabels(antnumsAll,rotation=90,fontsize=6)
        axs[i][p%2].xaxis.set_ticks_position('top')
        axs[i][p%2].set_title('polarization: ' + dirs[p] + '\n')
#         n=0
#     n=0
#     for node in sorted(inclNodes):
#         n += len(nodeDict[node]['ants'])
#         axs[0][1].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
#         axs[1][1].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
#     axs[0][1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][1].transAxes)
    axs[0][1].set_yticklabels([])
    axs[0][1].set_yticks([])
    axs[0][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[0][0].set_ylabel('Antenna Number')
#     axs[1][1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[1][1].transAxes)
    axs[1][1].set_yticklabels([])
    axs[1][1].set_yticks([])
    axs[1][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[1][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[1][0].set_ylabel('Antenna Number')
    cbar_ax = fig.add_axes([0.98,0.18,0.015,0.6])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Correlation Matrix - JD: %s, LST: %.0fh' % (str(jd),np.round(lst,0)))
    fig.subplots_adjust(top=1.28,wspace=0.05,hspace=1.1)
    fig.tight_layout(pad=2)
    plt.show()
    plt.close()
    
def plot_single_matrix(uv,data,vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.get_ants())
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,1,figsize=(16,16))
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    antnumsAll = sort_antennas(uv)
    nants = len(antnumsAll)
    if logScale is True:
        im = axs[0][0].imshow(data[pol],cmap='plasma',origin='upper',
                                extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
    else:
        im = axs[0][0].imshow(data[pol],cmap='plasma',origin='upper',extent=
                                [0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
    axs[0][0].set_xticks(np.arange(0,nantsTotal)+1)
    axs[0][0].set_xticklabels(antnumsAll,rotation=90,fontsize=6)
    axs[0][0].xaxis.set_ticks_position('top')
    axs[0][0].set_title('polarization: ' + dirs[p] + '\n')
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[0][0].axhline(len(antnumsAll)-n+.5,lw=4)
        axs[0][0].axvline(n+.5,lw=4)
        axs[0][0].text(n-len(nodeDict[node]['ants'])/2,-.5,node)
    axs[0][0].text(.42,-.05,'Node Number',transform=axs[0][0].transAxes)
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[0][0].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
    axs[0][0].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][0].transAxes)
    axs[0][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[0][0].set_ylabel('Antenna Number')
    axs[0][0].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][0].transAxes)
    cbar_ax = fig.add_axes([0.98,0.18,0.015,0.6])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Correlation Matrix - JD: %s, LST: %.0fh' % (str(jd),np.round(lst,0)))
    fig.subplots_adjust(top=1.28,wspace=0.05,hspace=1.1)
    fig.tight_layout(pad=2)
    plt.show()
    plt.close()
    
def get_hourly_files(uv, HHfiles, jd):
    """
    Generates a list of files spaced one hour apart throughout a night of observation, and the times those files were observed.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the given night, used only for grabbing the telescope location
    HHFiles: List
        List of all files from the desired night of observation
    jd: String
        JD of the night of observation
        
    Returns:
    -------
    use_files: List
        List of files separated by one hour
    use_lsts: List
        List of LSTs of observations in use_files
    """
    use_lsts = []
    use_files = []
    use_file_inds = []
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    for i,file in enumerate(HHfiles):
        try:
            dat = UVData()
            dat.read(file, read_data=False)
        except KeyError:
            continue
        jd = dat.time_array[0]
        t = Time(jd,format='jd',location=loc)
        lst = round(t.sidereal_time('mean').hour,2)
        if np.round(lst,0) == 24:
            continue
        if np.abs((lst-np.round(lst,0)))<0.05:
            if len(use_lsts)>0 and np.abs(use_lsts[-1]-lst)<0.5:
                if np.abs((lst-np.round(lst,0))) < abs((use_lsts[-1]-np.round(lst,0))):
                    use_lsts[-1] = lst
                    use_files[-1] = file
                    use_file_inds[-1] = i
            else:
                use_lsts.append(lst)
                use_files.append(file)
                use_file_inds.append(i)
    return use_files, use_lsts, use_file_inds

def get_baseline_groups(uv, bl_groups=[(14,0,'14m E-W'),(29,0,'29m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE')],
                       use_ants='auto'):
    """
    Generate dictionary containing baseline groups.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation to extract antenna position information from
    bl_groups: List
        Desired baseline types to extract, formatted as (length (float), N-S separation (float), label (string))
        
    Returns:
    --------
    bls: Dict
        Dictionary containing list of lists of redundant baseline numbers, formatted as bls[group label]
    """
    
    bls={}
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        for group in bl_groups:
            if np.abs(lengths[i]-group[0])<1:
                ant1 = uv.baseline_to_antnums(bl[0])[0]
                ant2 = uv.baseline_to_antnums(bl[0])[1]
                if use_ants == 'auto' or (ant1 in use_ants and ant2 in use_ants):
                    antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                    antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                    disp = (antPos2-antPos1)[0][0]
                    if np.abs(disp[2]-group[1])<0.5:
                        bls[group[2]] = bl
    return bls


    
def get_correlation_baseline_evolutions(uv,HHfiles,jd,use_ants='auto',badThresh=0.35,pols=['xx','yy'],bl_type=(14,0,'14m E-W'),
                                        removeBadAnts=False, plotMatrix=True,mat_pols=['xx','yy','xy','yx'],plotRatios=False):
    """
    Calculates the average correlation metric for a set of redundant baseline groups at one hour intervals throughout a night of observation.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the desired night, used only for getting telescope location information.
    HHfiles: List
        List of all files for a night of observation
    jd: String
        JD of the given night of observation
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas. Default is 0.35.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    bl_type: Tuple
        Redundant baseline group to calculate correlation metric for. Default is 14m E-W baselines
    removeBadAnts: Bool
        Option to exclude antennas marked as bad from calculation. Default is False.
    plotMatrix: Bool
        Option to plot the correlation matrix for observations once each hour. Default is True.
        
    Returns:
    -------
    result: Dict
        Per hour correlation metric, formatted as result[baseline type]['inter' or 'intra'][polarization]
    lsts: List
        LSTs that metric was calculated for, spaced 1 hour apart.
    bad_antennas: List
        Antenna numbers flagged as bad based on badThresh parameter.
    """
    files, lsts, inds = get_hourly_files(uv, HHfiles, jd)
    if use_ants == 'auto':
        use_ants = uv.get_ants()
    if plotRatios is True:
        files = [files[len(files)//2]]
        nTimes=1
    else:
        nTimes = len(files)
    if nTimes > 3:
        plotTimes = [0,nTimes//2,nTimes-1]
    else:
        plotTimes = np.arange(0,nTimes,1)
#     nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    JD = math.floor(uv.time_array[0])
    bad_antennas = []
    pols = mat_pols
#     corrSummary = generateDataTable(uv,pols=pols)
    result = {}
    for f in range(nTimes):
        file = files[f]
        ind = inds[f]
        sm = UVData()
        df = UVData()
        try:
#             print(f'Trying to read {file}')
            sm.read(file, skip_bad_files=True, antenna_nums=use_ants)
#             dffile = '%sdiff%s' % (file[0:-8],file[-5:])
#             df.read(dffile, skip_bad_files=True, antenna_nums=use_ants)
        except:
            i = -5
            read = False
            while i<5 and read==False:
                try:
                    file = HHfiles[ind+i]
#                     print(f'trying to read {file}')
                    sm.read(file, skip_bad_files=True, antenna_nums=use_ants)
#                     dffile = '%sdiff%s' % (file[0:-8],file[-5:])
#                     df.read(dffile, skip_bad_files=True, antenna_nums=use_ants)
                    read = True
                except:
                    i += 1
            if read == False:
                print(f'WARNING: unable to read {file}')
                continue
        if f in plotTimes:
#             print(f)
#             print(mat_pols)
            matrix, badAnts = calcEvenOddAmpMatrix(sm,nodes='auto',pols=mat_pols,badThresh=badThresh,plotRatios=plotRatios)
#             print(badAnts)
#             print(np.shape(matrix))
#             print(np.max(matrix))
#             print(matrix)
        if plotMatrix is True and f in plotTimes:
            plotCorrMatrix(sm, matrix, pols=mat_pols, nodes='auto',plotRatios=plotRatios)
        result = None
    return result,lsts,bad_antennas

# def generateDataTable(uv,pols=['xx','yy']):
#     """
#     Simple helper function to generate an empty dictionary of the format desired for get_correlation_baseline_evolutions()
    
#     Parameters:
#     ----------
#     uv: UVData Object
#         Sample observation to extract node and antenna information from.
#     pols: List
#         Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
        
#     Returns:
#     -------
#     dataObject: Dict
#         Empty dict formatted as dataObject[node #][polarization]['inter' or 'intra']
#     """
    
#     nodeDict, antDict, inclNodes = generate_nodeDict(uv)
#     dataObject = {}
#     for node in nodeDict:
#         dataObject[node] = {}
#         for pol in pols:
#             dataObject[node][pol] = {
#                 'inter' : [],
#                 'intra' : []
#             }
#     return dataObject


def get_baseline_type(uv,bl_type=(14,0,'14m E-W'),use_ants='auto'):
    """
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to get baseline information from.
    bl_type: Tuple
        Redundant baseline group to extract baseline numbers for. Formatted as (length, N-S separation, label).
    
    Returns:
    -------
    bl: List
        List of lists of redundant baseline numbers. Returns None if the provided bl_type is not found.
    """
    
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        if np.abs(lengths[i]-bl_type[0])<1:
            ant1 = uv.baseline_to_antnums(bl[0])[0]
            ant2 = uv.baseline_to_antnums(bl[0])[1]
            if (ant1 in use_ants and ant2 in use_ants) or use_ants == 'auto':
                antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                disp = (antPos2-antPos1)[0][0]
                if np.abs(disp[2]-bl_type[1])<0.5:
                    return bl
    return None


def plot_crosses(uvd, ref_ant):
    ants = uvd.get_ants()
    freqs = (uvd.freq_array[0])*10**(-6)
    times = uvd.time_array
    lsts = uvd.lst_array
    
    Nants = len(ants)
#     Nside = int(np.ceil(np.sqrt(Nants)))*3
    Nside = 4
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime

    xlim = (np.min(freqs), np.max(freqs))
    ylim = (60, 90)

    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2, Nside*60))

    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
            if k < Nants:
                px, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'xx')),axis=0))), color='red', alpha=0.75, linewidth=1)
                py, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'yy')),axis=0))), color='darkorange', alpha=0.75, linewidth=1)
                pxy, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'xy')),axis=0))), color='royalblue', alpha=0.75, linewidth=1)
                pyx, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'yx')),axis=0))), color='darkviolet', alpha=0.75, linewidth=1)
            
                ax.grid(False, which='both')
                ax.set_title(str(ants[k]), fontsize=14)
            
                if k == 0:
                    ax.legend([px, py, pxy, pyx], ['XX', 'YY', 'XY','YX'])
            
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r'$10\cdot\log_{10}$ amplitude', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel('freq (MHz)', fontsize=10)
            k += 1
    fig.show()
    plt.close()
    
def gather_source_list():
    sources = []
    sources.append((50.6750,-37.2083,'Fornax A'))
    sources.append((201.3667,-43.0192,'Cen A'))
    # sources.append((83.6333,22.0144,'Taurus A'))
    sources.append((252.7833,4.9925,'Hercules A'))
    sources.append((139.5250,-12.0947,'Hydra A'))
    sources.append((79.9583,-45.7789,'Pictor A'))
    sources.append((187.7042,12.3911,'Virgo A'))
    sources.append((83.8208,-59.3897,'Orion A'))
    sources.append((80.8958,-69.7561,'LMC'))
    sources.append((13.1875,-72.8286,'SMC'))
    sources.append((201.3667,-43.0192,'Cen A'))
    sources.append((83.6333,20.0144,'Crab Pulsar'))
    sources.append((128.8375,-45.1764,'Vela SNR'))
    cat_path = f'{DATA_PATH}/G4Jy_catalog.tsv'
    cat = open(cat_path)
    f = csv.reader(cat,delimiter='\n')
    for row in f:
        if len(row)>0 and row[0][0]=='J':
            s = row[0].split(';')
            tup = (float(s[1]),float(s[2]),'')
            sources.append(tup)
    return sources

# def clean_ds(HHfiles, difffiles, bls, area=1000., tol=1e-9, skip_wgts=0.2): 
    
#     uvd_ds = UVData()
#     uvd_ds.read(HHfiles[0], bls=bls[0], polarizations=-5, keep_all_metadata=False)
#     times = np.unique(uvd_ds.time_array)
#     Nfiles = int(1./((times[-1]-times[0])*24.))
#     try:
#         uvd_ds.read(HHfiles[:Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
#         uvd_ds.flag_array = np.zeros_like(uvd_ds.flag_array)
#         uvd_diff = UVData()
#         uvd_diff.read(difffiles[:Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
#     except:
#         uvd_ds.read(HHfiles[len(HHfiles)//2:len(HHfiles)//2+Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
#         uvd_ds.flag_array = np.zeros_like(uvd_ds.flag_array)
#         uvd_diff = UVData()
#         uvd_diff.read(difffiles[len(HHfiles)//2:len(HHfiles)//2+Nfiles], bls=bls, polarizations=[-5,-6], keep_all_metadata=False)
    
#     uvf_m, uvf_fws = hera_qm.xrfi.xrfi_h1c_pipe(uvd_ds, sig_adj=1, sig_init=3)
#     hera_qm.xrfi.flag_apply(uvf_m, uvd_ds)
    
#     freqs = uvd_ds.freq_array[0]
#     FM_idx = np.searchsorted(freqs*1e-6, [85,110])
#     flag_FM = np.zeros(freqs.size, dtype=bool)
#     flag_FM[FM_idx[0]:FM_idx[1]] = True
#     win = dspec.gen_window('bh7', freqs.size)
    
#     pols = ['nn','ee']
    
#     _data_sq_cleaned, data_rs = {}, {}
#     for bl in bls:
#         for i, pol in enumerate(pols):
#             key = (bl[0],bl[1],pol)
#             data = uvd_ds.get_data(key)
#             diff = uvd_diff.get_data(key)
#             wgts = (~uvd_ds.get_flags(key)*~flag_FM[np.newaxis,:]).astype(float)

#             d_even = (data+diff)*0.5
#             d_odd = (data-diff)*0.5
#             d_even_cl, d_even_rs, _ = dspec.high_pass_fourier_filter(d_even, wgts, area*1e-9, freqs[1]-freqs[0], 
#                                                                      tol=tol, skip_wgt=skip_wgts, window='bh7')
#             d_odd_cl, d_odd_rs, _ = dspec.high_pass_fourier_filter(d_odd, wgts, area*1e-9, freqs[1]-freqs[0],
#                                                                    tol=tol, skip_wgt=skip_wgts, window='bh7')

#             idx = np.where(np.mean(np.abs(d_even_cl), axis=1) == 0)[0]
#             d_even_cl[idx] = np.nan
#             d_even_rs[idx] = np.nan        
#             idx = np.where(np.mean(np.abs(d_odd_cl), axis=1) == 0)[0]
#             d_odd_cl[idx] = np.nan
#             d_odd_rs[idx] = np.nan

#             _d_even = np.fft.fftshift(np.fft.ifft((d_even_cl+d_even_rs)*win), axes=1)
#             _d_odd = np.fft.fftshift(np.fft.ifft((d_odd_cl+d_odd_rs)*win), axes=1)

#             _data_sq_cleaned[key] = _d_odd.conj()*_d_even
#             data_rs[key] = d_even_rs
        
#     return _data_sq_cleaned, data_rs, uvd_ds, uvd_diff

# def plot_wfds(uvd, _data_sq, pol):    
#     ants = uvd.get_ants()
#     freqs = uvd.freq_array[0]
#     times = uvd.time_array
#     lsts = uvd.lst_array
#     taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
#     Nants = len(ants)
#     Nside = int(np.ceil(np.sqrt(Nants)))
#     Yside = int(np.ceil(float(Nants)/Nside))
    
#     t_index = 0
#     jd = times[t_index]
#     utc = Time(jd, format='jd').datetime
    
    
#     fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2,Nside*2))
#     if pol == 'ee':
#         fig.suptitle("delay spectrum (auto) waterfalls from {0} -- {1} East Polarization".format(times[0], times[-1]), fontsize=14)
#     else:
#         fig.suptitle("delay spectrum (auto) waterfalls from {0} -- {1} North Polarization".format(times[0], times[-1]), fontsize=14)
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
#     fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

#     k = 0
#     for i in range(Yside):
#         for j in range(Nside):
#             ax = axes[i,j]
#             if k < Nants:
#                 key = (ants[k], ants[k], pol)
#                 ds = 10.*np.log10(np.sqrt(np.abs(_data_sq[key])/np.abs(_data_sq[key]).max(axis=1)[:,np.newaxis]))
#                 im = ax.imshow(ds, aspect='auto', rasterized=True,
#                            interpolation='nearest', vmin = -50, vmax = -30, 
#                            extent=[taus[0]*1e9, taus[-1]*1e9, np.max(lsts), np.min(lsts)])
        
#                 ax.set_title(str(ants[k]), fontsize=10)
#             else:
#                 ax.axis('off')
#             if j != 0:
#                 ax.set_yticklabels([])
#             else:
#                 [t.set_fontsize(12) for t in ax.get_yticklabels()]
#                 ax.set_ylabel('Time(LST)', fontsize=10)
#             if i != Yside-1:
#                 ax.set_xticklabels([])
#             else:
#                 [t.set_fontsize(10) for t in ax.get_xticklabels()]
#                 [t.set_rotation(25) for t in ax.get_xticklabels()]
#                 ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#                 ax.set_xlabel('Delay (ns)', fontsize=10)
#             k += 1
        
#     cbar_ax=fig.add_axes([0.95,0.15,0.02,0.7])        
#     cb = fig.colorbar(im, cax=cbar_ax)
#     cb.set_label('dB')
#     fig.show()
    
# def plot_ds(uvd, uvd_diff, _data_sq_cleaned, data_rs, skip_wgts=0.2):            
#     matplotlib.rcParams.update({'font.size': 8})
#     freqs = uvd.freq_array[0]
#     taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
#     FM_idx = np.searchsorted(freqs*1e-6, [85,110])
#     flag_FM = np.zeros(freqs.size, dtype=bool)
#     flag_FM[FM_idx[0]:FM_idx[1]] = True
    
#     ants = np.sort(uvd.get_ants())
#     pols = ['nn','ee']
#     colors = ['b','r']
#     nodes, antDict, inclNodes = generate_nodeDict(uvd)

#     for i, ant in enumerate(ants):
#         i2 = i % 2
#         if(i2 == 0):
#             fig = plt.figure(figsize=(10, 3), dpi=110)
#             grid = plt.GridSpec(5, 4, wspace=0.55, hspace=2)
#         for j, pol in enumerate(pols):
#             key = (ant,ant,pol)

#             ax = fig.add_subplot(grid[:5, 0+i2*2])

#             ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
#             _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
#             ns_ave = np.abs(np.nanmean(_diff, axis=0))
#             ns_ave_ = np.sqrt(np.abs(np.mean(_diff.conj()*_diff, axis=0))/(2*_diff.shape[0]))
#             norm = np.max(ds_ave)
#             plt.plot(taus*1e9, 10*np.log10(ds_ave/norm), color=colors[j], label=pols[j], linewidth=0.7)
#             plt.plot(taus*1e9, 10*np.log10(ns_ave/norm), color=colors[j], alpha=0.5, linewidth=0.5)
#             plt.plot(taus*1e9, 10*np.log10(ns_ave_/norm), color=colors[j], ls='--', linewidth=0.5)
#             plt.axvspan(250, 500, alpha=0.2, facecolor='y')
#             plt.axvspan(2500, 3000, alpha=0.2, facecolor='g')
#             plt.axvspan(3500, 4000, alpha=0.2, facecolor='m')
            
#             plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=False)
#             plt.xlim(0,4500)
#             plt.ylim(-60,0)
#             plt.title('ant {}'.format(ant))
#             plt.xlabel(r'$\tau$ (ns)')
#             plt.ylabel(r'$|\tilde{V}(\tau)|$ in dB')
#             for yl in range(-50,0,10):
#                 plt.axhline(y=yl, color='k', lw=0.5, alpha=0.1)
#             if(j == 0):
#                 plt.annotate('node {} (snap {})'.format(antDict[ant]['node'],antDict[ant]['snapLocs'][0]), xy=(0.04,0.930), xycoords='axes fraction')

#             ax2 = fig.add_subplot(grid[:3, 1+i2*2])
#             auto = np.abs(uvd.get_data(key))
#             auto_ave = np.mean(auto/np.median(auto, axis=1)[:,np.newaxis], axis=0)
#             wgts = (~uvd.get_flags(key)*~flag_FM[np.newaxis,:]).astype(float)
#             wgts = np.where(wgts > skip_wgts, wgts, 0)
#             auto_flagged = auto/wgts
#             auto_flagged[np.isinf(auto_flagged)] = np.nan
#             auto_flagged_ave = np.nanmean(auto_flagged/np.nanmedian(auto_flagged, axis=1)[:,np.newaxis], axis=0)

#             plt.plot(freqs/1e6, np.log10(auto_ave), linewidth=1.0, color=colors[j], label='autocorrelation')
#             plt.plot(freqs/1e6, np.log10(auto_flagged_ave*0.7), linewidth=1.0, color=colors[j], alpha=0.5)
#             if(j == 0):
#                 plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False, handlelength=0, handletextpad=0)
#             plt.xlim(freqs.min()/1e6, freqs.max()/1e6)
#             plt.ylabel(r'log$_{10}(|V(\nu)|)$')
#             plt.ylim(-1, 0.7)

#             ax3 = fig.add_subplot(grid[3:5, 1+i2*2])
#             data_rs_ave = np.nanmean(data_rs[key]/np.nanmedian(auto_flagged, axis=1)[:,np.newaxis], axis=0)
#             plt.plot(freqs/1e6, np.log10(np.abs(data_rs_ave)), linewidth=1.0, color=colors[j], label='clean residual')
#             if(j == 0):
#                 plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), frameon=False, handlelength=0, handletextpad=0)
#             plt.xlim(freqs.min()/1e6, freqs.max()/1e6)
#             plt.ylim(-5,0)
#             plt.xlabel(r'$\nu$ (MHz)')
#             plt.ylabel(r'log$_{10}(|V(\nu)|)$')
#     matplotlib.rcParams.update({'font.size': 10})
            
# def plot_ds_diagnosis(uvd_diff, _data_sq_cleaned):
#     freqs = uvd_diff.freq_array[0]
#     taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
#     ants = np.sort(uvd_diff.get_ants())
#     pols = ['nn','ee']
#     colors = ['b','r']
#     nodes, antDict, inclNodes = generate_nodeDict(uvd_diff)
#     marker = ['o', 'x', 'P', 's', 'd', 'p', '<', 'h', '+', '>', 'X', '*']
    
#     nodes = []
#     ds_ave_250_500_nn, ds_ave_250_500_ee = [], []
#     ds_peak_2500_3000_nn, ds_peak_2500_3000_ee = [], []
#     ds_ratio_3500_4000_nn, ds_ratio_3500_4000_ee = [], []
#     for ant in ants:
#         nodes.append(int(antDict[ant]['node']))
#         for pol in pols:
#             key = (ant,ant,pol)            
#             ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
#             _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
#             ns_ave = np.sqrt(np.abs(np.mean(_diff.conj()*_diff, axis=0))/(2*_diff.shape[0]))
#             norm = np.max(ds_ave)
#             idx_region1 = (np.abs(taus)*1e9 > 250) * (np.abs(taus)*1e9 < 500)
#             idx_region2 = (np.abs(taus)*1e9 > 2500) * (np.abs(taus)*1e9 < 3000)
#             idx_region_out = (np.abs(taus)*1e9 > 3000) * (np.abs(taus)*1e9 < 3200)
#             idx_region3 = (np.abs(taus)*1e9 > 3500) * (np.abs(taus)*1e9 < 4000)
#             if(pol == 'nn'):
#                 ds_ave_250_500_nn.append(np.nanmean(ds_ave[idx_region1]/norm))
#                 ds_peak_2500_3000_nn.append(np.std(ds_ave[idx_region2])/np.std(ds_ave[idx_region_out]))
#                 ds_ratio_3500_4000_nn.append(np.nanmean(ds_ave[idx_region3])/np.mean(ns_ave[idx_region3]))
#             else:
#                 ds_ave_250_500_ee.append(np.nanmean(ds_ave[idx_region1]/norm))
#                 ds_peak_2500_3000_ee.append(np.std(ds_ave[idx_region2])/np.std(ds_ave[idx_region_out]))
#                 ds_ratio_3500_4000_ee.append(np.nanmean(ds_ave[idx_region3])/np.mean(ns_ave[idx_region3]))
#     nodes = np.array(nodes)
#     N_nodes = nodes.size
#     ds_ave_250_500_nn = np.array(ds_ave_250_500_nn)
#     ds_ave_250_500_ee = np.array(ds_ave_250_500_ee)
#     ds_peak_2500_3000_nn = np.array(ds_peak_2500_3000_nn)
#     ds_peak_2500_3000_ee = np.array(ds_peak_2500_3000_ee)
#     ds_ratio_3500_4000_nn = np.array(ds_ratio_3500_4000_nn)
#     ds_ratio_3500_4000_ee = np.array(ds_ratio_3500_4000_ee)
            
#     plt.figure(figsize=(15,15))
#     plt.subplot(3,1,1)
#     node_number = np.unique(nodes)
#     for i, node in enumerate(node_number):
#         idx_ant = np.where(nodes == node)[0]
#         plt.plot(ants[idx_ant], 10*np.log10(ds_ave_250_500_nn[idx_ant]), 'bo', label='nn', marker=marker[i%N_nodes])
#         plt.plot(ants[idx_ant], 10*np.log10(ds_ave_250_500_ee[idx_ant]), 'ro', label='ee', marker=marker[i%N_nodes])
#         if(i == 0):
#             plt.legend()
#     for i, ant in enumerate(ants):
#         plt.annotate('{}'.format(ant), xy=(ants[i], 10*np.log10(ds_ave_250_500_nn[i])))
#     plt.xlabel('Antenna number')
#     plt.ylabel('dB (relative to the peak)')
#     plt.title('Averaged delay spectrum at 250-500 ns')
#     plt.subplot(3,1,2)
#     for i, node in enumerate(node_number):
#         idx_ant = np.where(nodes == node)[0]
#         plt.plot(ants[idx_ant], ds_peak_2500_3000_nn[idx_ant], 'bo', label='nn', marker=marker[i%N_nodes])
#         plt.plot(ants[idx_ant], ds_peak_2500_3000_ee[idx_ant], 'ro', label='ee', marker=marker[i%N_nodes])
#         if(i == 0):
#             plt.legend()
#     for i, ant in enumerate(ants):
#         plt.annotate('{}'.format(ant), xy=(ants[i], ds_peak_2500_3000_nn[i]))
#     plt.axhline(y=1, ls='--', color='k')
#     plt.xlabel('Antenna number')
#     plt.ylabel('$\sigma_{2500-3000}/\sigma_{3000-3200}$')
#     plt.title('Standard deviation ratio between 2500-3000 ns and 3000-3200 ns')
#     plt.subplot(3,1,3)
#     for i, node in enumerate(node_number):
#         idx_ant = np.where(nodes == node)[0]
#         plt.plot(ants[idx_ant], 10*np.log10(ds_ratio_3500_4000_nn[idx_ant]), 'bo', label='nn', marker=marker[i%N_nodes])
#         plt.plot(ants[idx_ant], 10*np.log10(ds_ratio_3500_4000_ee[idx_ant]), 'ro', label='ee', marker=marker[i%N_nodes])
#         if(i == 0):
#             plt.legend()
#     for i, ant in enumerate(ants):
#         plt.annotate('{}'.format(ant), xy=(ants[i], 10*np.log10(ds_ratio_3500_4000_nn[i])))
#     plt.axhline(y=0, ls='--', color='k')
#     plt.xlabel('Antenna number')
#     plt.ylabel('Deviation from the noise level in dB')
#     plt.title('Deviation of averaged delay spectrum at 3500-4000 ns from the noise level')
#     plt.subplots_adjust(hspace=0.4)
    
# def plot_ds_nodes(uvd_diff, _data_sq_cleaned):
#     freqs = uvd_diff.freq_array[0]
#     taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
    
#     ants = np.sort(uvd_diff.get_ants())
#     pols = ['nn','ee']
#     colors = ['b','r']
#     nodes, antDict, inclNodes = generate_nodeDict(uvd_diff)
    
#     nodes = []
#     ds_ave_nn, ns_ave_nn = [], []
#     ds_ave_ee, ns_ave_ee = [], []
#     for ant in ants:
#         nodes.append(int(antDict[ant]['node']))
#         for pol in pols:
#             key = (ant,ant,pol)
#             ds_ave = np.sqrt(np.abs(np.nanmean(_data_sq_cleaned[key], axis=0)))
#             _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)), axes=1)
#             ns_ave = np.abs(np.nanmean(_diff, axis=0))
#             norm = np.max(ds_ave)
#             if(pol == 'nn'):
#                 ds_ave_nn.append(ds_ave/norm)
#                 ns_ave_nn.append(ns_ave/norm)
#             else:
#                 ds_ave_ee.append(ds_ave/norm)
#                 ns_ave_ee.append(ns_ave/norm)
                
#     nodes = np.array(nodes)
#     ds_ave_nn = np.array(ds_ave_nn).reshape(ants.size,taus.size)
#     ns_ave_nn = np.array(ns_ave_nn).reshape(ants.size,taus.size)
#     ds_ave_ee = np.array(ds_ave_ee).reshape(ants.size,taus.size)
#     ns_ave_ee = np.array(ns_ave_ee).reshape(ants.size,taus.size)
#     node_number = np.unique(nodes)
#     for i, node in enumerate(node_number):
#         if(i % 2 == 0):
#             plt.figure(figsize=(16,4))
#         for j, pol in enumerate(pols):
#             plt.subplot(1,4,2*(i%2)+j+1)
#             idx_ant = np.where(nodes == node)[0]
#             if(pol == 'nn'):
#                 for idx in idx_ant:
#                     plt.plot(taus*1e9, 10*np.log10(ds_ave_nn[idx]), label='({},{})'.format(ants[idx],ants[idx]), linewidth=0.8)
#             else:                
#                 for idx in idx_ant:
#                     plt.plot(taus*1e9, 10*np.log10(ds_ave_ee[idx]), label='({},{})'.format(ants[idx],ants[idx]), linewidth=0.8)
#             plt.xlim(0,4500)
#             plt.ylim(-55,0)
#             plt.title('node {}, {}'.format(node, pol))
#             plt.xlabel(r'$\tau$ (ns)')
#             if(i % 2 == 0 and j == 0):
#                 plt.ylabel(r'$|\tilde{V}(\tau)|$ in dB')
#             plt.grid(axis='y')        
#             plt.legend(loc='upper left', ncol=2)
        
# def plot_wfds_cr(uvd, _data_sq, pol):
#     bls = uvd.get_antpairs()
#     freqs = uvd.freq_array[0]
#     times = uvd.time_array
#     lsts = uvd.lst_array
#     taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))
#     idx_mid = np.where(taus == 0)[0]
    
#     Nants = len(uvd.get_antpairs())
#     Nside = int(np.ceil(np.sqrt(Nants)))
#     Yside = int(np.ceil(float(Nants)/Nside))
    
#     t_index = 0
#     jd = times[t_index]
#     utc = Time(jd, format='jd').datetime
    
    
#     fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2,Nside*2), dpi=75)
#     if pol == 'ee':
#         fig.suptitle("delay spectrum (cross) waterfalls from {0} -- {1} East Polarization".format(times[0], times[-1]), fontsize=14)
#     else:
#         fig.suptitle("delay spectrum (cross) waterfalls from {0} -- {1} North Polarization".format(times[0], times[-1]), fontsize=14)
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
#     fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

#     k = 0
#     for i in range(Yside):
#         for j in range(Nside):
#             ax = axes[i,j]
#             if k < Nants:
#                 key = (bls[k][0], bls[k][1], pol)
#                 ds = 10.*np.log10(np.sqrt(np.abs(_data_sq[key])/np.abs(_data_sq[key][:,idx_mid])))
#                 im = ax.imshow(ds, aspect='auto', rasterized=True,
#                            interpolation='nearest', vmin = -30, vmax = 0, 
#                            extent=[taus[0]*1e9, taus[-1]*1e9, np.max(lsts), np.min(lsts)])
        
#                 ax.set_title(str(bls[k]), fontsize=10)
#             else:
#                 ax.axis('off')
#             if j != 0:
#                 ax.set_yticklabels([])
#             else:
#                 [t.set_fontsize(12) for t in ax.get_yticklabels()]
#                 ax.set_ylabel('Time(LST)', fontsize=10)
#             if i != Yside-1:
#                 ax.set_xticklabels([])
#             else:
#                 [t.set_fontsize(10) for t in ax.get_xticklabels()]
#                 [t.set_rotation(25) for t in ax.get_xticklabels()]
#                 ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#                 ax.set_xlabel('Delay (ns)', fontsize=10)
#             k += 1
        
#     cbar_ax=fig.add_axes([0.95,0.15,0.02,0.7])        
#     cb = fig.colorbar(im, cax=cbar_ax)
#     cb.set_label('dB')
#     fig.show()
# def plot_metric(metrics, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
#     '''Helper function for quickly plotting an individual antenna metric.'''

#     if ants is None:
#         ants = list(set([key[0] for key in metrics.keys()]))
#     if antpols is None:
#         antpols = list(set([key[1] for key in metrics.keys()]))
#     for antpol in antpols:
#         for i,ant in enumerate(ants):
#             metric = 0
#             if (ant,antpol) in metrics:
#                 metric = metrics[(ant,antpol)]
#             plt.plot(i,metric,'.')
#             plt.annotate(str(ant)+antpol,xy=(i,metrics[(ant,antpol)]))
#         plt.gca().set_prop_cycle(None)
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.xlabel(xlabel)
# def show_metric(ant_metrics, antmetfiles, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
#     print("Ant Metrics for {}".format(antmetfiles[1]))
#     plt.figure()
#     plot_metric(ant_metrics['final_mod_z_scores']['meanVij'],
#             title = 'Mean Vij Modified z-Score')

#     plt.figure()
#     plot_metric(ant_metrics['final_mod_z_scores']['redCorr'],
#             title = 'Redundant Visibility Correlation Modified z-Score')

#     plt.figure()
#     plot_metric(ant_metrics['final_mod_z_scores']['meanVijXPol'], antpols=['n'],
#             title = 'Modified z-score of (Vxy+Vyx)/(Vxx+Vyy)')

#     plt.figure()
#     plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['n'],
#             title = 'Modified z-Score of Power Correlation Ratio Cross/Same')
#     plt.figure()
#     plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['e'],
#             title = 'Modified z-Score of Power Correlation Ratio Cross/Same')

# def all_ant_mets(antmetfiles,HHfiles):
#     file = HHfiles[0]
#     uvd_hh = UVData()
#     uvd_hh.read_uvh5(file)
#     uvdx = uvd_hh.select(polarizations = -5, inplace = False)
#     uvdx.ants = np.unique(np.concatenate([uvdx.ant_1_array, uvdx.ant_2_array]))
#     ants = uvdx.get_ants()
#     times = uvd_hh.time_array
#     Nants = len(ants)
#     jd_start = np.floor(times.min())
#     antfinfiles = []
#     for i,file in enumerate(antmetfiles):
#         if i%50==0:
#             antfinfiles.append(antmetfiles[i])
#     Nfiles = len(antfinfiles)
#     Nfiles2 = len(antmetfiles)
#     xants = np.zeros((Nants*2, Nfiles2))
#     dead_ants = np.zeros((Nants*2, Nfiles2))
#     cross_ants = np.zeros((Nants*2, Nfiles2))
#     badants = []
#     pol2ind = {'n':0, 'e':1}
#     times = []

#     for i,file in enumerate(antfinfiles):
#         time = file[54:60]
#         times.append(time)

#     for i,file in enumerate(antmetfiles):
#         antmets = hera_qm.ant_metrics.load_antenna_metrics(file)
#         for j in antmets['xants']:
#             xants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
#         badants.extend(map(lambda x: x[0], antmets['xants']))
#         for j in antmets['crossed_ants']:
#             cross_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
#         for j in antmets['dead_ants']:
#             dead_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1

#     badants = np.unique(badants)
#     xants[np.where(xants==1)] *= np.nan
#     dead_ants[np.where(dead_ants==0)] *= np.nan
#     cross_ants[np.where(cross_ants==0)] *= np.nan

#     antslabels = []
#     for i in ants:
#         labeln = str(i) + 'n'
#         labele = str(i) + 'e'
#         antslabels.append(labeln)
#         antslabels.append(labele)

#     fig, ax = plt.subplots(1, figsize=(16,20))

#     # plotting
#     ax.matshow(xants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
#            extent=[0, len(times), Nants*2, 0])
#     ax.matshow(dead_ants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
#            extent=[0, len(times), Nants*2, 0])
#     ax.matshow(cross_ants, aspect='auto', cmap='RdBu', vmin=-.3, vmax=1.3,
#            extent=[0, len(times), Nants*2, 0])

#     # axes
#     ax.grid(color='k')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_xticks(np.arange(len(times))+0.5)
#     ax.set_yticks(np.arange(Nants*2)+0.5)
#     ax.tick_params(size=8)

#     if Nfiles > 20:
#         ticklabels = times
#         ax.set_xticklabels(ticklabels)
#     else:
#         ax.set_xticklabels(times)

#     ax.set_yticklabels(antslabels)

#     [t.set_rotation(30) for t in ax.get_xticklabels()]
#     [t.set_size(12) for t in ax.get_xticklabels()]
#     [t.set_rotation(0) for t in ax.get_yticklabels()]
#     [t.set_size(12) for t in ax.get_yticklabels()]

#     ax.set_title("Ant Metrics bad ants over observation", fontsize=14)
#     ax.set_xlabel('decimal of JD = {}'.format(int(jd_start)), fontsize=16)
#     ax.set_ylabel('antenna number and pol', fontsize=16)
#     red_ptch = mpatches.Patch(color='red')
#     grn_ptch = mpatches.Patch(color='green')
#     blu_ptch = mpatches.Patch(color='blue')
#     ax.legend([red_ptch, blu_ptch, grn_ptch], ['dead ant', 'cross ant', 'good ant'], fontsize=14)
