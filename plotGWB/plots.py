#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:00:42 2020

@author: elinore
"""

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from cycler import cycler


def color_marker_cycle(colors, markers):
    """
    Set matplotlib marker and color cycle flexibly
    """
    if colors is None:
        current_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if isinstance(markers, str):
            cycle = cycler(marker=markers) * cycler(color=current_colors)
        else:
            cycle = cycler(marker=markers, color=current_colors[:len(markers)])
    elif isinstance(colors, str) or isinstance(markers, str):
        cycle = cycler(color=colors) * cycler(marker=markers)
    else:
        cycle = cycler(color=colors, marker=markers)

    return cycle


# FIXME: check that this converts multiple maps correctly
# FIXME: should deal with single array_like map correctly as well
def healpix_to_mesh(hp_maps, gridsize=1000):
    """
    Transform healpix pixels to matplotlib mesh for prettier plotting
    (assumes Mollweide).

    Parameters
    ----------
    hp_maps: Dataframe
        Each column represents a single skymap

    gridsize: int
        Number of pixels on the resulting matplotlib grid. Since the pixel
        amplitudes will be interpolated, this should be chosen so that the
        matplotlib pixels are smaller than the healpix pixels

    Returns
    -------
    (lons, lats, maps):
        Pixel locations and values
    """
    # number of pixels
    xsize = gridsize
    ysize = gridsize//2

    # equal area pixels
    theta = np.arccos(np.linspace(-1, 1, ysize))
    phi = np.linspace(-np.pi, np.pi, xsize)
    PHI, THETA = np.meshgrid(phi, theta)

    # interpolate onto the new pixels
    grid_maps = hp_maps.apply(hp.get_interp_val, args=(THETA, PHI))
    #grid_map = hp.get_interp_val(signal, THETA, PHI)

    # x points (reversed for astro convention), y points (latitude), maps
    return phi[::-1], np.pi/2 - theta, grid_maps


# need a routine for plotting a series of skymaps & saving as pdf


def plot_map(lon, lat, grid_map, fig=None, sub=111, vmax=None, vmin=None,
             cmap='RdBu_r', time=None, time_unit='years'):
    """
    Plot a single skymap

    Parameters
    ----------
    lon: array_like
        Pixel longitudes (from healpix_to_mesh)

    lat: array_like
        Pixel latitudes (from healpix_to_mesh)

    grid_map: array_like
        Skymap values at each pixel

    fig: matplotlib figure or None
        Optionally, add plot to this figure

    sub: matplotlib subplot or gridspec set
        Make plot in this subplot of the figure

    vmax: float or None
        Desired maximum value of the colormap; if vmax or vmin are not both
        specified, the missing values will be chosen symmetrically.

    vmin: float or None
        Desired minimum value of the colormap; if vmax or vmin are not both
        specified, the missing values will be chosen symmetrically.

    cmap: matplotlib colormap
        Colormap for the pixel values; a diverging colormap and symmetric
        vmin/vmax will give the best result.

    time: float or None
        If given, will be printed as a title for the plot

    time_unit: string
        Units for the time, shown in title

    Returns
    -------
    ax_map: matplotlib axis
        axis of the plot
    """

    # if vmax and/or vmin is not specified, set symmetrically
    if vmax is None and vmin is None:
        # centered colormap with limits of the abs max of the data
        vmax = np.max(np.abs(np.ravel(grid_map)))
        vmin = -vmax
    elif vmax is None:
        vmin = -vmax
    elif vmin is None:
        vmax = -vmin

    # FIXME: should allow specifying an ax instead?
    # FIXME: add extra kwargs for figure?
    if fig is None:
        fig = plt.figure()

    # nb sub can be gridspec set
    ax_map = fig.add_subplot(sub, projection='mollweide')
    ax_map.pcolormesh(lon, lat, grid_map, rasterized=True, cmap=cmap,
                      vmin=vmin, vmax=vmax)

    if time is not None:
        ax_map.set_title('t = {} {}'.format(time, time_unit))

    ax_map.tick_params(labelbottom=False, labelleft=False)

    return ax_map


def plot_psr_trace(res, psr_index=None, time=None, ax=None, fig=None, sub=111,
                   psr_colors=None, psr_markers='o',
                   marker_kwargs={}, line_kwargs={}):
    """
    Plot a trace of pulsar timing residuals, with solid lines up until the time
    specified (if any) and dashed after

    Parameters
    ----------
    res: dataframe
        PTA residuals or set of healpix sky maps. Each column should
        represent a time and each row should represent a single pulsar

    psr_index: array_like or None
        index of the residuals which should be plotted.

    time: float or None
        Highlight traces up until this time, and show as dashes thereafter.
        Doesn't need to be exact, but should be close to one of the res column
        names. Defaults to the end.

    psr_colors: Single color or array of colors
        Colors of the highlighted residual traces. Default is to follow the
        current color cycle.

    psr_markers: Single marker or array of markers
        Hollow markers will be plotted at the end of each highlighted trace

    marker_kwargs: dict
        Additional desired kwargs for the pulsar markers

    line_kwargs: dict
        Additional desired kwargs for the pulsar traces
    """
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(sub)

    times = res.columns.values

    if time is None:
        # defaults to the very end
        time = times[-1]
    else:
        # find the closest value to the specified time
        closest_idx = np.argmin(np.abs(times - time))
        time = times[closest_idx]

    # remove the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    # line and marker properties
    past_res_kws = dict(legend=False, color=psr_colors)
    past_res_kws.update(line_kwargs)
    future_res_kws = line_kwargs.copy()
    future_res_kws.update(legend=False, linestyle='dotted', color='grey')
    psr_kws = dict(linestyle='None')
    psr_kws.update(marker_kwargs)

    # select the traces to plot
    if psrs is None:
        psr_traces = res.T
    else:
        # FIXME: easier to pass df of pulsars?
        # psr_idx = psrs.index
        psr_traces = res.loc[psr_idx, :].T

    # plot TOA traces
    psr_traces.plot(ax=ax, **future_res_kws)
    psr_traces.loc[:time].plot(ax=ax, **past_res_kws)

    # plot current values with markers
    cycle = color_marker_cycle(psr_colors, psr_markers)
    ax.set_prop_cycle(cycle)
    for p in psr_traces.loc[time]:
        ax.plot(time, p, **psr_kws)

    # plot a line to show zero
    ax.axhline(0, color='grey', linewidth=0.5, zorder=0)

    return


## making map + pulsar plots
# with PdfPages('/Users/elinore/plots/mc_gw/anisotropy/residuals_10yr_time_vs_freq/time2_psrs.pdf') as pdf:
#    for time in rt:
#        plt.figure(0)
#        s = 30
#        hp.mollview(rt[time], cmap='RdBu_r', min=-9e-8, max=9e-8, fig=0, sub=(211), title='%.2f years' %(time/3.15e7))
#        for p in psrs:
#            hp.projscatter(hp.pix2ang(32, p), marker=psr_markers[p], c=psr_colors[p], edgecolor='k', s=s)
#        plt.subplot(212)
#        ax = plt.gca()
#        ax.set_axis_off()
#        plt.hlines(0, 0, 3.15e8, color='lightgrey', lw=0.5)
#        rt.loc[psrs, :time].T.plot(ax=ax, legend=False, ylim=(-8e-8, 1e-7), xlim=(-100, 3.15e8),
#                                   color='#6a3d9a #cab2d6 #33a02c #b2df8a'.split())
#        for p in psrs:
#            plt.scatter(time, rt.loc[p, time], marker=psr_markers[p], color=psr_colors[p], edgecolor='k', s=s, zorder=10)
#        pdf.savefig(bbox_inches='tight')
#        plt.close()



# add another routine for skymaps + pulsar traces
# input is a dataframe of maps (index) at different times (columns)
