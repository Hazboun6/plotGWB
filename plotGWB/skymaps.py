# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:19:01 2016

@author: elinore

Routines for calculating timing residuals and redshifts of a GWB, along with
other functions useful for producing and analyzing skymaps.
"""

#FIXME: standardize notation for m, n, etc (which paper did I use?)

import numpy as np
import healpy as hp
import pandas as pd


# FIXME: add the time/freq array generator routines for convenience


def map_pixels(nside):
    """
    Returns a dataframe of the angle for each pixel in a given healpix
    pixelization. Columns: theta, phi.  Index gives the pixel number
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pixels = pd.DataFrame({'phi': phi, 'theta': theta})

    return pixels


def source_vectors(theta, phi):
    """
    Produces vectors along the polarization axes and the direction of GW
    propagation for sources at (theta, phi).
    """
    # theta, phi is the direction to the source
    # everything below needs the direction of GW propagation
    theta_prop = np.pi - theta
    phi_prop = np.pi + phi

    # observer's frame tangential components
    l = pd.DataFrame({('l', 'x'): np.cos(theta_prop)*np.cos(phi_prop),
                      ('l', 'y'): np.cos(theta_prop)*np.sin(phi_prop),
                      ('l', 'z'): -np.sin(theta_prop)})
    m = pd.DataFrame({('m', 'x'): -np.sin(phi_prop),
                      ('m', 'y'): np.cos(phi_prop),
                      ('m', 'z'): 0.})

    k = pd.DataFrame({('k', 'x'): np.sin(theta_prop)*np.cos(phi_prop),
                      ('k', 'y'): np.sin(theta_prop)*np.sin(phi_prop),
                      ('k', 'z'): np.cos(theta_prop)})

    src_vecs = k.join([l, m])
    src_vecs.index.name = 'src'

    return src_vecs


def pulsar_vectors(theta, phi):
    """
    Produces a pandas dataframe of Cartesian vectors in the direction of each
    pulsar at (theta, phi).
    """
    psr_vecs = pd.DataFrame({'x': np.sin(theta)*np.cos(phi),
                             'y': np.sin(theta)*np.sin(phi),
                             'z': np.cos(theta)})
    psr_vecs.index.name = 'psr'
    return psr_vecs


def antenna_patterns(theta_psr, phi_psr, theta_src, phi_src):
    """
    Plus and cross polarization patterns for all pulsar locations at
    (theta_psr, phi_psr) given a set of sources in the directions
    (theta_src, phi_src).

    This routine does the heavy lifting of all the dot products between
    sources and pulsar pixels.
    """

    p = pulsar_vectors(theta_psr, phi_psr)  # directions to pulsars
    s = source_vectors(theta_src, phi_src)  # source direction, pol basis

    # in order to create all dot product combinations, need to align p, s
    # memory pressure is an issue here since need ~num(psr)*num(src) rows
    # the following increases the size of p, but NOT s (hopefully)
    ps_index = pd.MultiIndex.from_product((p.index, s.index),
                                           names=('psr', 'src'))
    p = p.reindex(index=ps_index, level='psr')
    dot = pd.DataFrame({'pk': p.mul(s['k']).sum(axis=1),
                        'pl': p.mul(s['l']).sum(axis=1),
                        'pm': p.mul(s['m']).sum(axis=1)})


    # make df of antenna pattern values for each pulsar & source combination
    # sum over sources *after* multiplying by source amplitudes
    # note that convention for plus, cross is wrt source orientation, not observer
    # effect of psi is in the definition of +,x amplitudes
    antenna = pd.DataFrame({'plus': 0.5*(dot.pl**2 - dot.pm**2)/(1 + dot.pk),
                            'cross': dot.pl*dot.pm/(1 + dot.pk)})
    antenna.columns.set_names('pol', inplace=True)

    return antenna


def hplus_hcross(src, domain='freq', timeseries=np.zeros(1)):
    """
    Get the components of hplus, hcross for each source given its parameters.
    If domain is 'freq' (or begins with 'f'), calculate the amplitudes of each
    polarization in the frequency domain (for real frequencies only).
    If domain is 'time' (or begins with 't'), calculate for the timeseries.
    """
    a, b = inclination(src['iota'])
    cos2psi = np.cos(2*src['psi'])
    sin2psi = np.sin(2*src['psi'])
    sinPhi0 = np.sin(src['Phi0'])
    cosPhi0 = np.cos(src['Phi0'])

    if domain[0] is 'f':
        plus = 0.5*src['A']*(a*cos2psi*(cosPhi0 + 1j*sinPhi0) +
                             b*sin2psi*(sinPhi0 - 1j*cosPhi0))
        cross = 0.5*src['A']*(b*cos2psi*(sinPhi0 - 1j*cosPhi0) -
                              a*sin2psi*(cosPhi0 + 1j*sinPhi0))
        plus.name = 'plus'
        cross.name = 'cross'
    #elif domain[0] is 't':
    # do time calculations

    h = pd.concat([cross, plus], axis=1)
    h.index.name = 'pol'
    return h


def redshift(theta_psr, phi_psr, A_src, theta_src, phi_src,
             iota_src=0.0, Phi_src=0.0, Phi0_src=0.0, psi_src=0.0):
    """
    Produce single-epoch list of redshifts at pulsars at (theta_psr, phi_psr)
    for GW sources with properties: A (amplitude), iota (inclination),
    Phi (phase), Phi0 (initial phase), psi (polarization angle).
    """

    a =  1 + np.cos(iota_src)**2
    b = - 2*np.cos(iota_src)

    h = pd.DataFrame({'plus': a*A_src*np.cos(Phi_src + Phi0_src),
                      'cross': b*A_src*np.sin(Phi_src + Phi0_src)})

    F = antenna_patterns(theta_psr, phi_psr, theta_src, phi_src)

    # assume pulsar term is 0
    # multiply h and F for each source then sum over sources and polarizations
    z = -h.mul(F, level='src').sum(level='psr').sum(axis=1)

    return z


# this is the key function
# should probably have more loops and less giant arrays
def residuals_time(src, psr, times=np.zeros(1), zero_r0=False):
    """
    Calculate the timing residuals for each pulsar at each observation time,
    given a list of monochromatic sources. If zero_r0, the timing residuals
    will be calculated relative to their initial values.
    """

    # get signal from each src in both polarizations at all times
    tt, ff = np.meshgrid(times, src['zf'])
    tt, phiphi = np.meshgrid(times, src['Phi0'])
    wave_index = pd.MultiIndex.from_product([['cross', 'plus'], times],
                                            names=['pol', 'time'])
    if zero_r0:
        # will want to subtract the timing residual for the initial phase
        sin_IC = -np.sin(phiphi)
        cos_IC = -np.cos(phiphi)
    else:
        # don't subtract off initial value
        sin_IC = 0.
        cos_IC = 0.

    # amplitude from sine and cosine terms (function of time)
    #FIXME: need to rotate h+, hx:
    #h+' = h+*cos(2psi) + hx*sin(2psi)
    #hx' = -h+*sin(2psi) + hx*cos(2psi)
    wave_values = np.hstack((np.sin(2*np.pi*ff*tt + phiphi) + sin_IC,
                             np.cos(2*np.pi*ff*tt + phiphi) + cos_IC))
    waves = pd.DataFrame(data=wave_values, columns=wave_index,
                         index=src.index)
    # amplitudes based on source properties (not time dependent)
    a, b = inclination(src['iota'])
    amps = pd.DataFrame({'cross': b*src['A']/(2*np.pi*src['zf']),
                         'plus': a*src['A']/(2*np.pi*src['zf'])})

    # total strain amplitude in each polarization at each time for each source
    waves = amps.mul(waves, level='pol')
    del amps

    # antenna patterns for each pulsar-source combination
    F = antenna_patterns(psr['theta'], psr['phi'], src['theta'],
                         src['phi'])

    # expand waves so that each pulsar gets a copy of all sources
    waves = waves.reindex(index=F.index, level='src')

    # timing residuals (in s) for each psr: sum of waves*antenna patterns
    # over all polarizations and sources
    residuals = F.mul(waves, level='pol').sum(level='psr').sum(axis=1, level='time')

    return residuals


# FIXME: incorporate into other code
def inclination(iota):
    """
    A vector showing the components in the plus and cross polarization
    produced for a given inclination.
    a(iota), b(iota) in the notation of Sesana & Vecchio 2010.
    """
    a =  1 + np.cos(iota)**2
    b = -2*np.cos(iota)

    return a, b
