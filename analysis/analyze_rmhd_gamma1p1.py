#!/usr/bin/env python3
"""Audit the Gamma=1.1 nonrotating comparison against its saved control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Installed beside the shared readers; allow validation of the staged copy.
if (Path(__file__).parent/'wind_common.py').is_file():
    sys.path.insert(0,str(Path(__file__).parent))
else:
    sys.path.insert(0,'/Users/calder/code/ns-wind/analysis')
from wind_common import (REPO_DIR, M_SUN, read_definitions, read_parameters,
                         read_grid, read_catalog, read_snapshot)
from rmhd_diagnostics import derive_rmhd_profiles, outward_crossing_radii
from analyze_rmhd_suites import RunSummary, analyze_run, _json_safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=REPO_DIR/'runs/rmhd_gamma1p1_experiments/CS_REL_0_0p136')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    physics,units = read_definitions(run_dir/'definitions.h')
    assert physics == 'RMHD'
    pars = read_parameters(run_dir/'pluto.ini')
    radius_code = read_grid(run_dir/'grid.out')
    radius = radius_code*units['UNIT_LENGTH']/1e5
    active = radius > 12
    first = int(np.flatnonzero(active)[0])
    catalog = read_catalog(run_dir/'dbl.out')
    if not catalog:
        raise RuntimeError('No saved snapshots')
    expected_path = run_dir/'expected_steady_wind.json'
    expected = json.loads(expected_path.read_text()) if expected_path.is_file() else None
    histories = []
    final = None
    for number in sorted(catalog):
        snapshot = read_snapshot(run_dir,number,radius_code,catalog)
        prof = derive_rmhd_profiles(snapshot,radius_code,units,pars)
        rho = np.asarray(snapshot['rho'])*units['UNIT_DENSITY']
        pressure = np.asarray(snapshot['prs'])*units['UNIT_DENSITY']*units['UNIT_VELOCITY']**2
        theta = pressure/(rho*units['UNIT_VELOCITY']**2)
        roots = outward_crossing_radii(radius,prof,active)['sonic']
        sampled = {'base':float(prof['mdot_g_s'][first]/M_SUN)}
        for rr in [20,100,1000]:
            sampled[f'{rr}_km'] = float(np.interp(rr,radius[active],prof['mdot_g_s'][active])/M_SUN)
        sampled['outer'] = float(prof['mdot_g_s'][-1]/M_SUN)
        histories.append({'snapshot':number,'time_s':float(snapshot['time'])*units['UNIT_LENGTH']/units['UNIT_VELOCITY'],
                          'mdot_msun_s':sampled,'sonic_radius_km':roots[0] if roots else None,
                          'base_velocity_c':float(prof['velocity_c'][first]),
                          'inflow_cells':int(np.count_nonzero(prof['velocity_c'][active]<0))})
        final = (prof,rho,theta)
    complete = (run_dir/'run_complete.json').is_file()
    audit = None
    if complete:
        summary = analyze_run(RunSummary(suite='gamma1p1',name=run_dir.name,run_dir=run_dir,
                                        state='complete',description='',parameters={}),0.20)
        audit = summary.metrics
    report = {'run_directory':str(run_dir),'parameters':pars,'snapshot_count':len(histories),
              'run_complete':complete,'steadiness':audit,'expected_smooth_steady_wind':expected,
              'final':histories[-1],'histories':histories,
              'interpretation':'A completed stop time and a fitted steady ODE do not establish simulation steadiness. This is a phenomenological gamma-law wind, without neutrino heating or a physical PNS EOS.'}
    destination = run_dir/'diagnostics'
    destination.mkdir(exist_ok=True)
    (destination/'gamma_comparison.json').write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.15})
    fig,axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    times = np.asarray([h['time_s'] for h in histories])
    for label in ['base','20_km','100_km','1000_km','outer']:
        axes[0,0].plot(times,[h['mdot_msun_s'][label] for h in histories],label=label.replace('_',' '))
    axes[0,0].set(xlabel='Time [s]',ylabel=r'Mass flux [$M_\odot$ s$^{-1}$]')
    axes[0,0].set_yscale('symlog',linthresh=1e-4)
    axes[0,0].legend(fontsize=8)
    prof,rho,theta = final
    for ax,y in zip([axes[0,1],axes[1,0],axes[1,1]],
                    [prof['velocity_c'],rho,theta]):
        ax.plot(radius[active],np.asarray(y)[active],label='New comparison: final',color='#0072b2')
        ax.set_xscale('log')
        ax.set_xlabel('Radius [km]')
    axes[0,1].plot(radius[active],prof['sound_speed_c'][active],ls=':',color='#0072b2',label='New sound speed')
    reference = REPO_DIR/'runs/rmhd_experiments/resolution_2x'
    if (reference/'run_complete.json').is_file():
        rp,ru = read_definitions(reference/'definitions.h')
        rc = read_grid(reference/'grid.out')
        rr = rc*ru['UNIT_LENGTH']/1e5
        cp = read_catalog(reference/'dbl.out')
        ss = read_snapshot(reference,max(cp),rc,cp)
        pp = derive_rmhd_profiles(ss,rc,ru,read_parameters(reference/'pluto.ini'))
        live = rr>12
        for ax,y in zip([axes[0,1],axes[1,0],axes[1,1]],
                        [pp['velocity_c'],ss['rho']*ru['UNIT_DENSITY'],ss['prs']/ss['rho']]):
            ax.plot(rr[live],np.asarray(y)[live],color='#777',ls='--',label='Saved Γ=4/3 control')
    model_path = run_dir/'expected_steady_profiles.npz'
    if model_path.is_file():
        with np.load(model_path) as model:
            for ax,key in zip([axes[0,1],axes[1,0],axes[1,1]],['velocity_c','rho_g_cm3','theta']):
                ax.plot(model['radius_km'],model[key],color='#00887a',ls='-.',label='Independent steady prediction')
    if expected:
        axes[0,0].axhline(expected['mdot_msun_s'],color='#00887a',ls='-.',lw=1)
        axes[0,1].axvline(expected['sonic_radius_km'],color='#00887a',ls=':',lw=1)
    axes[0,1].set_ylabel(r'Radial speed / $c$')
    axes[0,1].set_yscale('symlog',linthresh=1e-5)
    axes[1,0].set(ylabel=r'Density [g cm$^{-3}$]',yscale='log')
    axes[1,1].set(ylabel=r'$P/(\rho c^2)$',yscale='log')
    for ax in axes.flat:
        if ax is not axes[0,0]:
            ax.legend(fontsize=8)
    fig.suptitle(f'Γ=1.1, CS_REL_0=0.136 — final t={times[-1]:.4f} s; '
                 f'{"complete" if complete else "partial"}, '
                 f'{audit["steady_label"] if audit else "steadiness not established"}')
    fig.savefig(destination/'gamma_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(_json_safe({'report':str(destination/'gamma_comparison.json'),
                      'plot':str(destination/'gamma_comparison.png'),
                      'final':histories[-1],'steadiness':audit}),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
