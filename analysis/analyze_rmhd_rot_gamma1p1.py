"""Measure sigma and fluxes from evolved rotating snapshots, never launch a run."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

REPO_DIR=Path(__file__).resolve().parents[1]
from wind_common import (M_SUN,read_catalog,read_definitions,read_grid,
                         read_parameters,read_snapshot)
from rmhd_diagnostics import derive_rmhd_profiles,relative_span


def finite_json(value):
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list): return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def analyze_case(case, late_fraction=0.2):
    if not (case/"dbl.out").exists():
        return {"state":"prepared; no evolved output","has_evolved_rotating_output":False}
    catalog=read_catalog(case/"dbl.out")
    if not catalog: return {"state":"no snapshots","has_evolved_rotating_output":False}
    _,units=read_definitions(case/"definitions.h")
    params=read_parameters(case/"pluto.ini")
    rcode=read_grid(case/"grid.out")
    latest=max(catalog)
    tmin=min(float(item["time"]) for item in catalog.values())
    tmax=max(float(item["time"]) for item in catalog.values())
    late=[n for n,item in sorted(catalog.items()) if float(item["time"])>=tmax-late_fraction*(tmax-tmin)]
    histories={str(r):{"sigma":[],"lorentz_gamma":[],"radial_speed_c":[],"mdot_msun_s":[],
                         "poynting_to_kinetic":[],"poynting_to_kinetic_plus_enthalpy":[]}
               for r in [100,1000,9000]}
    last=None
    peaks=[]
    for number in late:
        data=read_snapshot(case,number,rcode,catalog)
        last=derive_rmhd_profiles(data,rcode,units,params)
        radius=last["radius_km"]
        active=(radius>12)&(radius<=9000)
        peaks.append(float(np.max(last["magnetization"][active])))
        for r,history in histories.items():
            interp=lambda key:float(np.interp(float(r),radius,last[key]))
            history["sigma"].append(interp("magnetization"))
            history["lorentz_gamma"].append(interp("lorentz_gamma"))
            history["radial_speed_c"].append(interp("velocity_c"))
            history["mdot_msun_s"].append(interp("mdot_g_s")/M_SUN)
            kinetic=interp("kinetic_power_erg_s")
            enthalpy=interp("thermal_enthalpy_power_erg_s")
            em=interp("poynting_power_erg_s")
            history["poynting_to_kinetic"].append(em/kinetic if kinetic!=0 else np.nan)
            history["poynting_to_kinetic_plus_enthalpy"].append(em/(kinetic+enthalpy) if kinetic+enthalpy!=0 else np.nan)
    statistics={r:{key:{"median":float(np.median(values)),"relative_span":relative_span(np.asarray(values))}
                   for key,values in history.items()} for r,history in histories.items()}
    radius=last["radius_km"]
    bulk=(radius>=20)&(radius<=9000)
    radial_span=relative_span(last["mdot_g_s"][bulk])
    outflow=bool(np.all(last["velocity_c"][bulk]>0))
    enough=len(late)>=12
    steady=enough and outflow and radial_span<=.10 and statistics["9000"]["mdot_msun_s"]["relative_span"]<=.05 and statistics["9000"]["radial_speed_c"]["relative_span"]<=.05
    # Actual numerical boundary mass flux, including Rusanov diffusion.
    faces=[]
    for path in case.glob("face_flux.*.csv"):
        with path.open() as fp:
            for row in csv.DictReader(fp):
                if float(row["time_code"])>=tmax-late_fraction*(tmax-tmin) and 12-1e-8<=float(row["face_radius_km"])<=12.041:
                    faces.append(row)
    boundary=None
    if faces:
        closest=min(float(row["face_radius_km"]) for row in faces)
        face=[row for row in faces if abs(float(row["face_radius_km"])-closest)<1e-8]
        numerical=np.asarray([float(row["numerical_mdot_msun_s"]) for row in face])
        diffusive=np.asarray([float(row["diffusive_mdot_msun_s"]) for row in face])
        boundary={"face_radius_km":closest,"numerical_mdot_msun_s_median":float(np.median(numerical)),
                  "diffusive_fraction_absolute_median":float(np.median(np.abs(diffusive)/np.maximum(np.abs(numerical),1e-300)))}
    intervals=[]
    selected=(radius>12)&(radius<=9000)&(last["velocity_c"]>0)&(last["magnetization"]>=1)
    ids=np.flatnonzero(selected)
    if ids.size:
        for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            intervals.append([float(radius[group[0]]),float(radius[group[-1]])])
    return finite_json({"state":"analyzed evolved output" if tmax>tmin else "initial state only",
         "has_evolved_rotating_output":"P_ROT_MS" in params and tmax>tmin,
         "parameters":params,"latest_snapshot":latest,"latest_time_s":tmax*units["UNIT_LENGTH"]/units["UNIT_VELOCITY"],
         "completion_marker_present":(case/"run_complete.json").exists(),"late_snapshot_count":len(late),
         "late_statistics_at_radius_km":statistics,"late_peak_sigma_median":float(np.median(peaks)),
         "final_sigma_at_least_one_intervals_km":intervals,
         "final_radial_mdot_relative_span_20_to_9000km":radial_span,"final_bulk_outflow":outflow,
         "adopted_steadiness_pass":steady,"steadiness_thresholds":{"temporal_outer_speed":.05,"temporal_outer_mdot":.05,"radial_mdot":.10},
         "boundary_face_flux":boundary,
         "interpretation":"A peak sigma>=1 alone does not establish an escaping, relativistic or Poynting-dominated wind. Check radius/time persistence, Lorentz factor, power ratios, mass flux and boundary diffusion."})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotating-dir",type=Path,
                        default=REPO_DIR/"runs/rmhd_rot_gamma1p1_experiments")
    parser.add_argument("--output-dir",type=Path,
                        default=REPO_DIR/"runs/rmhd_rot_gamma1p1_analysis")
    parser.add_argument("--late-fraction",type=float,default=0.2)
    args=parser.parse_args()
    if not 0<args.late_fraction<=1:
        parser.error("--late-fraction must be in (0, 1]")
    cases=sorted(path.parent for path in args.rotating_dir.glob("*/experiment.json"))
    if not cases:
        parser.error(f"No prepared cases found in {args.rotating_dir}")
    results={case.name:analyze_case(case,args.late_fraction) for case in cases}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/"analysis_manifest.json"
    report={"suite":"rot_gamma1p1","rotating_directory":str(args.rotating_dir.resolve()),
            "late_fraction":args.late_fraction,"cases":results}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    count=sum(row["has_evolved_rotating_output"] for row in results.values())
    print(f"Wrote {output}; {count} cases have evolved rotating output.")


if __name__=="__main__":
    main()
