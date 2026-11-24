"""Pre-origin EDA and leakage/row-alignment checks for practice 5."""
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.home()/".codex"/"tmp"/"practice5_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_experiment import OUT, ROOT, read_day, predict_baselines
from run_tree import enrich, features, sample_day


def main():
    days={d:enrich(read_day(d)) for d in range(1,15)}
    checks=[]
    for d in range(5,11):
        x=days[d]
        for c in (10*36000+30*600,11*36000,11*36000+30*600):
            full=features(x,c)
            if full is None:continue
            left=np.searchsorted(x["obs_tick"],c)
            prefix=dict(x)
            for field in ("obs_tick","cum_good","vol_good","bid_good","ask_good"):
                prefix[field]=x[field][:left]
            prefix["grid"]=x["grid"][:c-x["start"]]
            clipped=features(prefix,c)
            assert clipped is not None
            assert np.array_equal(full[1],clipped[1]) and full[0]==clipped[0]
            checks.append({"day":d,"origin_tick":c,"check":"feature_prefix_equal","passed":True})
        samples=sample_day(x,d,stop=11*36000)
        assert all(t<11*36000 for _,_,t in samples[3])
        checks.append({"day":d,"origin_tick":11*36000,"check":"B_labels_before_refresh","passed":True})
        # Every sampled 7.5-second bin retains the raw target-row weight.
        all_samples=sample_day(x,d)
        grouped={}
        for weight,(_,origin,label_tick) in zip(all_samples[2],all_samples[3]):
            assert origin<=label_tick<origin+1800
            key=(origin,(label_tick-origin)//75)
            grouped[key]=grouped.get(key,0.)+weight
        for (origin,b),total_weight in grouped.items():
            raw=(np.searchsorted(x["ticks"],origin+(b+1)*75)-
                 np.searchsorted(x["ticks"],origin+b*75))
            assert abs(total_weight-raw)<1e-8
        checks.append({"day":d,"origin_tick":-1,"check":"sample_weight_equals_raw_rows","passed":True})
    target=days[14]
    c=11*36000
    mask=(target["tick"]>=c)&(target["tick"]<c+1800)&np.isnan(target["price"])
    export=pd.read_csv(ROOT/"prediction.csv",dtype={"time":str})
    assert len(export)==int(mask.sum())==1799
    assert export.time.tolist()==target["frame"].loc[mask,"time"].tolist()
    assert export.groupby("time").predicted_price.nunique().max()==1
    assert np.isfinite(export.predicted_price).all() and (export.predicted_price>0).all()
    checks.append({"day":14,"origin_tick":c,"check":"prediction_csv_row_alignment","passed":True})
    pd.DataFrame(checks).to_csv(OUT/"leakage_and_output_checks.csv",index=False)
    coverage=[]
    for d in range(1,14):
        x=days[d]
        candidates=list(range(9*36000+10*600,15*36000+25*600+1,1800))
        covered=[c for c in candidates if c+1799<=x["end"]]
        valid=[c for c in covered if features(x,c) is not None]
        coverage.append({"day":d,"candidate_origins":len(candidates),
                         "full_horizon_available":len(covered),"feature_quality_pass":len(valid),
                         "excluded_no_full_horizon":len(candidates)-len(covered),
                         "excluded_feature_quality":len(covered)-len(valid)})
    pd.DataFrame(coverage).to_csv(OUT/"training_origin_coverage.csv",index=False)

    # Development-day price curves, each rebased to its first observed price.
    fig,ax=plt.subplots(figsize=(11,5.4))
    agg=[]
    for d in range(1,11):
        x=days[d]
        t=x["obs_tick"]
        p=x["prices"]
        unique,first,counts=np.unique(t,return_index=True,return_counts=True)
        last=first+counts-1
        t=t[last];p=p[last]
        five=(t//3000)*3000
        z=pd.DataFrame({"bin":five,"price":p,"cum":x["cum_good"][last]})
        z=z.groupby("bin",sort=True).agg(price=("price","last"),cum=("cum","last"))
        z["day"]=d
        z["change_pct"]=(z.price/z.price.iloc[0]-1)*100
        z["volume_increment"]=z.cum.diff()
        z["five_min_price_change_abs"]=z.price.diff().abs()
        agg.append(z.reset_index())
        ax.plot(z.index.to_numpy()/36000,z.change_pct.to_numpy(),lw=1,alpha=.75,label=f"day{d}")
    ax.axvline(11,color="black",ls="--",lw=1)
    ax.set(xlabel="Time of day (hour)",ylabel="Price change from first 5-min bin (%)",
           title="Development days: observed price path")
    ax.legend(ncol=5,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"price_paths.png",dpi=160);plt.close(fig)
    a=pd.concat(agg,ignore_index=True)
    a.to_csv(OUT/"five_min_development_data.csv",index=False)
    agg2=a.groupby("bin").agg(n_days=("day","nunique"),
        median_volume=("volume_increment","median"),q25_volume=("volume_increment",lambda z:z.quantile(.25)),
        q75_volume=("volume_increment",lambda z:z.quantile(.75)),
        median_abs_price_move=("five_min_price_change_abs","median")).reset_index()
    agg2.to_csv(OUT/"intraday_pattern_summary.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    hour=agg2.bin/36000
    axes[0].plot(hour,agg2.median_volume,color="#1876a3")
    axes[0].fill_between(hour,agg2.q25_volume,agg2.q75_volume,color="#1876a3",alpha=.2)
    axes[0].set_ylabel("5-min cumulative volume increment")
    axes[1].plot(hour,agg2.median_abs_price_move,color="#ba5c18")
    axes[1].set_ylabel("Median absolute 5-min price move")
    axes[1].set_xlabel("Time of day (hour)")
    for ax in axes:ax.axvline(11,color="black",ls="--",lw=1)
    fig.tight_layout();fig.savefig(OUT/"intraday_patterns.png",dpi=160);plt.close(fig)

    x=days[14];obs=x["obs_tick"];p=x["prices"]
    select=(obs>=10*36000+50*600)&(obs<c)
    loc=np.flatnonzero(select)
    stride=max(1,len(loc)//1400)
    loc=loc[::stride]
    bid=x["bid_good"][loc];ask=x["ask_good"][loc]
    imb=np.divide(bid-ask,bid+ask,out=np.zeros_like(bid),where=bid+ask>0)
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(obs[loc]/36000,p[loc],color="#176f94",lw=1)
    ax2=ax.twinx();ax2.plot(obs[loc]/36000,imb,color="#bd621d",lw=.7,alpha=.65)
    ax.set(xlabel="Time of day (hour)",ylabel="Observed price",title="Day14: available 10:50-11:00 data")
    ax2.set_ylabel("Total bid/ask imbalance")
    fig.tight_layout();fig.savefig(OUT/"day14_preorigin.png",dpi=160);plt.close(fig)
    (OUT/"diagnostic_info.json").write_text(json.dumps({"checks_passed":len(checks),
        "target_rows":int(mask.sum()),"target_unique_times":int(export.time.nunique()),
        "duplicate_target_rows":int(len(export)-export.time.nunique())},indent=2),encoding="utf-8")
    print("diagnostics passed",len(checks),flush=True)


if __name__=="__main__":main()
