"""Reproducible 3-minute, fixed-origin validation on the original tick rows."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "주식데이터"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
START = time.time()
SEED = 42
HORIZON = 1800  # 0.1-second ticks


def tick_index(s: pd.Series) -> np.ndarray:
    x = s.str.split(":", expand=True)
    return (x[0].astype(int) * 36000 + x[1].astype(int) * 600 +
            (x[2].astype(float) * 10).round().astype(int)).to_numpy()


def read_day(d: int):
    frame = pd.read_csv(DATA / f"ticks_day{d}.csv", dtype={"time": str})
    tick = tick_index(frame.time)
    price = frame.price.to_numpy(float)
    good = np.isfinite(price)
    ticks = tick[good]
    prices = price[good]
    # Last original row at each observed timestamp; no future interpolation.
    unique, first, counts = np.unique(ticks, return_index=True, return_counts=True)
    last_price = prices[first + counts - 1]
    grid_start, grid_end = int(unique[0]), int(unique[-1])
    grid = np.full(grid_end - grid_start + 1, np.nan)
    grid[unique - grid_start] = last_price
    grid = pd.Series(grid).ffill().to_numpy()
    return {"frame": frame, "tick": tick, "price": price, "ticks": ticks,
            "prices": prices, "grid": grid, "start": grid_start,
            "end": grid_end, "duplicate": int(len(ticks)-len(unique)),
            "missing_grid": int(grid.size-len(unique))}


def origins(day, dense=False):
    if dense:
        candidates = np.arange(9*36000+30*600, 15*36000+24*600+1, 1800)
    else:
        candidates = [10*36000+30*600, 11*36000, 11*36000+30*600]
    return [int(c) for c in candidates if c-3000 >= day["start"] and
            c+HORIZON-1 <= day["end"]]


def predict_baselines(day, c, target_ticks):
    g = day["grid"]
    i = c - 1 - day["start"]
    p0 = g[i]
    h = (target_ticks - (c-1))/10
    med10 = np.median(g[i-99:i+1])
    med30 = np.median(g[i-299:i+1])
    past60 = g[i-600]
    trend = ((p0-past60)/60)*h
    older10 = np.median(g[i-699:i-599])
    median_trend = ((med10-older10)/60)*h
    return {"last_price": np.full(len(h),p0), "median10": np.full(len(h),med10),
            "median30": np.full(len(h),med30), "trend025": p0+0.25*trend,
            "trend100": p0+trend, "median_trend025": p0+0.25*median_trend}


def metrics(rows, subset):
    z = rows[subset]
    return z.groupby("model").apply(lambda x: pd.Series({
        "mae": np.average(x.mae, weights=x.n), "worst_mae": x.mae.max(),
        "windows": len(x), "days": x.day.nunique(),
        "day_equal_mae": x.groupby("day").apply(
            lambda y: np.average(y.mae, weights=y.n), include_groups=False).mean()
    }), include_groups=False).reset_index()


def main():
    days = {d: read_day(d) for d in range(1,15)}
    target = days[14]
    c11 = 11*36000
    target_mask = (target["tick"] >= c11) & (target["tick"] < c11+1800) & np.isnan(target["price"])
    assert target_mask.sum() == 1799
    quality = []
    for d,x in days.items():
        quality.append({"day":d,"rows":len(x["frame"]),"last_tick":x["end"],
                        "duplicate_timestamps":x["duplicate"],"missing_grid_ticks":x["missing_grid"]})
    pd.DataFrame(quality).to_csv(OUT/"data_quality.csv",index=False)
    print("loaded days",round(time.time()-START,1),flush=True)
    rows = []
    for d in range(5,11):
        x = days[d]
        for dense in (False,True):
            for c in origins(x,dense):
                lo = np.searchsorted(x["ticks"],c)
                hi = np.searchsorted(x["ticks"],c+1800)
                tt = x["ticks"][lo:hi]
                yy = x["prices"][lo:hi]
                assert len(yy)>0
                preds = predict_baselines(x,c,tt)
                for name,p in preds.items():
                    rows.append({"day":d,"origin_tick":c,"origin":f"{c//36000:02d}:{(c//600)%60:02d}:{(c//10)%60:02d}",
                                 "set":"dense" if dense else "legacy", "model":name,
                                 "n":len(yy),"mae":np.abs(yy-p).mean()})
    r = pd.DataFrame(rows)
    dev=r
    tables={}
    for label,df,mask in [("dev_11",dev,(dev.origin_tick==c11)&(dev.set=="legacy")),
                          ("dev_18",dev,dev.set=="legacy"),
                          ("dev_dense",dev,dev.set=="dense")]:
        t=metrics(df,mask)
        t.to_csv(OUT/f"{label}_metrics.csv",index=False)
        tables[label]=t.set_index("model").to_dict("index")
    # All candidates face the same predeclared promotion checks.
    base="last_price"
    decisions=[]
    for model in ["median10","median30","trend025","trend100","median_trend025"]:
        b11=tables["dev_11"][base]["mae"]
        m11=tables["dev_11"][model]["mae"]
        pivot=dev[(dev.set=="legacy")&(dev.origin_tick==c11)].pivot(index="day",columns="model",values="mae")
        wins=int((pivot[model]<pivot[base]-1e-10).sum())
        passed=(m11<=0.99*b11 and wins>=4 and
                tables["dev_18"][model]["mae"]<=tables["dev_18"][base]["mae"] and
                tables["dev_dense"][model]["day_equal_mae"]<=tables["dev_dense"][base]["day_equal_mae"] and
                tables["dev_18"][model]["worst_mae"]<=1.25*tables["dev_18"][base]["worst_mae"] and
                tables["dev_dense"][model]["worst_mae"]<=1.25*tables["dev_dense"][base]["worst_mae"])
        decisions.append({"model":model,"vs":base,"wins":wins,"passed":bool(passed),
                          "dev_11_mae":m11,"dev_dense_day_equal_mae":tables["dev_dense"][model]["day_equal_mae"]})
    eligible=[x for x in decisions if x["passed"]]
    chosen=min(eligible,key=lambda x:x["dev_11_mae"])["model"] if eligible else base
    # Next candidate must improve on the best simple method under the same checks.
    if chosen!=base:
        for z in decisions:
            if z["model"]!=chosen:
                z["passed_vs_chosen"]=False
    pd.DataFrame(decisions).to_csv(OUT/"selection_checks.csv",index=False)
    print("baseline selected",chosen,"elapsed",round(time.time()-START,1),flush=True)
    # Holdout is calculated only after the method has been frozen.
    final_rows=[]
    for d in range(11,14):
        x=days[d]
        for dense in (False,True):
            for c in origins(x,dense):
                lo=np.searchsorted(x["ticks"],c)
                hi=np.searchsorted(x["ticks"],c+HORIZON)
                tt=x["ticks"][lo:hi]; yy=x["prices"][lo:hi]
                pp=predict_baselines(x,c,tt)[chosen]
                final_rows.append({"day":d,"origin_tick":c,
                                   "origin":f"{c//36000:02d}:{(c//600)%60:02d}:{(c//10)%60:02d}",
                                   "set":"dense" if dense else "legacy","model":chosen,
                                   "n":len(yy),"mae":np.abs(yy-pp).mean()})
    final=pd.DataFrame(final_rows)
    for label,mask in [("final_11",(final.origin_tick==c11)&(final.set=="legacy")),
                       ("final_dense",final.set=="dense")]:
        metrics(final,mask).to_csv(OUT/f"{label}_metrics.csv",index=False)
    pd.concat([dev,final],ignore_index=True).to_csv(OUT/"baseline_windows.csv",index=False)
    tt=target["tick"][target_mask]
    pred=predict_baselines(target,c11,tt)[chosen]
    export=pd.DataFrame({"time":target["frame"].loc[target_mask,"time"].to_numpy(),
                         "predicted_price":pred})
    assert np.isfinite(pred).all() and (pred>0).all()
    assert export.groupby("time").predicted_price.nunique().max()==1
    export.to_csv(ROOT/"prediction.csv",index=False,float_format="%.10f")
    check=pd.read_csv(ROOT/"prediction.csv",dtype={"time":str})
    assert len(check)==1799 and check.time.tolist()==export.time.tolist()
    assert np.isfinite(check.predicted_price).all()
    assert np.max(np.abs(check.predicted_price.to_numpy()-pred))<1e-8
    info={"chosen":chosen,"elapsed_sec":round(time.time()-START,2),
          "target_rows":len(check),"target_unique_times":check.time.nunique(),
          "first_time":check.time.iloc[0],"last_time":check.time.iloc[-1],
          "last_observed_price":target["grid"][c11-1-target["start"]]}
    (OUT/"run_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
