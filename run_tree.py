"""Small absolute-error trees; day-forward training and fixed-origin inference."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home()/".codex"/"tmp"/"practice5_deps"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from run_experiment import DATA, OUT, ROOT, HORIZON, origins, predict_baselines, read_day

START=time.time()
C11=11*36000


def enrich(day):
    f=day["frame"]
    for col in ("cum_volume","volume","bid_total","ask_total"):
        day[col]=pd.to_numeric(f[col],errors="coerce").to_numpy(float)
    good=np.isfinite(day["price"])
    day["obs_tick"]=day["tick"][good]
    day["cum_good"]=day["cum_volume"][good]
    day["vol_good"]=day["volume"][good]
    day["bid_good"]=day["bid_total"][good]
    day["ask_good"]=day["ask_total"][good]
    return day


def features(day,c):
    g=day["grid"]
    i=c-1-day["start"]
    if i<3000 or i>=len(g): return None
    loc=np.searchsorted(day["obs_tick"],c,side="left")-1
    if loc<0 or (c-1-day["obs_tick"][loc])>50: return None
    w=day["obs_tick"]
    if np.diff(w[max(0,np.searchsorted(w,c-3000)-1):loc+1]).max(initial=0)>50: return None
    p0=g[i]
    changes=[p0-g[i-k] for k in (10,50,100,300,600,1800)]
    med=[p0-np.median(g[i-k+1:i+1]) for k in (100,300,600)]
    sec=g[i-600:i+1:10]
    vol=[float(np.std(np.diff(sec[-31:]))),float(np.std(np.diff(sec)))]
    def aggregate(window):
        left=np.searchsorted(w,c-window)
        cv=day["cum_good"]
        reported=day["vol_good"][left:loc+1].sum()
        inc=max(0.0,cv[loc]-cv[max(0,left-1)])
        return inc,reported,left
    v10,r10,_=aggregate(100)
    v60,r60,left60=aggregate(600)
    bid=day["bid_good"][loc]; ask=day["ask_good"][loc]
    den=bid+ask
    imbalance=(bid-ask)/den if den>0 else 0.0
    left10=np.searchsorted(w,c-100)
    old=max(0,left10-1)
    oldden=day["bid_good"][old]+day["ask_good"][old]
    oldimb=(day["bid_good"][old]-day["ask_good"][old])/oldden if oldden>0 else 0.0
    price_features=changes+med+vol
    extra=[np.log1p(v10),np.log1p(v60),v10/(v60+1),
           np.log1p(r60),np.sign(v60-r60)*np.log1p(abs(v60-r60)),
           float(v60!=r60),imbalance,imbalance-oldimb,
           (c-1-w[loc])/10,(loc-left60+1)/600, float(den==0)]
    return p0,np.asarray(price_features+extra+[c/36000],float)


def sample_day(day,d,stop=None):
    xx=[]; yy=[]; ww=[]; meta=[]
    for c in range(9*36000+10*600,15*36000+25*600+1,1800):
        if c+HORIZON-1>day["end"] or (stop is not None and c+HORIZON>stop): continue
        feat=features(day,c)
        if feat is None: continue
        p0,base=feat
        lo=np.searchsorted(day["ticks"],c)
        hi=np.searchsorted(day["ticks"],c+HORIZON)
        ticks=day["ticks"][lo:hi]
        prices=day["prices"][lo:hi]
        bins=(ticks-c)//75
        for b in range(24):
            ids=np.flatnonzero(bins==b)
            if len(ids)==0:continue
            rng=np.random.default_rng(np.random.SeedSequence([42,d,int(c),b]))
            selected=rng.choice(ids,size=min(2,len(ids)),replace=False)
            for j in selected:
                h=(ticks[j]-(c-1))/10
                xx.append(np.r_[base,h])
                yy.append(prices[j]-p0)
                ww.append(len(ids)/len(selected))
                meta.append((d,c,int(ticks[j])))
    return np.asarray(xx),np.asarray(yy),np.asarray(ww),meta


def fit_model(x,y,w,leaves,price_only=False):
    if price_only:x=np.column_stack([x[:,:11],x[:,-2:]])
    m=HistGradientBoostingRegressor(loss="absolute_error",learning_rate=.05,
            max_iter=150,max_leaf_nodes=leaves,min_samples_leaf=50,
            l2_regularization=1.,early_stopping=False,random_state=42)
    m.fit(x,y,sample_weight=w)
    return m


def predict(m,day,c,ticks,price_only=False):
    feat=features(day,c)
    if feat is None:
        return predict_baselines(day,c,ticks)["last_price"],True
    p0,base=feat
    h=(ticks-(c-1))/10
    x=np.tile(base,(len(h),1))
    x=np.column_stack([x,h])
    if price_only:x=np.column_stack([x[:,:11],x[:,-2:]])
    return p0+m.predict(x),False


def evaluate(m,day,d,kind,price_only=False):
    rows=[]
    for dense in (False,True):
        for c in origins(day,dense):
            lo=np.searchsorted(day["ticks"],c)
            hi=np.searchsorted(day["ticks"],c+HORIZON)
            ticks=day["ticks"][lo:hi]
            yy=day["prices"][lo:hi]
            pred,fallback=predict(m,day,c,ticks,price_only)
            last=predict_baselines(day,c,ticks)["last_price"]
            for weight in (1.,.5,.25):
                pp=weight*pred+(1-weight)*last
                rows.append({"day":d,"origin_tick":c,"set":"dense" if dense else "legacy",
                             "model":f"{kind}_w{weight:g}","n":len(yy),"mae":np.abs(yy-pp).mean(),
                             "fallback":fallback})
    return rows


def evaluate_B(day,d,previous):
    records=[]; logs=[]
    all_origins={(c,"legacy") for c in origins(day,False)} | {(c,"dense") for c in origins(day,True)}
    start_s=9*36000+30*600
    groups={}
    for c,label in all_origins:
        s=start_s+((c-start_s)//18000)*18000
        groups.setdefault(s,[]).append((c,label))
    for s,items in sorted(groups.items()):
        train=previous+[sample_day(day,d,stop=s)]
        x=np.concatenate([a[0] for a in train]);y=np.concatenate([a[1] for a in train]);w=np.concatenate([a[2] for a in train])
        start=time.time()
        m=fit_model(x,y,w,7)
        logs.append({"day":d,"model":"B_leaf7_full","refresh_tick":s,"train_rows":len(y),
                     "seconds":time.time()-start,"latest_train_day":d})
        for c,label in items:
            lo=np.searchsorted(day["ticks"],c);hi=np.searchsorted(day["ticks"],c+HORIZON)
            ticks=day["ticks"][lo:hi];yy=day["prices"][lo:hi]
            pred,fallback=predict(m,day,c,ticks)
            last=predict_baselines(day,c,ticks)["last_price"]
            for weight in (1.,.5,.25):
                pp=weight*pred+(1-weight)*last
                records.append({"day":d,"origin_tick":c,"set":label,
                                "model":f"B_leaf7_full_w{weight:g}","n":len(yy),
                                "mae":np.abs(yy-pp).mean(),"fallback":fallback})
    return records,logs


def summary(rows,mask):
    z=rows[mask]
    result={}
    for model,g in z.groupby("model"):
        result[model]={"mae":float(np.average(g.mae,weights=g.n)),"worst":float(g.mae.max()),
                      "day_equal":float(g.groupby("day").apply(lambda a:np.average(a.mae,weights=a.n),include_groups=False).mean()),
                      "fallbacks":int(g.fallback.sum()),"windows":len(g)}
    return result


def main():
    days={d:enrich(read_day(d)) for d in range(1,15)}
    samples={d:sample_day(days[d],d) for d in range(1,14)}
    print("samples",{d:len(z[1]) for d,z in samples.items()},"elapsed",round(time.time()-START,1),flush=True)
    records=[]; fits=[]
    # Two predeclared tree sizes, A policy. B uses 30-minute refreshes.
    for d in range(5,11):
        for policy,leaves,price_only in [("A",7,False),("A",15,False),("A",7,True)]:
            train=[samples[k] for k in range(1,d)]
            x=np.concatenate([a[0] for a in train]); y=np.concatenate([a[1] for a in train]); w=np.concatenate([a[2] for a in train])
            name=f"{policy}_leaf{leaves}_{'price' if price_only else 'full'}"
            start=time.time()
            m=fit_model(x,y,w,leaves,price_only)
            records+=evaluate(m,days[d],d,name,price_only)
            fits.append({"day":d,"model":name,"train_rows":len(y),"seconds":time.time()-start,
                         "latest_train_day":d if policy=="B" else d-1})
            print("fit",d,name,"sec",round(time.time()-start,1),"elapsed",round(time.time()-START,1),flush=True)
        scored,logs=evaluate_B(days[d],d,[samples[k] for k in range(1,d)])
        records+=scored;fits+=logs
        print("B refreshes",d,len(logs),"elapsed",round(time.time()-START,1),flush=True)
    r=pd.DataFrame(records)
    r.to_csv(OUT/"tree_windows.csv",index=False)
    pd.DataFrame(fits).to_csv(OUT/"tree_fit_log.csv",index=False)
    legacy=pd.read_csv(OUT/"baseline_windows.csv")
    base=legacy[(legacy.day.between(5,10))&(legacy.model=="last_price")]
    b11=summary(base.assign(fallback=False),base.origin_tick==C11)["last_price"]
    b18=summary(base.assign(fallback=False),base.set=="legacy")["last_price"]
    bdense=summary(base.assign(fallback=False),base.set=="dense")["last_price"]
    s11=summary(r,r.origin_tick==C11); s18=summary(r,r.set=="legacy"); sd=summary(r,r.set=="dense")
    pivot=r[(r.set=="legacy")&(r.origin_tick==C11)].pivot(index="day",columns="model",values="mae")
    bp=base[(base.set=="legacy")&(base.origin_tick==C11)].set_index("day").mae
    checks=[]
    for name,z in s11.items():
        wins=int((pivot[name]<bp-1e-10).sum())
        complete=True
        passed=(complete and z["mae"]<=.99*b11["mae"] and wins>=4 and
                s18[name]["mae"]<=b18["mae"] and sd[name]["day_equal"]<=bdense["day_equal"] and
                s18[name]["worst"]<=1.25*b18["worst"] and sd[name]["worst"]<=1.25*bdense["worst"])
        checks.append({"model":name,"dev11_mae":z["mae"],"dev18_mae":s18[name]["mae"] if complete else np.nan,
                       "dense_day_equal_mae":sd[name]["day_equal"] if complete else np.nan,
                       "wins":wins,"passed":bool(passed),"dense_validation_complete":complete,
                       "dense_fallbacks":sd[name]["fallbacks"] if complete else np.nan})
    pd.DataFrame(checks).to_csv(OUT/"tree_selection_checks.csv",index=False)
    print("checks",checks,"elapsed",round(time.time()-START,1),flush=True)
    eligible=[c for c in checks if c["passed"]]
    chosen=min(eligible,key=lambda z:z["dev11_mae"])["model"] if eligible else "last_price"
    # The predeclared 1% simplicity tie-break applies to the winning model.
    if eligible:
        best=min(z["dev11_mae"] for z in eligible)
        simple=sorted([z for z in eligible if z["dev11_mae"]<=best*1.01],key=lambda z:(len(z["model"]),z["dev11_mae"]))
        chosen=simple[0]["model"]
    print("chosen",chosen,flush=True)
    # Holdout evaluation only after the selection is frozen.
    final=[]
    for d in range(11,14):
        if chosen=="last_price":break
        policy,leaf,feature,weight=chosen.split("_")
        leaf=int(leaf.removeprefix("leaf")); weight=float(weight.removeprefix("w"))
        train=[samples[k] for k in range(1,d)]
        if policy=="B":
            scored,_=evaluate_B(days[d],d,train)
            final+=scored
            continue
        x=np.concatenate([a[0] for a in train]);y=np.concatenate([a[1] for a in train]);w=np.concatenate([a[2] for a in train])
        m=fit_model(x,y,w,leaf,feature=="price")
        final+=evaluate(m,days[d],d,f"{policy}_leaf{leaf}_{feature}",feature=="price")
    if final:
        pd.DataFrame(final).to_csv(OUT/"tree_holdout_windows.csv",index=False)
    # Fit final model, including only samples with completed labels before 11:00 on day14.
    target=days[14]
    target_mask=(target["tick"]>=C11)&(target["tick"]<C11+1800)&np.isnan(target["price"])
    tt=target["tick"][target_mask]
    if chosen=="last_price":
        pred=predict_baselines(target,C11,tt)["last_price"]
    else:
        train=[samples[k] for k in range(1,14)]
        if policy=="B":train.append(sample_day(target,14,stop=C11))
        x=np.concatenate([a[0] for a in train]);y=np.concatenate([a[1] for a in train]);w=np.concatenate([a[2] for a in train])
        m=fit_model(x,y,w,leaf,feature=="price")
        tree,_=predict(m,target,C11,tt,feature=="price")
        last=predict_baselines(target,C11,tt)["last_price"]
        pred=weight*tree+(1-weight)*last
    export=pd.DataFrame({"time":target["frame"].loc[target_mask,"time"].to_numpy(),"predicted_price":pred})
    assert len(export)==1799 and np.isfinite(pred).all() and (pred>0).all()
    assert export.groupby("time").predicted_price.nunique().max()==1
    export.to_csv(ROOT/"prediction.csv",index=False,float_format="%.10f")
    check=pd.read_csv(ROOT/"prediction.csv",dtype={"time":str})
    assert check.time.tolist()==export.time.tolist() and np.max(np.abs(check.predicted_price.to_numpy()-pred))<1e-8
    (OUT/"tree_run_info.json").write_text(json.dumps({"chosen":chosen,"elapsed_sec":time.time()-START,
             "n_predictions":len(check),"n_unique_times":check.time.nunique(),
             "min_prediction":float(pred.min()),"max_prediction":float(pred.max())},indent=2),encoding="utf-8")
    print("done",round(time.time()-START,1),flush=True)


if __name__=="__main__":main()
