"""
DD Predictor — Flask Backend API v4
=====================================
- Uses CSV for historical data and predictions
- Fetches upcoming fixtures from football-data.co.uk fixtures CSV
- No external API key restrictions
"""

import os
import io
import math
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

DATA_PATH     = os.environ.get("DATA_PATH", "data/processed/all_seasons.csv")
DC_RHO        = 0.10
FORM_WEIGHT   = 0.25
TIME_DECAY    = 0.0018
DRAW_MIN_PROB = 0.27
MAX_GOALS     = 8

_df_cache    = None
_model_cache = None

def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        _df_cache = _df_cache.sort_values("Date").reset_index(drop=True)
        _df_cache["GW"] = _df_cache.groupby("Season").cumcount() // 10 + 1
    return _df_cache

def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = fit_model(get_df())
    return _model_cache

def reload_data():
    global _df_cache, _model_cache
    _df_cache = None
    _model_cache = None
    return get_model()

def dc_tau(hg, ag, lh, la, rho):
    if hg == 0 and ag == 0: return 1 - lh * la * rho
    elif hg == 0 and ag == 1: return 1 + lh * rho
    elif hg == 1 and ag == 0: return 1 + la * rho
    elif hg == 1 and ag == 1: return 1 - rho
    return 1.0

def dc_ll(params, teams, hts, ats, hgs, ags, ws):
    n = len(teams)
    atk = dict(zip(teams, params[:n]))
    dfc = dict(zip(teams, params[n:2*n]))
    ha  = params[-1]
    ll  = 0.0
    for i in range(len(hts)):
        ht, at, hg, ag, w = hts[i], ats[i], hgs[i], ags[i], ws[i]
        if ht not in atk or at not in atk: continue
        lh = np.exp(ha + atk[ht] + dfc[at])
        la = np.exp(atk[at] + dfc[ht])
        tau = dc_tau(hg, ag, lh, la, DC_RHO)
        if tau <= 0: continue
        ll += w * (np.log(tau) + hg*np.log(lh) - lh - math.lgamma(hg+1)
                               + ag*np.log(la) - la - math.lgamma(ag+1))
    return -ll

def fit_model(df):
    teams = sorted(set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))
    n     = len(teams)
    now   = pd.Timestamp.now()
    ws    = np.exp(-TIME_DECAY * (now - df["Date"]).dt.days.values)
    x0    = np.concatenate([np.ones(n), -np.ones(n), [0.3]])
    bounds = [(0.1,4.0)]*n + [(-3.0,0.0)]*n + [(0.0,1.0)]
    res = minimize(dc_ll, x0,
                   args=(teams, df["HomeTeam"].tolist(), df["AwayTeam"].tolist(),
                         df["FTHG"].tolist(), df["FTAG"].tolist(), ws),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-9})
    return {
        "attack":         dict(zip(teams, res.x[:n])),
        "defence":        dict(zip(teams, res.x[n:2*n])),
        "home_advantage": float(res.x[-1]),
        "teams":          teams,
    }

def poisson_prob(lam, k):
    try: return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    except: return 0.0

def predict_match(ht, at, model, hfm=1.0, afm=1.0):
    atk, dfc, ha = model["attack"], model["defence"], model["home_advantage"]
    if ht not in atk or at not in atk: return None
    lh = max(np.exp(ha + atk[ht] + dfc[at]) * (1 + FORM_WEIGHT*(hfm-1)), 0.1)
    la = max(np.exp(atk[at] + dfc[ht]) * (1 + FORM_WEIGHT*(afm-1)), 0.1)
    pm = np.zeros((MAX_GOALS+1, MAX_GOALS+1))
    for hg, ag in product(range(MAX_GOALS+1), range(MAX_GOALS+1)):
        pm[hg][ag] = max(poisson_prob(lh,hg)*poisson_prob(la,ag)*dc_tau(hg,ag,lh,la,DC_RHO), 0.0)
    if pm.sum() > 0: pm /= pm.sum()
    ph  = float(np.sum(np.tril(pm,-1)))
    pd_ = float(np.sum(np.diag(pm)))
    pa  = float(np.sum(np.triu(pm,1)))
    res = "D" if pd_ >= DRAW_MIN_PROB else ("H" if ph >= pa else "A")
    bh, ba = np.unravel_index(pm.argmax(), pm.shape)
    top5 = pm.flatten().argsort()[-5:][::-1]
    scores = [{"score":f"{i//(MAX_GOALS+1)}-{i%(MAX_GOALS+1)}",
               "probability":round(float(pm[i//(MAX_GOALS+1)][i%(MAX_GOALS+1)])*100,1)} for i in top5]
    return {
        "predicted_result": res,
        "predicted_result_label": {"H":"Home Win","D":"Draw","A":"Away Win"}[res],
        "prob_home_win":    round(ph*100,1),
        "prob_draw":        round(pd_*100,1),
        "prob_away_win":    round(pa*100,1),
        "expected_home_goals": round(float(lh),2),
        "expected_away_goals": round(float(la),2),
        "expected_total_goals": round(float(lh+la),2),
        "predicted_score":  f"{bh}-{ba}",
        "likely_scores":    scores,
        "over_25":          bool(lh+la > 2.5),
    }

def get_form(df, team, n=6):
    home = df[df["HomeTeam"]==team][["Date","FTR"]].copy()
    home["Result"] = home["FTR"].map({"H":"W","D":"D","A":"L"})
    away = df[df["AwayTeam"]==team][["Date","FTR"]].copy()
    away["Result"] = away["FTR"].map({"A":"W","D":"D","H":"L"})
    games = pd.concat([home[["Date","Result"]], away[["Date","Result"]]])
    games = games.sort_values("Date", ascending=False).head(n)
    if len(games) == 0:
        return {"multiplier":1.0,"form_string":"","recent_points":0}
    dw  = np.exp(-0.25*np.arange(len(games)))
    pts = games["Result"].map({"W":3,"D":1,"L":0}).values
    mult = round(0.90+(np.average(pts,weights=dw)/3.0)*0.20, 4)
    return {"multiplier":float(mult),"form_string":"".join(games["Result"].tolist()),
            "recent_points":int(games["Result"].map({"W":3,"D":1,"L":0}).sum())}

def get_cards(df, ht, at):
    hg = df[df["HomeTeam"]==ht]; ag = df[df["AwayTeam"]==at]
    lhy = float(df["HY"].mean()) if "HY" in df.columns else 2.0
    lay = float(df["AY"].mean()) if "AY" in df.columns else 2.0
    hy = float(hg["HY"].mean()) if len(hg)>3 else lhy
    ay = float(ag["AY"].mean()) if len(ag)>3 else lay
    return {"pred_home_yellows":round(hy,1),"pred_away_yellows":round(ay,1),
            "pred_total_yellows":round(hy+ay,1)}

def get_h2h(df, ht, at, n=5):
    mask = (((df["HomeTeam"]==ht)&(df["AwayTeam"]==at))|
            ((df["HomeTeam"]==at)&(df["AwayTeam"]==ht)))
    h2h = df[mask].sort_values("Date",ascending=False).head(n)
    return [{"date":r["Date"].strftime("%Y-%m-%d"),"home_team":r["HomeTeam"],
             "away_team":r["AwayTeam"],"score":f"{int(r['FTHG'])}-{int(r['FTAG'])}",
             "result":r["FTR"]} for _,r in h2h.iterrows()]

def build_fixture_list(rows, df, model, gameweek, season, source="csv"):
    out = []
    for _, m in rows.iterrows():
        ht = str(m["HomeTeam"]).strip()
        at = str(m["AwayTeam"]).strip()
        dt = m["Date"].strftime("%Y-%m-%d") if hasattr(m["Date"], "strftime") else str(m["Date"])[:10]
        hf = get_form(df, ht); af = get_form(df, at)
        pred = predict_match(ht, at, model, hf["multiplier"], af["multiplier"])
        if not pred: continue
        cards = get_cards(df, ht, at)
        h2h   = get_h2h(df, ht, at)
        actual = None
        ftr = m.get("FTR","") if source=="csv" else ""
        if pd.notna(ftr) and str(ftr) in ["H","D","A"]:
            actual = {"home_goals":int(m["FTHG"]),"away_goals":int(m["FTAG"]),"result":ftr}
        out.append({
            "fixture_id":f"{ht}-{at}-{dt}","match_date":dt,
            "match_time":str(m.get("Time",""))[:5] if "Time" in m.index else "",
            "status":"FINISHED" if actual else "SCHEDULED",
            "home_team":ht,"away_team":at,"prediction":pred,
            "cards":cards,"home_form":hf,"away_form":af,
            "head_to_head":h2h,"actual":actual,
        })
    return jsonify({
        "gameweek":gameweek,"season":season,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model_version":"v5","source":source,
        "model_accuracy":{"overall":53.3,"home_win":53.9,"away_win":52.7,"over_under":58.0},
        "fixtures":out,
    })

# ── ROUTES ──

@app.route("/api/health")
def health():
    df = get_df()
    return jsonify({
        "status":"ok","model":"DD Predictor v5",
        "timestamp":datetime.now(timezone.utc).isoformat(),
        "seasons":df["Season"].unique().tolist(),
        "latest_season":df["Season"].max(),
        "total_fixtures":len(df),
    })

@app.route("/api/seasons")
def seasons():
    df = get_df()
    result = []
    for season in sorted(df["Season"].unique()):
        s = df[df["Season"]==season]
        gws = sorted(s["GW"].unique().tolist())
        result.append({"season":season,"gameweeks":gws,"max_gw":max(gws),"fixtures":len(s)})
    return jsonify({"seasons":result})

@app.route("/api/predictions/<int:gameweek>", methods=["GET","POST"])
def predictions(gameweek):
    """Historical gameweek predictions from CSV"""
    try:
        df     = get_df()
        model  = get_model()
        season = request.args.get("season", df["Season"].max())
        rows   = df[(df["Season"]==season) & (df["GW"]==gameweek)]
        if len(rows) == 0:
            available = sorted(df[df["Season"]==season]["GW"].unique())
            return jsonify({"error":f"GW{gameweek} not found in {season}. Available: {available}"}), 404
        return build_fixture_list(rows, df, model, gameweek, season, source="csv")
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/upcoming/<int:gameweek>")
def upcoming(gameweek):
    """
    Fetch upcoming fixtures from football-data.co.uk and predict them.
    Works for any future gameweek — no CSV update needed.
    """
    try:
        df    = get_df()
        model = get_model()

        # Pull fixtures CSV from football-data.co.uk
        resp = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=15)
        resp.raise_for_status()
        fix_df = pd.read_csv(io.StringIO(resp.text), encoding="latin1")

        # Filter to Premier League
        if "Div" in fix_df.columns:
            fix_df = fix_df[fix_df["Div"] == "E0"]

        fix_df["Date"] = pd.to_datetime(fix_df["Date"], dayfirst=True, errors="coerce")
        fix_df = fix_df.dropna(subset=["Date","HomeTeam","AwayTeam"])
        fix_df = fix_df.sort_values("Date").reset_index(drop=True)

        # Assign GW numbers — offset from last played GW
        current_season = df["Season"].max()
        season_df = df[df["Season"]==current_season]
        max_played_gw = int(season_df["GW"].max()) if len(season_df) > 0 else 34
        fix_df["GW"] = fix_df.index // 10 + max_played_gw + 1

        gw_rows = fix_df[fix_df["GW"]==gameweek]

        if len(gw_rows) == 0:
            available = sorted(fix_df["GW"].unique().tolist())
            return jsonify({
                "error": f"GW{gameweek} not found in upcoming fixtures.",
                "available_upcoming": available,
            }), 404

        return build_fixture_list(gw_rows, df, model, gameweek, current_season, source="upcoming")

    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/reload", methods=["POST"])
def reload():
    try:
        model = reload_data()
        df    = get_df()
        return jsonify({"status":"reloaded","fixtures":len(df),"seasons":df["Season"].unique().tolist(),"teams":len(model["teams"])})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/team/<team_name>")
def team_stats(team_name):
    try:
        df = get_df()
        home = df[df["HomeTeam"]==team_name]
        away = df[df["AwayTeam"]==team_name]
        all_games = len(home)+len(away)
        if all_games == 0:
            return jsonify({"error":f"Team '{team_name}' not found"}), 404
        h = home[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HY","AY"]].copy()
        h["TeamGoals"]=h["FTHG"]; h["OppGoals"]=h["FTAG"]; h["Opponent"]=h["AwayTeam"]
        h["Venue"]="Home"; h["Result"]=h["FTR"].map({"H":"W","D":"D","A":"L"}); h["Yellows"]=h["HY"]
        a = away[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HY","AY"]].copy()
        a["TeamGoals"]=a["FTAG"]; a["OppGoals"]=a["FTHG"]; a["Opponent"]=a["HomeTeam"]
        a["Venue"]="Away"; a["Result"]=a["FTR"].map({"A":"W","D":"D","H":"L"}); a["Yellows"]=a["AY"]
        recent = pd.concat([h,a]).sort_values("Date",ascending=False).head(10)
        last_10 = [{"date":r["Date"].strftime("%Y-%m-%d"),"opponent":r["Opponent"],
                    "venue":r["Venue"],"score":f"{int(r['TeamGoals'])}-{int(r['OppGoals'])}",
                    "result":r["Result"],"yellows":int(r["Yellows"]) if not pd.isna(r["Yellows"]) else 0}
                   for _,r in recent.iterrows()]
        seasons_out = []
        for season in sorted(df["Season"].unique()):
            sh=df[(df["HomeTeam"]==team_name)&(df["Season"]==season)]
            sa=df[(df["AwayTeam"]==team_name)&(df["Season"]==season)]
            sg=len(sh)+len(sa)
            if sg==0: continue
            sw=int((sh["FTR"]=="H").sum()+(sa["FTR"]=="A").sum())
            sd=int((sh["FTR"]=="D").sum()+(sa["FTR"]=="D").sum())
            sl=int((sh["FTR"]=="A").sum()+(sa["FTR"]=="H").sum())
            gf=int(sh["FTHG"].sum()+sa["FTAG"].sum()); ga=int(sh["FTAG"].sum()+sa["FTHG"].sum())
            seasons_out.append({"season":season,"played":sg,"won":sw,"drawn":sd,"lost":sl,
                                 "gf":gf,"ga":ga,"gd":gf-ga,"points":sw*3+sd})
        wins=int((home["FTR"]=="H").sum()+(away["FTR"]=="A").sum())
        draws=int((home["FTR"]=="D").sum()+(away["FTR"]=="D").sum())
        losses=int((home["FTR"]=="A").sum()+(away["FTR"]=="H").sum())
        gf=int(home["FTHG"].sum()+away["FTAG"].sum()); ga=int(home["FTAG"].sum()+away["FTHG"].sum())
        hy=float(home["HY"].mean()) if "HY" in df.columns else 0
        ay=float(away["AY"].mean()) if "AY" in df.columns else 0
        return jsonify({
            "team":team_name,
            "summary":{"games":all_games,"wins":wins,"draws":draws,"losses":losses,
                        "goals_for":gf,"goals_against":ga,"goal_diff":gf-ga,
                        "win_rate":round(wins/all_games*100,1),
                        "avg_goals_scored":round(gf/all_games,2),
                        "avg_goals_conceded":round(ga/all_games,2),
                        "avg_home_yellows":round(hy,2),"avg_away_yellows":round(ay,2)},
            "last_10":last_10,"seasons":seasons_out,
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/table")
def form_table():
    try:
        df = get_df()
        season = request.args.get("season", df["Season"].max())
        s = df[df["Season"]==season]
        teams = sorted(set(s["HomeTeam"].tolist()+s["AwayTeam"].tolist()))
        rows = []
        for team in teams:
            h=s[s["HomeTeam"]==team]; a=s[s["AwayTeam"]==team]
            g=len(h)+len(a)
            if g==0: continue
            w=int((h["FTR"]=="H").sum()+(a["FTR"]=="A").sum())
            d=int((h["FTR"]=="D").sum()+(a["FTR"]=="D").sum())
            l=int((h["FTR"]=="A").sum()+(a["FTR"]=="H").sum())
            gf=int(h["FTHG"].sum()+a["FTAG"].sum()); ga=int(h["FTAG"].sum()+a["FTHG"].sum())
            form=get_form(df,team,6)
            rows.append({"team":team,"played":g,"won":w,"drawn":d,"lost":l,
                         "gf":gf,"ga":ga,"gd":gf-ga,"points":w*3+d,"form":form["form_string"]})
        rows.sort(key=lambda x:(-x["points"],-(x["gd"]),-x["gf"]))
        for i,r in enumerate(rows): r["position"]=i+1
        return jsonify({"season":season,"table":rows})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
