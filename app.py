"""
DD Predictor — Flask Backend API v6
=====================================
Improvements:
  1. Referee card adjustment
  2. League position differential
  3. Derby flag (attack + card boost)
  4. Home/away form split
"""

import os
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

DATA_PATH  = os.environ.get("DATA_PATH", "data/processed/all_seasons.csv")
FD_API_KEY = os.environ.get("FOOTBALL_API_KEY", "642390403d9549ebbeb29c158f77dfcd")
FD_BASE    = "https://api.football-data.org/v4"
FD_HEADERS = {"X-Auth-Token": FD_API_KEY}

DC_RHO        = 0.10
FORM_WEIGHT   = 0.25
TIME_DECAY    = 0.0018
DRAW_MIN_PROB = 0.27
MAX_GOALS     = 8
DERBY_ATTACK_BOOST = 1.06
DERBY_CARD_BOOST   = 1.25

TEAM_MAP = {
    "Manchester City FC":"Man City","Manchester United FC":"Man United",
    "Arsenal FC":"Arsenal","Liverpool FC":"Liverpool","Chelsea FC":"Chelsea",
    "Tottenham Hotspur FC":"Tottenham","Newcastle United FC":"Newcastle",
    "Aston Villa FC":"Aston Villa","West Ham United FC":"West Ham",
    "Brighton & Hove Albion FC":"Brighton","Wolverhampton Wanderers FC":"Wolves",
    "Fulham FC":"Fulham","Brentford FC":"Brentford","Crystal Palace FC":"Crystal Palace",
    "Everton FC":"Everton","Nottingham Forest FC":"Nott'm Forest",
    "Bournemouth AFC":"Bournemouth","AFC Bournemouth":"Bournemouth",
    "Leicester City FC":"Leicester","Ipswich Town FC":"Ipswich",
    "Southampton FC":"Southampton","Luton Town FC":"Luton","Burnley FC":"Burnley",
    "Sheffield United FC":"Sheffield United","Sunderland AFC":"Sunderland",
    "Leeds United FC":"Leeds","Coventry City FC":"Coventry",
    "Middlesbrough FC":"Middlesbrough","Watford FC":"Watford","Norwich City FC":"Norwich",
}

DERBIES = [
    {"Arsenal","Tottenham"},
    {"Man City","Man United"},
    {"Liverpool","Everton"},
    {"Chelsea","Arsenal"},
    {"Chelsea","Tottenham"},
    {"Chelsea","West Ham"},
    {"West Ham","Tottenham"},
    {"Newcastle","Sunderland"},
    {"Crystal Palace","Brighton"},
    {"Brentford","Fulham"},
    {"Leeds","Man United"},
]

def norm(name): return TEAM_MAP.get(name, name)
def is_derby(h, a): return any(len({h,a}&d)==2 for d in DERBIES)

# ── CACHE ──
_df_cache = None
_model_cache = None
_ref_cache = None
_table_cache = None

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

def get_ref_stats():
    global _ref_cache
    if _ref_cache is None:
        _ref_cache = build_referee_stats(get_df())
    return _ref_cache

def get_table():
    global _table_cache
    if _table_cache is None:
        _table_cache = build_table(get_df())
    return _table_cache

def reload_all():
    global _df_cache, _model_cache, _ref_cache, _table_cache
    _df_cache = _model_cache = _ref_cache = _table_cache = None
    get_model(); get_ref_stats(); get_table()

# ── TABLE ──
def build_table(df):
    teams = sorted(set(df["HomeTeam"].tolist()+df["AwayTeam"].tolist()))
    table = {}
    for team in teams:
        h=df[df["HomeTeam"]==team]; a=df[df["AwayTeam"]==team]
        w=int((h["FTR"]=="H").sum()+(a["FTR"]=="A").sum())
        d=int((h["FTR"]=="D").sum()+(a["FTR"]=="D").sum())
        gf=int(h["FTHG"].sum()+a["FTAG"].sum()); ga=int(h["FTAG"].sum()+a["FTHG"].sum())
        table[team]={"points":w*3+d,"gd":gf-ga,"gf":gf}
    sorted_teams=sorted(table.keys(),key=lambda t:(-table[t]["points"],-table[t]["gd"],-table[t]["gf"]))
    return {team:i+1 for i,team in enumerate(sorted_teams)}

def position_adjustment(hp, ap, n=20):
    gap = ap - hp
    norm_gap = gap / n
    hm = max(0.85, min(1.15, 1.0+(norm_gap*0.08)))
    am = max(0.85, min(1.15, 1.0-(norm_gap*0.08)))
    return hm, am

# ── REFEREE ──
def build_referee_stats(df):
    if "Referee" not in df.columns: return {}
    league_avg_y = (df["HY"]+df["AY"]).mean() if "HY" in df.columns else 4.0
    league_avg_r = (df["HR"]+df["AR"]).mean() if "HR" in df.columns else 0.2
    stats = {}
    for ref in df["Referee"].dropna().unique():
        rg = df[df["Referee"]==ref]
        if len(rg)<5: continue
        shrink = min(len(rg)/20, 1.0)
        avg_y = (rg["HY"]+rg["AY"]).mean()
        avg_r = (rg["HR"]+rg["AR"]).mean()
        stats[ref] = {
            "yellow_mult": round(shrink*(avg_y/league_avg_y)+(1-shrink)*1.0, 3),
            "red_mult":    round(shrink*(avg_r/league_avg_r)+(1-shrink)*1.0, 3),
        }
    return stats

# ── DIXON-COLES MLE ──
def dc_tau(hg,ag,lh,la,rho):
    if hg==0 and ag==0: return 1-lh*la*rho
    elif hg==0 and ag==1: return 1+lh*rho
    elif hg==1 and ag==0: return 1+la*rho
    elif hg==1 and ag==1: return 1-rho
    return 1.0

def dc_ll(params,teams,hts,ats,hgs,ags,ws):
    n=len(teams); atk=dict(zip(teams,params[:n])); dfc=dict(zip(teams,params[n:2*n])); ha=params[-1]; ll=0.0
    for i in range(len(hts)):
        ht,at,hg,ag,w=hts[i],ats[i],hgs[i],ags[i],ws[i]
        if ht not in atk or at not in atk: continue
        lh=np.exp(ha+atk[ht]+dfc[at]); la=np.exp(atk[at]+dfc[ht])
        tau=dc_tau(hg,ag,lh,la,DC_RHO)
        if tau<=0: continue
        ll+=w*(np.log(tau)+hg*np.log(lh)-lh-math.lgamma(hg+1)+ag*np.log(la)-la-math.lgamma(ag+1))
    return -ll

def fit_model(df):
    teams=sorted(set(df["HomeTeam"].tolist()+df["AwayTeam"].tolist())); n=len(teams)
    ws=np.exp(-TIME_DECAY*(pd.Timestamp.now()-df["Date"]).dt.days.values)
    x0=np.concatenate([np.ones(n),-np.ones(n),[0.3]])
    bounds=[(0.1,4.0)]*n+[(-3.0,0.0)]*n+[(0.0,1.0)]
    res=minimize(dc_ll,x0,args=(teams,df["HomeTeam"].tolist(),df["AwayTeam"].tolist(),
                                 df["FTHG"].tolist(),df["FTAG"].tolist(),ws),
                 method="L-BFGS-B",bounds=bounds,options={"maxiter":200,"ftol":1e-9})
    return {"attack":dict(zip(teams,res.x[:n])),"defence":dict(zip(teams,res.x[n:2*n])),
            "home_advantage":float(res.x[-1]),"teams":teams}

# ── FORM (home/away split) ──
def get_form(df, team, n=6):
    home=df[df["HomeTeam"]==team][["Date","FTR"]].copy()
    home["Result"]=home["FTR"].map({"H":"W","D":"D","A":"L"})
    away=df[df["AwayTeam"]==team][["Date","FTR"]].copy()
    away["Result"]=away["FTR"].map({"A":"W","D":"D","H":"L"})

    def mult(games, res_map):
        if len(games)==0: return 1.0
        games=games.sort_values("Date",ascending=False).head(n)
        dw=np.exp(-0.25*np.arange(len(games)))
        pts=games["FTR"].map(res_map).fillna(1).values
        return round(0.90+(np.average(pts,weights=dw)/3.0)*0.20,4)

    hm = mult(home, {"H":3,"D":1,"A":0})
    am = mult(away, {"A":3,"D":1,"H":0})

    all_games=pd.concat([
        home[["Date","Result"]],away[["Date","Result"]]
    ]).sort_values("Date",ascending=False).head(n)
    form_str="".join(all_games["Result"].tolist()) if len(all_games)>0 else ""

    return {"home_mult":float(hm),"away_mult":float(am),
            "combined_mult":float((hm+am)/2),"form_string":form_str,
            "recent_points":int(all_games["Result"].map({"W":3,"D":1,"L":0}).sum()) if len(all_games)>0 else 0}

# ── CARDS ──
def get_cards(df, ht, at, referee=None, ref_stats=None, derby=False):
    hg=df[df["HomeTeam"]==ht]; ag=df[df["AwayTeam"]==at]
    lhy=float(df["HY"].mean()) if "HY" in df.columns else 2.0
    lay=float(df["AY"].mean()) if "AY" in df.columns else 2.0
    hy=float(hg["HY"].mean()) if len(hg)>3 else lhy
    ay=float(ag["AY"].mean()) if len(ag)>3 else lay
    ref_mult=1.0
    if referee and ref_stats and referee in ref_stats:
        ref_mult=ref_stats[referee]["yellow_mult"]
    derby_mult=DERBY_CARD_BOOST if derby else 1.0
    total_mult=ref_mult*derby_mult
    return {"pred_home_yellows":round(hy*total_mult,1),
            "pred_away_yellows":round(ay*total_mult,1),
            "pred_total_yellows":round((hy+ay)*total_mult,1),
            "referee_multiplier":round(ref_mult,3),"derby_flag":derby}

# ── H2H ──
def get_h2h(df,ht,at,n=5):
    mask=(((df["HomeTeam"]==ht)&(df["AwayTeam"]==at))|((df["HomeTeam"]==at)&(df["AwayTeam"]==ht)))
    h2h=df[mask].sort_values("Date",ascending=False).head(n)
    return [{"date":r["Date"].strftime("%Y-%m-%d"),"home_team":r["HomeTeam"],
             "away_team":r["AwayTeam"],"score":f"{int(r['FTHG'])}-{int(r['FTAG'])}",
             "result":r["FTR"]} for _,r in h2h.iterrows()]

# ── PREDICT ──
def poisson_prob(lam,k):
    try: return (math.exp(-lam)*(lam**k))/math.factorial(k)
    except: return 0.0

def predict_match(ht, at, model, hf, af, home_pos=10, away_pos=10, derby=False):
    atk,dfc,ha=model["attack"],model["defence"],model["home_advantage"]
    if ht not in atk or at not in atk: return None
    lh=np.exp(ha+atk[ht]+dfc[at])
    la=np.exp(atk[at]+dfc[ht])
    # Home/away form split
    lh*=(1+FORM_WEIGHT*(hf["home_mult"]-1))
    la*=(1+FORM_WEIGHT*(af["away_mult"]-1))
    # Position adjustment
    hpm,apm=position_adjustment(home_pos,away_pos)
    lh*=hpm; la*=apm
    # Derby boost
    if derby: lh*=DERBY_ATTACK_BOOST; la*=DERBY_ATTACK_BOOST
    lh=max(lh,0.1); la=max(la,0.1)

    pm=np.zeros((MAX_GOALS+1,MAX_GOALS+1))
    for hg,ag in product(range(MAX_GOALS+1),range(MAX_GOALS+1)):
        pm[hg][ag]=max(poisson_prob(lh,hg)*poisson_prob(la,ag)*dc_tau(hg,ag,lh,la,DC_RHO),0.0)
    if pm.sum()>0: pm/=pm.sum()

    ph=float(np.sum(np.tril(pm,-1)))
    pd_=float(np.sum(np.diag(pm)))
    pa=float(np.sum(np.triu(pm,1)))
    result="D" if pd_>=DRAW_MIN_PROB else ("H" if ph>=pa else "A")
    bh,ba=np.unravel_index(pm.argmax(),pm.shape)
    top5=pm.flatten().argsort()[-5:][::-1]
    scores=[{"score":f"{i//(MAX_GOALS+1)}-{i%(MAX_GOALS+1)}",
             "probability":round(float(pm[i//(MAX_GOALS+1)][i%(MAX_GOALS+1)])*100,1)} for i in top5]
    return {
        "predicted_result":result,
        "predicted_result_label":{"H":"Home Win","D":"Draw","A":"Away Win"}[result],
        "prob_home_win":round(ph*100,1),"prob_draw":round(pd_*100,1),"prob_away_win":round(pa*100,1),
        "expected_home_goals":round(float(lh),2),"expected_away_goals":round(float(la),2),
        "expected_total_goals":round(float(lh+la),2),
        "predicted_score":f"{bh}-{ba}","likely_scores":scores,"over_25":bool(lh+la>2.5),
        "derby":derby,"home_pos":home_pos,"away_pos":away_pos,
    }

# ── PROCESS MATCHES ──
def process_matches(matches, gameweek, season):
    df        = get_df()
    model     = get_model()
    ref_stats = get_ref_stats()
    table     = get_table()
    out       = []

    for m in matches:
        ht = norm(m["homeTeam"]["name"])
        at = norm(m["awayTeam"]["name"])
        dt = m.get("utcDate","")[:10]
        tm = m.get("utcDate","")[11:16]

        hf      = get_form(df, ht)
        af      = get_form(df, at)
        hp      = table.get(ht, 10)
        ap      = table.get(at, 10)
        derby   = is_derby(ht, at)
        referee = m.get("referees",[{}])[0].get("name") if m.get("referees") else None

        pred  = predict_match(ht, at, model, hf, af, hp, ap, derby)
        cards = get_cards(df, ht, at, referee, ref_stats, derby)
        h2h   = get_h2h(df, ht, at)

        if not pred: continue

        status = m.get("status","SCHEDULED")
        actual = None
        if status == "FINISHED":
            score = m.get("score",{}).get("fullTime",{})
            hg = score.get("home") or 0
            ag = score.get("away") or 0
            actual = {"home_goals":hg,"away_goals":ag,
                      "result":"H" if hg>ag else "D" if hg==ag else "A"}

        out.append({
            "fixture_id":m.get("id"),"match_date":dt,"match_time":tm,
            "status":status,"home_team":ht,"away_team":at,
            "prediction":pred,"cards":cards,
            "home_form":hf,"away_form":af,
            "head_to_head":h2h,"actual":actual,
            "derby":derby,"home_pos":hp,"away_pos":ap,
            "referee":referee,
        })

    return jsonify({
        "gameweek":gameweek,"season":season,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model_version":"v6",
        "model_accuracy":{"overall":53.3,"home_win":53.9,"away_win":52.7,"over_under":57.8},
        "fixtures":out,
    })

# ── ROUTES ──

@app.route("/api/value_bets/<int:gameweek>")
def value_bets(gameweek):
    """
    Find Away Win value bets for a gameweek.
    For historical gameweeks: uses closing odds from our CSV.
    Strategy: Away Win with 5-8% edge over bookmaker implied probability.
    """
    try:
        url = f"{FD_BASE}/competitions/PL/matches"
        matches = []; season_label = "2025-26"
        for year, label in [(2025,"2025-26"),(2024,"2024-25")]:
            resp = requests.get(url, headers=FD_HEADERS,
                                params={"matchday":gameweek,"season":year}, timeout=15)
            if resp.status_code == 200:
                matches = resp.json().get("matches",[])
                if matches: season_label=label; break

        if not matches:
            return jsonify({"error":f"No fixtures found for Gameweek {gameweek}"}), 404

        df    = get_df()
        model = get_model()
        table = get_table()

        value_bets_list = []
        all_fixtures    = []

        for m in matches:
            ht = norm(m["homeTeam"]["name"])
            at = norm(m["awayTeam"]["name"])
            dt = m.get("utcDate","")[:10]
            tm = m.get("utcDate","")[11:16]
            status = m.get("status","SCHEDULED")

            hf = get_form(df, ht)
            af = get_form(df, at)
            hp = table.get(ht, 10)
            ap = table.get(at, 10)
            derby = is_derby(ht, at)

            pred = predict_match(ht, at, model, hf, af, hp, ap, derby)
            if not pred: continue

            away_prob = pred["prob_away_win"] / 100.0

            # Look up historical odds from CSV for this fixture
            hist_match = df[
                (df["HomeTeam"]==ht) &
                (df["AwayTeam"]==at) &
                (df["Date"].dt.strftime("%Y-%m-%d")==dt)
            ]

            avg_odds  = None
            b365_odds = None
            away_edge = None
            has_odds  = False

            if len(hist_match) > 0 and "AvgA" in hist_match.columns:
                row = hist_match.iloc[0]
                avg_odds  = float(row["AvgA"]) if not pd.isna(row.get("AvgA")) else None
                b365_odds = float(row["B365A"]) if "B365A" in row.index and not pd.isna(row.get("B365A")) else None

                if avg_odds and avg_odds > 1.0:
                    implied = 1.0 / avg_odds
                    away_edge = round(away_prob - implied, 4)
                    has_odds = True

            # For future/unplayed fixtures without odds yet
            if not has_odds:
                avg_odds  = None
                b365_odds = None
                away_edge = None

            fixture_data = {
                "home_team":   ht,
                "away_team":   at,
                "match_date":  dt,
                "match_time":  tm,
                "status":      status,
                "prob_home":   pred["prob_home_win"],
                "prob_draw":   pred["prob_draw"],
                "prob_away":   pred["prob_away_win"],
                "away_prob":   round(away_prob, 4),
                "avg_odds":    avg_odds,
                "b365_odds":   b365_odds,
                "away_edge":   away_edge,
                "has_odds":    has_odds,
                "model_prob":  round(away_prob, 4),
                "implied_prob": round(1.0/avg_odds, 4) if avg_odds and avg_odds > 1.0 else None,
                "expected_value": round((away_prob * (avg_odds-1)) - (1-away_prob), 4) if avg_odds and avg_odds > 1.0 else None,
            }
            all_fixtures.append(fixture_data)

            # Value bet criteria: 5-8% edge, has odds
            if has_odds and away_edge is not None and 0.05 <= away_edge < 0.08:
                value_bets_list.append(fixture_data)

        # Sort value bets by edge descending
        value_bets_list.sort(key=lambda x: x["away_edge"] or 0, reverse=True)

        return jsonify({
            "gameweek":     gameweek,
            "season":       season_label,
            "value_bets":   value_bets_list,
            "all_fixtures":  all_fixtures,
            "strategy":     {"market":"Away Win","min_edge":0.05,"max_edge":0.08},
            "has_live_odds": False,
            "note":         "Using closing odds from historical data. Live odds coming soon.",
        })

    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
# v6 with odds
