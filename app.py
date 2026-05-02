"""
DD Predictor — Flask Backend API
=================================
Serves Premier League predictions to the frontend.
Connects directly to the v5 Dixon-Coles MLE model.

Endpoints:
  GET  /api/health                     — health check
  POST /api/predictions/<gameweek>     — predictions (frontend sends fixtures)
  GET  /api/predictions/<gameweek>     — predictions (backend fetches fixtures)
  GET  /api/team/<team_name>           — team stats and history
  GET  /api/table                      — current form table
"""

import os
import json
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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_KEY      = os.environ.get("FOOTBALL_API_KEY", "642390403d9549ebbeb29c158f77dfcd")
API_BASE     = "https://api.football-data.org/v4"
DATA_PATH    = os.environ.get("DATA_PATH", "data/processed/all_seasons.csv")
HEADERS      = {"X-Auth-Token": API_KEY}

DC_RHO        = 0.10
FORM_GAMES    = 6
FORM_WEIGHT   = 0.25
TIME_DECAY    = 0.0018
DRAW_MIN_PROB = 0.27
MAX_GOALS     = 8

TEAM_NAME_MAP = {
    "Manchester City FC":         "Man City",
    "Manchester United FC":       "Man United",
    "Arsenal FC":                 "Arsenal",
    "Liverpool FC":               "Liverpool",
    "Chelsea FC":                 "Chelsea",
    "Tottenham Hotspur FC":       "Tottenham",
    "Newcastle United FC":        "Newcastle",
    "Aston Villa FC":             "Aston Villa",
    "West Ham United FC":         "West Ham",
    "Brighton & Hove Albion FC":  "Brighton",
    "Wolverhampton Wanderers FC": "Wolves",
    "Fulham FC":                  "Fulham",
    "Brentford FC":               "Brentford",
    "Crystal Palace FC":          "Crystal Palace",
    "Everton FC":                 "Everton",
    "Nottingham Forest FC":       "Nott'm Forest",
    "Bournemouth AFC":            "Bournemouth",
    "AFC Bournemouth":            "Bournemouth",
    "Leicester City FC":          "Leicester",
    "Ipswich Town FC":            "Ipswich",
    "Southampton FC":             "Southampton",
    "Luton Town FC":              "Luton",
    "Burnley FC":                 "Burnley",
    "Sheffield United FC":        "Sheffield United",
    "Sunderland AFC":             "Sunderland",
    "Leeds United FC":            "Leeds",
    "Watford FC":                 "Watford",
    "Norwich City FC":            "Norwich",
    "Coventry City FC":           "Coventry",
    "Middlesbrough FC":           "Middlesbrough",
}

def normalise_team(name):
    return TEAM_NAME_MAP.get(name, name)

# ─────────────────────────────────────────────
# DATA (cached in memory)
# ─────────────────────────────────────────────

_df_cache    = None
_model_cache = None

def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        _df_cache = _df_cache.sort_values("Date").reset_index(drop=True)
    return _df_cache

def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = fit_model(get_df())
    return _model_cache

# ─────────────────────────────────────────────
# DIXON-COLES MLE
# ─────────────────────────────────────────────

def dc_tau(hg, ag, lam_h, lam_a, rho):
    if hg == 0 and ag == 0:
        return 1 - lam_h * lam_a * rho
    elif hg == 0 and ag == 1:
        return 1 + lam_h * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_a * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0

def dc_log_likelihood(params, teams, home_teams, away_teams, home_goals, away_goals, weights):
    n = len(teams)
    attack   = dict(zip(teams, params[:n]))
    defence  = dict(zip(teams, params[n:2*n]))
    home_adv = params[-1]
    log_lik  = 0.0
    for i in range(len(home_teams)):
        ht, at = home_teams[i], away_teams[i]
        hg, ag = home_goals[i], away_goals[i]
        w = weights[i]
        if ht not in attack or at not in attack:
            continue
        lam_h = np.exp(home_adv + attack[ht] + defence[at])
        lam_a = np.exp(attack[at] + defence[ht])
        tau = dc_tau(hg, ag, lam_h, lam_a, DC_RHO)
        if tau <= 0:
            continue
        log_lik += w * (
            np.log(tau) +
            hg * np.log(lam_h) - lam_h - math.lgamma(hg + 1) +
            ag * np.log(lam_a) - lam_a - math.lgamma(ag + 1)
        )
    return -log_lik

def fit_model(df):
    teams = sorted(set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))
    n = len(teams)
    now = pd.Timestamp.now()
    days_ago = (now - df["Date"]).dt.days.values
    weights = np.exp(-TIME_DECAY * days_ago)
    x0 = np.concatenate([np.ones(n), -np.ones(n), [0.3]])
    bounds = [(0.1, 4.0)] * n + [(-3.0, 0.0)] * n + [(0.0, 1.0)]
    constraints = [{"type": "eq", "fun": lambda x: x[0] - 1.0}]
    result = minimize(
        dc_log_likelihood, x0,
        args=(teams, df["HomeTeam"].tolist(), df["AwayTeam"].tolist(),
              df["FTHG"].tolist(), df["FTAG"].tolist(), weights),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return {
        "attack":         dict(zip(teams, result.x[:n])),
        "defence":        dict(zip(teams, result.x[n:2*n])),
        "home_advantage": float(result.x[-1]),
        "teams":          teams,
    }

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def poisson_prob(lam, k):
    try:
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    except:
        return 0.0

def predict_match(home_team, away_team, model, home_form_mult=1.0, away_form_mult=1.0):
    attack, defence, home_adv = model["attack"], model["defence"], model["home_advantage"]
    if home_team not in attack or away_team not in attack:
        return None
    lam_home = np.exp(home_adv + attack[home_team] + defence[away_team])
    lam_away = np.exp(attack[away_team] + defence[home_team])
    lam_home *= (1 + FORM_WEIGHT * (home_form_mult - 1))
    lam_away *= (1 + FORM_WEIGHT * (away_form_mult - 1))
    lam_home, lam_away = max(lam_home, 0.1), max(lam_away, 0.1)

    prob_matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for hg, ag in product(range(MAX_GOALS + 1), range(MAX_GOALS + 1)):
        p = poisson_prob(lam_home, hg) * poisson_prob(lam_away, ag)
        tau = dc_tau(hg, ag, lam_home, lam_away, DC_RHO)
        prob_matrix[hg][ag] = max(p * tau, 0.0)

    total = prob_matrix.sum()
    if total > 0:
        prob_matrix /= total

    p_home = float(np.sum(np.tril(prob_matrix, -1)))
    p_draw  = float(np.sum(np.diag(prob_matrix)))
    p_away  = float(np.sum(np.triu(prob_matrix, 1)))

    if p_draw >= DRAW_MIN_PROB:
        result = "D"
    elif p_home >= p_away:
        result = "H"
    else:
        result = "A"

    best_hg, best_ag = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)

    flat = prob_matrix.flatten()
    top5i = flat.argsort()[-5:][::-1]
    likely_scores = []
    for idx in top5i:
        hg = idx // (MAX_GOALS + 1)
        ag = idx % (MAX_GOALS + 1)
        likely_scores.append({
            "score": f"{hg}-{ag}",
            "probability": round(float(prob_matrix[hg][ag]) * 100, 1),
        })

    return {
        "predicted_result":       result,
        "predicted_result_label": {"H": "Home Win", "D": "Draw", "A": "Away Win"}[result],
        "prob_home_win":          round(p_home * 100, 1),
        "prob_draw":              round(p_draw * 100, 1),
        "prob_away_win":          round(p_away * 100, 1),
        "expected_home_goals":    round(float(lam_home), 2),
        "expected_away_goals":    round(float(lam_away), 2),
        "expected_total_goals":   round(float(lam_home + lam_away), 2),
        "predicted_score":        f"{best_hg}-{best_ag}",
        "likely_scores":          likely_scores,
        "over_25":                bool(lam_home + lam_away > 2.5),
    }

def get_form(df, team, n=6):
    home = df[df["HomeTeam"] == team][["Date", "FTR"]].copy()
    home["Result"] = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
    away = df[df["AwayTeam"] == team][["Date", "FTR"]].copy()
    away["Result"] = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
    games = pd.concat([home[["Date", "Result"]], away[["Date", "Result"]]])
    games = games.sort_values("Date", ascending=False).head(n)
    if len(games) == 0:
        return {"multiplier": 1.0, "form_string": "", "recent_points": 0}
    dw = np.exp(-0.25 * np.arange(len(games)))
    pts = games["Result"].map({"W": 3, "D": 1, "L": 0}).values
    weighted = np.average(pts, weights=dw)
    multiplier = round(0.90 + (weighted / 3.0) * 0.20, 4)
    return {
        "multiplier":    float(multiplier),
        "form_string":   "".join(games["Result"].tolist()),
        "recent_points": int(games["Result"].map({"W": 3, "D": 1, "L": 0}).sum()),
    }

def get_cards(df, home_team, away_team):
    hg = df[df["HomeTeam"] == home_team]
    ag = df[df["AwayTeam"] == away_team]
    league_hy = float(df["HY"].mean()) if "HY" in df.columns else 2.0
    league_ay = float(df["AY"].mean()) if "AY" in df.columns else 2.0
    home_y = float(hg["HY"].mean()) if len(hg) > 3 else league_hy
    away_y = float(ag["AY"].mean()) if len(ag) > 3 else league_ay
    return {
        "pred_home_yellows":  round(home_y, 1),
        "pred_away_yellows":  round(away_y, 1),
        "pred_total_yellows": round(home_y + away_y, 1),
    }

def get_h2h(df, home_team, away_team, n=5):
    mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
        ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df[mask].sort_values("Date", ascending=False).head(n)
    results = []
    for _, row in h2h.iterrows():
        results.append({
            "date":      row["Date"].strftime("%Y-%m-%d"),
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "score":     f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "result":    row["FTR"],
        })
    return results

def build_fixture_response(matches, gameweek):
    df    = get_df()
    model = get_model()
    fixtures = []
    for m in matches:
        home_raw  = m["homeTeam"]["name"]
        away_raw  = m["awayTeam"]["name"]
        home_team = normalise_team(home_raw)
        away_team = normalise_team(away_raw)

        home_form = get_form(df, home_team)
        away_form = get_form(df, away_team)
        pred      = predict_match(home_team, away_team, model, home_form["multiplier"], away_form["multiplier"])
        cards     = get_cards(df, home_team, away_team)
        h2h       = get_h2h(df, home_team, away_team)

        status = m.get("status", "SCHEDULED")
        actual = None
        if status == "FINISHED":
            score = m.get("score", {}).get("fullTime", {})
            hg = score.get("home") or 0
            ag = score.get("away") or 0
            actual = {
                "home_goals": hg,
                "away_goals": ag,
                "result": "H" if hg > ag else "D" if hg == ag else "A",
            }

        fixtures.append({
            "fixture_id":   m.get("id"),
            "match_date":   m.get("utcDate", "")[:10],
            "match_time":   m.get("utcDate", "")[11:16],
            "status":       status,
            "home_team":    home_team,
            "away_team":    away_team,
            "prediction":   pred,
            "cards":        cards,
            "home_form":    home_form,
            "away_form":    away_form,
            "head_to_head": h2h,
            "actual":       actual,
        })

    return jsonify({
        "gameweek":      gameweek,
        "season":        "2024-25",
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "model_version": "v5",
        "model_accuracy": {
            "overall":    53.3,
            "home_win":   53.9,
            "away_win":   52.7,
            "over_under": 58.0,
        },
        "fixtures": fixtures,
    })

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "DD Predictor v5", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route("/api/predictions/<int:gameweek>", methods=["GET", "POST"])
def predictions(gameweek):
    """
    POST: frontend sends fixtures JSON (used when server IP is not allowlisted)
    GET:  server fetches fixtures directly from football-data.org
    """
    try:
        matches = []

        if request.method == "POST":
            body = request.get_json(force=True) or {}
            matches = body.get("matches", [])

        if not matches:
            url = f"{API_BASE}/competitions/PL/matches"
            for season_year in [2024, 2025]:
                try:
                    resp = requests.get(url, headers=HEADERS,
                                        params={"matchday": gameweek, "season": season_year},
                                        timeout=15)
                    if resp.status_code == 200:
                        matches = resp.json().get("matches", [])
                        if matches:
                            break
                except Exception:
                    pass

        if not matches:
            return jsonify({"error": f"No fixtures found for gameweek {gameweek}."}), 404

        return build_fixture_response(matches, gameweek)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team/<team_name>")
def team_stats(team_name):
    try:
        df = get_df()
        home = df[df["HomeTeam"] == team_name]
        away = df[df["AwayTeam"] == team_name]
        all_games = len(home) + len(away)

        if all_games == 0:
            return jsonify({"error": f"Team '{team_name}' not found"}), 404

        h = home[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HY", "AY"]].copy()
        h["TeamGoals"] = h["FTHG"]; h["OppGoals"] = h["FTAG"]
        h["Opponent"] = h["AwayTeam"]; h["Venue"] = "Home"
        h["Result"] = h["FTR"].map({"H": "W", "D": "D", "A": "L"})
        h["Yellows"] = h["HY"]

        a = away[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HY", "AY"]].copy()
        a["TeamGoals"] = a["FTAG"]; a["OppGoals"] = a["FTHG"]
        a["Opponent"] = a["HomeTeam"]; a["Venue"] = "Away"
        a["Result"] = a["FTR"].map({"A": "W", "D": "D", "H": "L"})
        a["Yellows"] = a["AY"]

        recent = pd.concat([h, a]).sort_values("Date", ascending=False).head(10)
        last_10 = []
        for _, row in recent.iterrows():
            last_10.append({
                "date":     row["Date"].strftime("%Y-%m-%d"),
                "opponent": row["Opponent"],
                "venue":    row["Venue"],
                "score":    f"{int(row['TeamGoals'])}-{int(row['OppGoals'])}",
                "result":   row["Result"],
                "yellows":  int(row["Yellows"]) if not pd.isna(row["Yellows"]) else 0,
            })

        seasons = []
        for season in sorted(df["Season"].unique()):
            sh = df[(df["HomeTeam"] == team_name) & (df["Season"] == season)]
            sa = df[(df["AwayTeam"] == team_name) & (df["Season"] == season)]
            sg = len(sh) + len(sa)
            if sg == 0:
                continue
            sw = int((sh["FTR"] == "H").sum() + (sa["FTR"] == "A").sum())
            sd = int((sh["FTR"] == "D").sum() + (sa["FTR"] == "D").sum())
            sl = int((sh["FTR"] == "A").sum() + (sa["FTR"] == "H").sum())
            gf = int(sh["FTHG"].sum() + sa["FTAG"].sum())
            ga = int(sh["FTAG"].sum() + sa["FTHG"].sum())
            seasons.append({
                "season": season, "played": sg,
                "won": sw, "drawn": sd, "lost": sl,
                "gf": gf, "ga": ga, "gd": gf - ga,
                "points": sw * 3 + sd,
            })

        wins   = int((home["FTR"] == "H").sum() + (away["FTR"] == "A").sum())
        draws  = int((home["FTR"] == "D").sum() + (away["FTR"] == "D").sum())
        losses = int((home["FTR"] == "A").sum() + (away["FTR"] == "H").sum())
        gf     = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga     = int(home["FTAG"].sum() + away["FTHG"].sum())
        hy     = float(home["HY"].mean()) if "HY" in df.columns else 0
        ay     = float(away["AY"].mean()) if "AY" in df.columns else 0

        return jsonify({
            "team": team_name,
            "summary": {
                "games": all_games, "wins": wins, "draws": draws, "losses": losses,
                "goals_for": gf, "goals_against": ga, "goal_diff": gf - ga,
                "win_rate":           round(wins / all_games * 100, 1),
                "avg_goals_scored":   round(gf / all_games, 2),
                "avg_goals_conceded": round(ga / all_games, 2),
                "avg_home_yellows":   round(hy, 2),
                "avg_away_yellows":   round(ay, 2),
            },
            "last_10": last_10,
            "seasons": seasons,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/table")
def form_table():
    try:
        df = get_df()
        current_season = df["Season"].max()
        s = df[df["Season"] == current_season]
        teams = sorted(set(s["HomeTeam"].tolist() + s["AwayTeam"].tolist()))
        rows = []
        for team in teams:
            h = s[s["HomeTeam"] == team]
            a = s[s["AwayTeam"] == team]
            g = len(h) + len(a)
            if g == 0: continue
            w  = int((h["FTR"] == "H").sum() + (a["FTR"] == "A").sum())
            d  = int((h["FTR"] == "D").sum() + (a["FTR"] == "D").sum())
            l  = int((h["FTR"] == "A").sum() + (a["FTR"] == "H").sum())
            gf = int(h["FTHG"].sum() + a["FTAG"].sum())
            ga = int(h["FTAG"].sum() + a["FTHG"].sum())
            form = get_form(df, team, 6)
            rows.append({
                "team": team, "played": g, "won": w, "drawn": d, "lost": l,
                "gf": gf, "ga": ga, "gd": gf - ga, "points": w * 3 + d,
                "form": form["form_string"],
            })
        rows.sort(key=lambda x: (-x["points"], -(x["gd"]), -x["gf"]))
        for i, r in enumerate(rows):
            r["position"] = i + 1
        return jsonify({"season": current_season, "table": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
