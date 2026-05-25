"""
Esports Elo Edge Engine — Web API
FastAPI backend for Vercel. Auto-refreshes Elo every 60 minutes.
"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from engine import (
    EloEngine, load_or_fetch, fetch_lol_upcoming,
    fetch_lol_history, build_engine_from_matches, save_cache,
)

app = FastAPI(title="Esports Elo Edge Engine")

_engine = None
_engine_info = {}

def get_engine():
    global _engine, _engine_info
    if _engine is None:
        _engine = load_or_fetch("lol", force=False)
        _engine_info = {
            "teams": len(_engine.teams),
            "matches": _engine.matches_processed,
            "game": "lol",
            "last_refresh": datetime.now(timezone.utc).isoformat(),
        }
    return _engine


@app.get("/")
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Esports Elo Edge Engine</h1>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/refresh")
async def refresh():
    """Called by Vercel cron every 60 minutes to update Elo with new results."""
    global _engine, _engine_info
    try:
        matches = fetch_lol_history(pages=5)  # last ~400 matches (recent results only)
        if matches and _engine:
            # Ingest new completed matches into existing engine
            existing_dates = set()
            for tid, t in _engine.teams.items():
                if t.last_match_date:
                    existing_dates.add(t.last_match_date)

            new_count = 0
            for m in sorted(matches, key=lambda x: x.get("date", "")):
                if m.get("date", "") > max(existing_dates) if existing_dates else True:
                    _engine.update(
                        winner_id=str(m["winner_id"]), loser_id=str(m["loser_id"]),
                        w_score=m.get("w_score", 1), l_score=m.get("l_score", 0),
                        bo=m.get("bo", 3), league=m.get("league", ""),
                        match_date=m.get("date", ""),
                        winner_name=m.get("winner_name", ""),
                        loser_name=m.get("loser_name", ""),
                    )
                    new_count += 1

            _engine_info["last_refresh"] = datetime.now(timezone.utc).isoformat()
            _engine_info["teams"] = len(_engine.teams)
            _engine_info["matches"] = _engine.matches_processed
            return {"status": "refreshed", "new_matches": new_count, **_engine_info}
        return {"status": "no_new_data"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/predict")
async def predict(
    team_a: str = Query(...), team_b: str = Query(...),
    bo: int = Query(3), market_price_a: float = Query(None),
    bankroll: float = Query(420.0),
):
    engine = get_engine()
    a_id = engine.find_team(team_a)
    b_id = engine.find_team(team_b)
    if not a_id:
        return JSONResponse({"error": f"Team not found: {team_a}. Try /api/search?q={team_a}"}, 404)
    if not b_id:
        return JSONResponse({"error": f"Team not found: {team_b}. Try /api/search?q={team_b}"}, 404)

    if market_price_a is not None and 0 < market_price_a < 1:
        return engine.predict_with_market(a_id, b_id, market_price_a, bo, bankroll)
    return engine.predict(a_id, b_id, bo)


@app.get("/api/upcoming")
async def upcoming():
    engine = get_engine()
    matches = fetch_lol_upcoming()
    results = []
    for u in matches:
        a_id = engine.find_team(u["team_a"])
        b_id = engine.find_team(u["team_b"])
        pred = None
        if a_id and b_id:
            pred = engine.predict(a_id, b_id, u["bo"])
        results.append({
            **u,
            "model_a": pred["p_a"] if pred else None,
            "model_b": pred["p_b"] if pred else None,
            "elo_a": pred["elo_a"] if pred else None,
            "elo_b": pred["elo_b"] if pred else None,
        })
    return {"matches": results, "engine": _engine_info}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=2)):
    engine = get_engine()
    return engine.search_teams(q)


@app.get("/api/rankings")
async def rankings(limit: int = Query(30), min_matches: int = Query(15)):
    engine = get_engine()
    return engine.top_teams(min_matches=min_matches, limit=limit)


@app.get("/api/ingest")
async def ingest(
    winner: str = Query(...), loser: str = Query(...),
    score: str = Query("2-0"), bo: int = Query(3), league: str = Query(""),
):
    engine = get_engine()
    w_id = engine.find_team(winner)
    l_id = engine.find_team(loser)
    if not w_id: return JSONResponse({"error": f"Not found: {winner}"}, 404)
    if not l_id: return JSONResponse({"error": f"Not found: {loser}"}, 404)
    parts = score.split("-")
    engine.update(w_id, l_id, int(parts[0]), int(parts[1]), bo,
                  league, "", date.today().isoformat())
    w = engine.teams[w_id]; l = engine.teams[l_id]
    return {
        "status": "updated",
        "winner": {"name": w.name, "elo": round(w.elo)},
        "loser": {"name": l.name, "elo": round(l.elo)},
    }
