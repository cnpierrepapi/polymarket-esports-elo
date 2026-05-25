"""
ESPORTS ELO EDGE ENGINE
========================
Team Elo rating engine for LoL and CS2 esports.
You enter a matchup + market price, the engine tells you if there's edge.

Data Sources:
    - LoL: Riot LoL Esports API (free, public key, live + historical)
    - CS2: bo3.gg API (free, historical match results)
    Both auto-refresh every 60 minutes.

Academic Foundations:
    [1] Glickman (1999) — Dynamic paired comparison (Elo) for competitive games
    [2] Angelini et al. (2021) — Weighted Elo with margin + tier weighting
    [3] Platt (1999) — Logistic recalibration for overconfident probabilities
    [4] Manski (2006) — Edges >15% = model error, not alpha
    [5] Kelly (1956) — Optimal geometric growth bet sizing

Usage:
    python engine.py                              # Show top 30 LoL teams
    python engine.py --predict "T1" "Gen.G" --bo 5
    python engine.py --predict "T1" "Gen.G" --bo 5 --market 0.55
    python engine.py --search "G2"
    python engine.py --upcoming                   # Show upcoming matches
    python engine.py --game cs2                   # CS2 mode
    python engine.py --refresh                    # Force data refresh
"""

import os
import sys
import math
import json
import argparse
import time
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import requests

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32" and not getattr(sys, '_utf8_wrapped', False):
    try:
        import io as _io
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.buffer.closed:
            sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, 'buffer') and not sys.stderr.buffer.closed:
            sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        sys._utf8_wrapped = True
    except (ValueError, AttributeError):
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console(force_terminal=True)
    RICH = True
except ImportError:
    console = None
    RICH = False

# =============================================================================
# CONSTANTS
# =============================================================================

RIOT_API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"  # public
RIOT_BASE = "https://esports-api.lolesports.com/persisted/gw"
BO3GG_BASE = "https://api.bo3.gg/api/v1"
DATA_DIR = Path(__file__).parent / "data"

INITIAL_ELO = 1000.0
K_BASE = 32
PLATT_A = 0.80  # logit shrinkage factor

# League tier weights
LEAGUE_TIERS = {
    # Tier S — International
    "worlds": 2.5, "msi": 2.5, "ewc": 2.0,
    # Tier A — Major regional leagues
    "lck": 2.0, "lpl": 2.0, "lec": 1.8, "lcs": 1.5, "lco": 1.2,
    "pcs": 1.3, "vcs": 1.2, "cblol": 1.3, "lla": 1.2,
    # Tier B — Secondary leagues
    "nacl": 0.8, "lck challengers": 0.8, "ljl": 0.8, "nlc": 0.7,
    "lrn": 0.6, "liga portuguesa": 0.6, "superliga": 0.7,
    # CS2 tiers
    "s": 2.0, "a": 1.5, "b": 1.0, "c": 0.6, "d": 0.3,
}

def get_tier_weight(league_name: str, tier: str = "") -> float:
    """Get tier weight from league name or explicit tier."""
    if tier:
        return LEAGUE_TIERS.get(tier, 1.0)
    ln = league_name.lower()
    for key, weight in LEAGUE_TIERS.items():
        if key in ln:
            return weight
    return 1.0


# =============================================================================
# ELO ENGINE (shared between LoL and CS2)
# =============================================================================

@dataclass
class TeamRating:
    name: str
    team_id: str
    elo: float = INITIAL_ELO
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    last_match_date: str = ""
    league: str = ""
    recent_results: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.matches_played if self.matches_played > 0 else 0.5

    @property
    def form(self) -> str:
        return "".join("W" if r[1] else "L" for r in self.recent_results[-5:])


class EloEngine:
    def __init__(self, game: str = "lol"):
        self.game = game
        self.teams: dict[str, TeamRating] = {}
        self.name_index: dict[str, str] = {}
        self.last_refresh: Optional[datetime] = None
        self.matches_processed: int = 0

    def get_team(self, team_id: str, name: str = "", league: str = "") -> TeamRating:
        tid = str(team_id)
        if tid not in self.teams:
            self.teams[tid] = TeamRating(name=name or f"Team_{tid}", team_id=tid, league=league)
        t = self.teams[tid]
        if name and t.name.startswith("Team_"):
            t.name = name
        if name:
            self.name_index[name.lower()] = tid
        if league:
            t.league = league
        return t

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def margin_factor(self, w_score: int, l_score: int, bo: int) -> float:
        if bo <= 1:
            return 1.0
        margin = w_score - l_score
        max_margin = max(bo // 2, 1)
        return 1.0 + 0.4 * (margin / max_margin)

    def update(self, winner_id: str, loser_id: str,
               w_score: int = 1, l_score: int = 0,
               bo: int = 3, league: str = "", tier: str = "",
               match_date: str = "",
               winner_name: str = "", loser_name: str = ""):
        w = self.get_team(str(winner_id), winner_name, league)
        l = self.get_team(str(loser_id), loser_name, league)

        e_w = self.expected_score(w.elo, l.elo)
        tw = get_tier_weight(league, tier)
        mf = self.margin_factor(w_score, l_score, bo)
        k = K_BASE * tw * mf

        w.elo += k * (1 - e_w)
        l.elo += k * (0 - (1 - e_w))
        w.matches_played += 1; l.matches_played += 1
        w.wins += 1; l.losses += 1
        w.last_match_date = match_date; l.last_match_date = match_date
        w.recent_results.append((match_date, True, l.elo))
        l.recent_results.append((match_date, False, w.elo))
        w.recent_results = w.recent_results[-10:]
        l.recent_results = l.recent_results[-10:]
        self.matches_processed += 1

    def predict(self, team_a_id: str, team_b_id: str, bo: int = 3) -> dict:
        a = self.teams.get(str(team_a_id))
        b = self.teams.get(str(team_b_id))
        if not a or not b:
            return {"error": "Team not found"}

        p_raw = self.expected_score(a.elo, b.elo)

        # Platt calibration
        if 0.001 < p_raw < 0.999:
            logit = math.log(p_raw / (1 - p_raw))
            p_cal = 1.0 / (1.0 + math.exp(-PLATT_A * logit))
        else:
            p_cal = p_raw

        # BO correction (training data is ~BO3 heavy)
        bo_corr = 0.0
        p_match = p_cal
        if bo == 1:
            p_map = self._invert_bo3(p_cal)
            p_match = p_map
            bo_corr = p_match - p_cal
        elif bo == 5:
            p_map = self._invert_bo3(p_cal)
            p_match = p_map**3 * (10 - 15*p_map + 6*p_map**2)
            bo_corr = p_match - p_cal

        return {
            "team_a": a.name, "team_b": b.name,
            "team_a_id": a.team_id, "team_b_id": b.team_id,
            "p_a": round(p_match, 4), "p_b": round(1 - p_match, 4),
            "p_raw": round(p_raw, 4), "p_calibrated": round(p_cal, 4),
            "bo_correction": round(bo_corr, 4), "bo_type": bo,
            "elo_a": round(a.elo, 1), "elo_b": round(b.elo, 1),
            "matches_a": a.matches_played, "matches_b": b.matches_played,
            "form_a": a.form, "form_b": b.form,
            "win_rate_a": round(a.win_rate, 3), "win_rate_b": round(b.win_rate, 3),
            "league_a": a.league, "league_b": b.league,
            "last_match_a": a.last_match_date, "last_match_b": b.last_match_date,
        }

    def predict_with_market(self, team_a_id: str, team_b_id: str,
                            market_price_a: float, bo: int = 3,
                            bankroll: float = 420.0) -> dict:
        pred = self.predict(team_a_id, team_b_id, bo)
        if "error" in pred:
            return pred

        market_b = round(1 - market_price_a, 3)
        edge_a = pred["p_a"] - market_price_a
        edge_b = pred["p_b"] - market_b
        kelly_a = max(0, self._kelly(pred["p_a"], market_price_a))
        kelly_b = max(0, self._kelly(pred["p_b"], market_b))
        manski = max(abs(edge_a), abs(edge_b)) > 0.15

        if manski:
            signal = "MANSKI"
            reasoning = f"Edge {max(abs(edge_a), abs(edge_b)):.1%} > 15% -- likely model error [Manski 2006]"
        elif edge_a >= 0.04 and edge_a > edge_b:
            signal = f"BUY {pred['team_a']}"
            reasoning = f"Model {pred['p_a']:.1%} > Market {market_price_a:.1%}, edge {edge_a:+.1%}"
        elif edge_b >= 0.04:
            signal = f"BUY {pred['team_b']}"
            reasoning = f"Model {pred['p_b']:.1%} > Market {market_b:.1%}, edge {edge_b:+.1%}"
        else:
            signal = "PASS"
            reasoning = f"Max edge {max(edge_a, edge_b):.1%} < 4% threshold"

        pred["market"] = {
            "price_a": market_price_a, "price_b": market_b,
            "edge_a": round(edge_a, 4), "edge_b": round(edge_b, 4),
            "kelly_a": round(kelly_a, 4), "kelly_b": round(kelly_b, 4),
            "size_a": round(bankroll * min(kelly_a, 0.10), 2),
            "size_b": round(bankroll * min(kelly_b, 0.10), 2),
            "signal": signal, "manski_flag": manski, "reasoning": reasoning,
        }
        return pred

    def _kelly(self, p: float, c: float) -> float:
        if c <= 0 or c >= 1 or p <= 0: return 0.0
        return max(0, (p - (1-p) * c / (1-c)) * 0.25)

    def _invert_bo3(self, p_match: float) -> float:
        if p_match <= 0.01: return 0.01
        if p_match >= 0.99: return 0.99
        p = p_match
        for _ in range(50):
            f = p**2 * (3 - 2*p) - p_match
            fp = 6*p - 6*p**2
            if abs(fp) < 1e-12: break
            p_new = max(0.01, min(0.99, p - f / fp))
            if abs(p_new - p) < 1e-8: break
            p = p_new
        return p

    def find_team(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        if q in self.name_index:
            return self.name_index[q]
        candidates = [(tid, self.teams[tid].matches_played)
                       for name, tid in self.name_index.items()
                       if q in name or name in q]
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        return None

    def search_teams(self, query: str, limit: int = 20) -> list[dict]:
        q = query.lower().strip()
        results = []
        for tid, t in self.teams.items():
            if q in t.name.lower():
                results.append({
                    "id": tid, "name": t.name, "elo": round(t.elo),
                    "matches": t.matches_played, "win_rate": round(t.win_rate, 3),
                    "form": t.form, "league": t.league,
                    "last_match": t.last_match_date,
                })
        results.sort(key=lambda x: x["elo"], reverse=True)
        return results[:limit]

    def top_teams(self, min_matches: int = 15, limit: int = 30) -> list[dict]:
        teams = [(tid, t) for tid, t in self.teams.items() if t.matches_played >= min_matches]
        teams.sort(key=lambda x: x[1].elo, reverse=True)
        return [{
            "rank": i+1, "id": t.team_id, "name": t.name,
            "elo": round(t.elo), "matches": t.matches_played,
            "wins": t.wins, "losses": t.losses,
            "win_rate": round(t.win_rate, 3), "form": t.form,
            "league": t.league, "last_match": t.last_match_date,
        } for i, (_, t) in enumerate(teams[:limit])]


# =============================================================================
# LOL DATA: Riot Esports API
# =============================================================================

def fetch_lol_history(pages: int = 40) -> list[dict]:
    """
    Fetch LoL match history from Riot Esports API.
    Paginates backward from current schedule. 80 events per page.
    40 pages = ~3200 matches = ~2-3 months of all leagues.
    """
    headers = {"x-api-key": RIOT_API_KEY}
    all_matches = []
    page_token = None

    for i in range(pages):
        params = {"hl": "en-US"}
        if page_token:
            params["pageToken"] = page_token

        try:
            r = requests.get(f"{RIOT_BASE}/getSchedule", params=params,
                             headers=headers, timeout=15)
            if r.status_code != 200:
                break

            data = r.json().get("data", {}).get("schedule", {})
            events = data.get("events", [])

            for e in events:
                if e.get("state") != "completed":
                    continue
                match = e.get("match", {})
                if not match:
                    continue

                teams = match.get("teams", [])
                if len(teams) < 2:
                    continue

                t1 = teams[0]
                t2 = teams[1]
                t1_wins = t1.get("result", {}).get("gameWins", 0)
                t2_wins = t2.get("result", {}).get("gameWins", 0)

                if t1_wins == 0 and t2_wins == 0:
                    continue

                winner_idx = 0 if t1_wins > t2_wins else 1
                bo = match.get("strategy", {}).get("count", 1)
                league = e.get("league", {}).get("name", "")

                all_matches.append({
                    "winner_id": teams[winner_idx].get("code", teams[winner_idx].get("name", "")),
                    "loser_id": teams[1-winner_idx].get("code", teams[1-winner_idx].get("name", "")),
                    "winner_name": teams[winner_idx].get("name", ""),
                    "loser_name": teams[1-winner_idx].get("name", ""),
                    "w_score": max(t1_wins, t2_wins),
                    "l_score": min(t1_wins, t2_wins),
                    "bo": bo,
                    "league": league,
                    "date": e.get("startTime", "")[:10],
                })

            # Get older page token
            page_token = data.get("pages", {}).get("older", "")
            if not page_token:
                break

            if i % 10 == 0:
                d = all_matches[-1]["date"] if all_matches else "?"
                print(f"  Page {i}: {len(all_matches)} matches (oldest: {d})")

        except Exception as ex:
            print(f"  [WARN] Page {i} failed: {ex}")
            break

    print(f"  Total LoL matches: {len(all_matches)}")
    return all_matches


def fetch_lol_upcoming() -> list[dict]:
    """Fetch upcoming/live LoL matches."""
    headers = {"x-api-key": RIOT_API_KEY}
    try:
        r = requests.get(f"{RIOT_BASE}/getSchedule", params={"hl": "en-US"},
                         headers=headers, timeout=15)
        if r.status_code != 200:
            return []

        events = r.json().get("data", {}).get("schedule", {}).get("events", [])
        upcoming = []
        for e in events:
            if e.get("state") not in ("unstarted", "inProgress"):
                continue
            match = e.get("match")
            if not match:
                continue
            teams = match.get("teams", [])
            if len(teams) < 2:
                continue

            strategy = match.get("strategy") or {}
            bo = strategy.get("count", 1)
            league = e.get("league", {}).get("name", "")

            t1_result = teams[0].get("result") or {}
            t2_result = teams[1].get("result") or {}
            upcoming.append({
                "team_a": teams[0].get("name", "?"),
                "team_b": teams[1].get("name", "?"),
                "team_a_code": teams[0].get("code", ""),
                "team_b_code": teams[1].get("code", ""),
                "bo": bo,
                "league": league,
                "start": e.get("startTime", "")[:16],
                "state": e.get("state", ""),
                "score_a": t1_result.get("gameWins", 0),
                "score_b": t2_result.get("gameWins", 0),
            })

        return upcoming
    except Exception as ex:
        import traceback
        print(f"[WARN] Upcoming fetch failed: {ex}")
        traceback.print_exc()
        return upcoming if upcoming else []


# =============================================================================
# CS2 DATA: bo3.gg API
# =============================================================================

def fetch_cs2_history(start_offset: int = 60000, batches: int = 150) -> list[dict]:
    """Fetch CS2 match history from bo3.gg."""
    all_matches = []
    offset = start_offset
    team_ids_needed = set()

    for i in range(batches):
        try:
            r = requests.get(f"{BO3GG_BASE}/matches", params={
                "page[limit]": 100, "page[offset]": offset,
            }, timeout=15)
            if r.status_code != 200:
                break

            results = r.json().get("results", [])
            if not results:
                break

            for m in results:
                if m.get("team1_id") and m.get("winner_team_id") and m.get("team1_score") is not None:
                    w_id = m["winner_team_id"]
                    l_id = m["loser_team_id"]
                    w_score = m["team1_score"] if m["team1_id"] == w_id else m["team2_score"]
                    l_score = m["team2_score"] if m["team1_id"] == w_id else m["team1_score"]
                    all_matches.append({
                        "winner_id": str(w_id), "loser_id": str(l_id),
                        "winner_name": "", "loser_name": "",
                        "w_score": w_score, "l_score": l_score,
                        "bo": m.get("bo_type", 3),
                        "tier": m.get("tier", "c"),
                        "league": "", "date": (m.get("start_date") or "")[:10],
                    })
                    team_ids_needed.add(w_id)
                    team_ids_needed.add(l_id)

            offset += 100
            if i % 20 == 0 and all_matches:
                print(f"  Batch {i}: {len(all_matches)} CS2 matches (latest: {all_matches[-1]['date']})")
            if i % 10 == 0 and i > 0:
                time.sleep(0.3)
        except Exception as e:
            if i > 5:
                break

    # Fetch team names
    print(f"  Fetching names for {len(team_ids_needed)} CS2 teams...")
    names = {}
    for tid in team_ids_needed:
        try:
            r = requests.get(f"{BO3GG_BASE}/teams/{tid}", timeout=5)
            if r.status_code == 200:
                names[tid] = r.json().get("name", f"Team_{tid}")
        except:
            pass
        if len(names) % 200 == 0 and len(names) > 0:
            print(f"    {len(names)}/{len(team_ids_needed)} names fetched")
            time.sleep(0.3)

    for m in all_matches:
        m["winner_name"] = names.get(int(m["winner_id"]), m["winner_id"])
        m["loser_name"] = names.get(int(m["loser_id"]), m["loser_id"])

    print(f"  Total CS2 matches: {len(all_matches)}")
    return all_matches


# =============================================================================
# BUILD ENGINE
# =============================================================================

def build_engine_from_matches(matches: list[dict], game: str = "lol") -> EloEngine:
    engine = EloEngine(game=game)
    matches.sort(key=lambda m: m.get("date", ""))
    for m in matches:
        engine.update(
            winner_id=str(m["winner_id"]), loser_id=str(m["loser_id"]),
            w_score=m.get("w_score", 1), l_score=m.get("l_score", 0),
            bo=m.get("bo", 3), league=m.get("league", ""),
            tier=m.get("tier", ""), match_date=m.get("date", ""),
            winner_name=m.get("winner_name", ""),
            loser_name=m.get("loser_name", ""),
        )
    engine.last_refresh = datetime.now()
    return engine


def save_cache(matches: list[dict], game: str):
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / f"{game}_matches.json", "w") as f:
        json.dump(matches, f)


def load_cache(game: str) -> list[dict]:
    p = DATA_DIR / f"{game}_matches.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def load_or_fetch(game: str = "lol", force: bool = False) -> EloEngine:
    matches = [] if force else load_cache(game)

    if not matches:
        if game == "lol":
            print("[INFO] Fetching LoL match history from Riot API...")
            matches = fetch_lol_history(pages=40)
        else:
            print("[INFO] Fetching CS2 match history from bo3.gg...")
            matches = fetch_cs2_history()
        if matches:
            save_cache(matches, game)

    print(f"[INFO] Building {game.upper()} Elo from {len(matches)} matches...")
    engine = build_engine_from_matches(matches, game)
    print(f"[INFO] {len(engine.teams)} teams rated")
    return engine


# =============================================================================
# DISPLAY
# =============================================================================

def display_prediction(pred: dict):
    if not RICH:
        print(f"\n{pred['team_a']} vs {pred['team_b']} (BO{pred['bo_type']})")
        print(f"Model: {pred['p_a']:.1%} / {pred['p_b']:.1%} | Elo: {pred['elo_a']} / {pred['elo_b']}")
        if "market" in pred:
            m = pred["market"]
            print(f"Market: {m['price_a']:.1%} / {m['price_b']:.1%} | Edge: {m['edge_a']:+.1%} / {m['edge_b']:+.1%}")
            print(f"Signal: {m['signal']} | {m['reasoning']}")
        return

    console.print()
    bo_note = f"BO correction: {pred['bo_correction']:+.1%}" if pred['bo_correction'] != 0 else "BO3 baseline (no correction)"
    console.print(Panel(
        f"[bold cyan]{pred['team_a']}[/bold cyan] vs [bold cyan]{pred['team_b']}[/bold cyan]\n"
        f"[dim]BO{pred['bo_type']} | Platt Calibrated | {bo_note}[/dim]",
        box=box.DOUBLE))

    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("", style="bold")
    t.add_column(pred["team_a"], justify="center")
    t.add_column(pred["team_b"], justify="center")
    t.add_row("Model Prob", f"[bold cyan]{pred['p_a']:.1%}[/bold cyan]", f"[bold cyan]{pred['p_b']:.1%}[/bold cyan]")
    t.add_row("Elo", str(pred["elo_a"]), str(pred["elo_b"]))
    t.add_row("Matches", str(pred["matches_a"]), str(pred["matches_b"]))
    t.add_row("Win Rate", f"{pred['win_rate_a']:.0%}", f"{pred['win_rate_b']:.0%}")
    t.add_row("Form", pred["form_a"], pred["form_b"])
    t.add_row("League", pred.get("league_a", ""), pred.get("league_b", ""))
    t.add_row("Raw Elo P", f"{pred['p_raw']:.1%}", f"{1-pred['p_raw']:.1%}")
    t.add_row("Calibrated", f"{pred['p_calibrated']:.1%}", f"{1-pred['p_calibrated']:.1%}")
    console.print(t)

    if "market" in pred:
        m = pred["market"]
        if m["signal"].startswith("BUY"):
            console.print(Panel(f"[bold green]{m['signal']}[/bold green]", box=box.HEAVY, style="green"))
        elif m["signal"] == "MANSKI":
            console.print(Panel(f"[bold yellow]MANSKI WARNING[/bold yellow]\n[yellow]{m['reasoning']}[/yellow]", box=box.HEAVY, style="yellow"))
        else:
            console.print(Panel(f"[dim]{m['signal']} -- {m['reasoning']}[/dim]", box=box.ROUNDED))

        et = Table(title="Edge Analysis", box=box.SIMPLE_HEAVY)
        et.add_column("Side", style="bold")
        et.add_column("Model", justify="right")
        et.add_column("Market", justify="right")
        et.add_column("Edge", justify="right")
        et.add_column("Kelly %", justify="right")
        et.add_column("Size $", justify="right")
        for side, name, model_p, mkt_p, edge, kelly, size in [
            ("A", pred["team_a"], pred["p_a"], m["price_a"], m["edge_a"], m["kelly_a"], m["size_a"]),
            ("B", pred["team_b"], pred["p_b"], m["price_b"], m["edge_b"], m["kelly_b"], m["size_b"]),
        ]:
            ec = "green" if edge > 0.03 else "red" if edge < -0.03 else "white"
            et.add_row(name, f"{model_p:.1%}", f"{mkt_p:.1%}",
                       f"[{ec}]{edge:+.1%}[/{ec}]",
                       f"{kelly:.1%}" if kelly > 0 else "-",
                       f"${size:.0f}" if size > 0 else "-")
        console.print(et)
    console.print()


def display_upcoming(upcoming: list[dict], engine: EloEngine):
    if not RICH:
        for u in upcoming:
            print(f"{u['start']} | {u['team_a']} vs {u['team_b']} | {u['league']} BO{u['bo']} | {u['state']}")
        return

    t = Table(title="Upcoming / Live Matches", box=box.HEAVY_HEAD)
    t.add_column("Time", style="dim")
    t.add_column("Match", style="bold")
    t.add_column("League")
    t.add_column("BO")
    t.add_column("State")
    t.add_column("Model", justify="right")
    t.add_column("Elo A", justify="right")
    t.add_column("Elo B", justify="right")

    for u in upcoming:
        a_id = engine.find_team(u["team_a"])
        b_id = engine.find_team(u["team_b"])
        model_str = "-"
        elo_a_str = "-"
        elo_b_str = "-"
        if a_id and b_id:
            pred = engine.predict(a_id, b_id, u["bo"])
            model_str = f"{pred['p_a']:.0%}/{pred['p_b']:.0%}"
            elo_a_str = str(pred["elo_a"])
            elo_b_str = str(pred["elo_b"])

        state_style = "[green]LIVE[/green]" if u["state"] == "inProgress" else "[dim]scheduled[/dim]"
        score = f" ({u['score_a']}-{u['score_b']})" if u["state"] == "inProgress" else ""

        t.add_row(
            u["start"][11:16] if len(u["start"]) > 11 else u["start"],
            f"{u['team_a']} vs {u['team_b']}{score}",
            u["league"], str(u["bo"]), state_style,
            model_str, elo_a_str, elo_b_str,
        )
    console.print(t)


def display_rankings(teams: list[dict]):
    if not RICH:
        for t in teams:
            print(f"{t['rank']:>3}. {t['name']:<25} Elo:{t['elo']:<6} {t['form']}")
        return

    t = Table(title="Team Elo Rankings", box=box.HEAVY_HEAD)
    t.add_column("#", justify="right", style="bold")
    t.add_column("Team", style="cyan")
    t.add_column("Elo", justify="right", style="bold")
    t.add_column("W-L", justify="center")
    t.add_column("Win%", justify="right")
    t.add_column("Form", justify="center")
    t.add_column("League")
    for team in teams:
        wr_c = "green" if team["win_rate"] >= 0.6 else "yellow" if team["win_rate"] >= 0.5 else "red"
        form_c = "".join(f"[green]{c}[/green]" if c == "W" else f"[red]{c}[/red]" for c in team["form"])
        t.add_row(str(team["rank"]), team["name"], str(team["elo"]),
                  f"{team['wins']}-{team['losses']}", f"[{wr_c}]{team['win_rate']:.0%}[/{wr_c}]",
                  form_c, team.get("league", ""))
    console.print(t)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Esports Elo Edge Engine")
    parser.add_argument("--game", type=str, default="lol", choices=["lol", "cs2"])
    parser.add_argument("--predict", nargs=2, metavar=("TEAM_A", "TEAM_B"))
    parser.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    parser.add_argument("--market", type=float, default=None)
    parser.add_argument("--bankroll", type=float, default=420.0)
    parser.add_argument("--search", type=str)
    parser.add_argument("--upcoming", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--min-matches", type=int, default=15)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--ingest", nargs=2, metavar=("WINNER", "LOSER"))
    parser.add_argument("--score", type=str, default="2-0")
    parser.add_argument("--league", type=str, default="")
    args = parser.parse_args()

    engine = load_or_fetch(args.game, force=args.refresh)

    if args.search:
        results = engine.search_teams(args.search)
        if not results:
            print(f"No teams found matching '{args.search}'")
        else:
            for r in results:
                print(f"  {r['name']:<30} Elo:{r['elo']:<6} M:{r['matches']:<4} {r['form']} | {r['league']}")
        return

    if args.upcoming and args.game == "lol":
        print("[INFO] Fetching upcoming LoL matches...")
        upcoming = fetch_lol_upcoming()
        display_upcoming(upcoming, engine)
        return

    if args.ingest:
        w_name, l_name = args.ingest
        w_id = engine.find_team(w_name)
        l_id = engine.find_team(l_name)
        if not w_id: print(f"[ERROR] Not found: {w_name}"); return
        if not l_id: print(f"[ERROR] Not found: {l_name}"); return
        parts = args.score.split("-")
        engine.update(w_id, l_id, int(parts[0]), int(parts[1]), args.bo,
                      args.league, "", date.today().isoformat())
        w = engine.teams[w_id]; l = engine.teams[l_id]
        print(f"Updated: {w.name} ({w.elo:.0f}) beat {l.name} ({l.elo:.0f})")
        return

    if args.predict:
        a_name, b_name = args.predict
        a_id = engine.find_team(a_name)
        b_id = engine.find_team(b_name)
        if not a_id: print(f"[ERROR] Not found: {a_name}. Try --search \"{a_name}\""); return
        if not b_id: print(f"[ERROR] Not found: {b_name}. Try --search \"{b_name}\""); return
        if args.market is not None:
            pred = engine.predict_with_market(a_id, b_id, args.market, args.bo, args.bankroll)
        else:
            pred = engine.predict(a_id, b_id, args.bo)
        display_prediction(pred)
        return

    rankings = engine.top_teams(min_matches=args.min_matches, limit=args.top)
    display_rankings(rankings)


if __name__ == "__main__":
    main()
