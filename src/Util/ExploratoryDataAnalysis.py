import os
import json
import time
import re
import urllib.parse
from collections import deque, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import hashlib

import pandas as pd
import matplotlib.pyplot as plt

import requests
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

PLAYER_TAG_RE = re.compile(r"^#[A-Z0-9]+$")

# URL and key to get game data
API_BASE = "https://api.clashroyale.com/v1"
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjFjODhiNjk4LWUwY2ItNGJjNy04ZDk5LWFiNDU0ZWVkNDE5NCIsImlhdCI6MTc3MDg0Mjc2Mywic3ViIjoiZGV2ZWxvcGVyL2VlMjY0OTJmLTBiNDMtMmQyZC1iMDA5LWU0OGFhNDY0Yjk3MiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyI1MC4zNC43OC4xMDkiXSwidHlwZSI6ImNsaWVudCJ9XX0.ZdZ4ysmxSP5vg-j6_hFM81zSDTHZn6VsJRnbrSjxSPh8tg5V77vmh4xsaC2jZ42xZ_vXfcoNwaEHjDF6uLxQBg"

# Only get games from allowed game modes, ladder and ranked
ALLOWED_GAMEMODE_IDS = {72000006, 72000450, 54000137}

def normalize_tag(tag: str) -> str:
    """
    Ensure each player tag is in all caps and preceeded with '#'
    """
    tag = tag.strip().upper()
    if not tag.startswith("#"):
        tag = "#" + tag
    return tag

def fetch_all_cards():
    """
    Fetch all Clash Royale cards and return a cleaned pandas DataFrame.

    Data preprocessing:
      - merges 'items' + 'supportItems'
      - removes 'iconUrls'
      - adds 'is_support' flag
    """
    url = "https://api.clashroyale.com/v1/cards"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Error fetching cards:", response.status_code, response.text)
        return None

    data = response.json()

    # Main cards
    items = data.get("items", [])

    # Support cards
    support_items = data.get("supportItems", [])

    # Add support flag
    for card in items:
        card["is_support"] = False

    for card in support_items:
        card["is_support"] = True

    # Merge both lists
    cards = items + support_items

    # Remove iconUrls
    for card in cards:
        card.pop("iconUrls", None)

    # Build DataFrame
    df = pd.DataFrame(cards)

    # Ensure ID is int (important for your lookup tables)
    df["id"] = df["id"].astype(int)

    return df

def get_battlelog_raw(
    player_tag: str,
    session: Optional[requests.Session] = None,
) -> Any:
    """
    Fetch raw battlelog JSON once.
    If status code is not 200, raise immediately.
    """

    tag = normalize_tag(player_tag)
    encoded_tag = urllib.parse.quote(tag)

    url = f"{API_BASE}/players/{encoded_tag}/battlelog"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    sess = session or requests

    resp = sess.get(url, headers=headers, timeout=30)

    if resp.status_code != 200:
        resp.raise_for_status()

    return resp.json()

def battle_uid(b):
    """
    In order to determine which battles are unique, we need an identifier. This hashes all the unique information about the battle.
    """
    gm_id = (b.get("gameMode") or {}).get("id")
    bt = b.get("battleTime")

    team0 = (b.get("team") or [{}])[0]
    opp0  = (b.get("opponent") or [{}])[0]

    t1 = normalize_tag(team0.get("tag"))
    t2 = normalize_tag(opp0.get("tag"))

    # Canonical ordering so A vs B == B vs A
    a, c = sorted([t1, t2])

    p_c = team0.get("crowns", 0) or 0
    o_c = opp0.get("crowns", 0) or 0
    lo, hi = sorted([p_c, o_c])

    raw = f"{bt}|{gm_id}|{a}|{c}|{lo}-{hi}"
    return hashlib.sha1(raw.encode()).hexdigest()

def filter_unique_battles(
    battlelog: Any,
    seen_battle_ids: set,
    battles_per_player: int,
):
    """
    Iterate battle-by-battle.
    Keep only unseen battles from allowed game modes.
    Stop once we collect battles_per_player unique ones.
    """

    if not isinstance(battlelog, list):
        return []

    unique = []

    for b in battlelog:
        # Filter by allowed game modes first
        gm_id = (b.get("gameMode") or {}).get("id")
        if gm_id not in ALLOWED_GAMEMODE_IDS:
            continue

        uid = battle_uid(b)

        # Skip if we've already seen this match globally
        if uid in seen_battle_ids:
            continue

        seen_battle_ids.add(uid)
        unique.append(b)

        if len(unique) >= battles_per_player:
            break

    return unique

def is_player_tag(obj: Any) -> bool:
    """Return True if obj looks like a player tag."""
    return isinstance(obj, str) and bool(PLAYER_TAG_RE.match(obj.strip().upper()))

def iter_tags_in_obj(obj: Any) -> Iterable[str]:
    """Walk nested structures and yield only strings that look like player tags."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_tags_in_obj(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_tags_in_obj(item)
    else:
        if is_player_tag(obj):
            yield normalize_tag(obj)

def compact_battles_for_df(battles: Any) -> List[Dict[str, Any]]:
    """
    Convert raw battle objects into compact rows suitable for a DataFrame.

    Assumes:
      - Exactly 8 regular cards in `cards`
      - Exactly 1 support card in `supportCards`
      - Always present for both players
    """
    if not isinstance(battles, list):
        return []

    rows: List[Dict[str, Any]] = []

    for b in battles:
        gm_id = (b.get("gameMode") or {}).get("id")

        team0 = b["team"][0]
        opp0  = b["opponent"][0]

        # 8 regular + 1 support
        p_ids = [c["id"] for c in team0["cards"][:8]]
        p_ids.append(team0["supportCards"][0]["id"])

        o_ids = [c["id"] for c in opp0["cards"][:8]]
        o_ids.append(opp0["supportCards"][0]["id"])

        player_crowns = team0.get("crowns", 0)
        opp_crowns    = opp0.get("crowns", 0)
        crown_diff    = player_crowns - opp_crowns

        if crown_diff > 0:
            result = "win"
        elif crown_diff < 0:
            result = "loss"
        else:
            result = "draw"

        rows.append({
            "battleTime": b.get("battleTime"),
            "gameModeId": gm_id,
            "playerTag": team0.get("tag"),
            "opponentTag": opp0.get("tag"),
            "playerCrowns": player_crowns,
            "opponentCrowns": opp_crowns,
            "crown_diff": crown_diff,
            "result": result,
            "playerCards": p_ids,      # length 9
            "opponentCards": o_ids,    # length 9
        })

    return rows

def bfs_battlelog_crawl(
    seed_tag: str,
    *,
    max_players: int = 5,
    battles_per_player: int = 5,
    request_delay_s: float = 0.15,
    out_path: str = os.path.join("data", "battlelog_crawl.json"),
    checkpoint_every: int = 50,
) -> Dict[str, Any]:
    """
    Breadth-first traversal of the player graph induced by battlelogs.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seed = normalize_tag(seed_tag)
    queue = deque([seed])

    visited: Set[str] = set()
    players: Dict[str, Any] = {}
    edges: Set[Tuple[str, str]] = set()

    session = requests.Session()
    seen_battle_ids: Set[str] = set()

    def checkpoint():
        payload = {
            "meta": {
                "seed": seed,
                "max_players": max_players,
                "battles_per_player": battles_per_player,
                "players_fetched": len(visited),
            },
            "players": players
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    while queue and len(visited) < max_players:
        current = queue.popleft()
        if current in visited:
            continue

        try:
            raw = get_battlelog_raw(current, session=session)
        except requests.RequestException as e:
            # Store the error and move on
            players[current] = {"error": str(e), "battlelog": None}
            visited.add(current)
            continue

        trimmed = filter_unique_battles(raw, seen_battle_ids, battles_per_player)

        # Skip any players with no battles
        if (len(trimmed) == 0):
            continue

        compact = compact_battles_for_df(trimmed)
        if not compact:
            continue  # nothing useful for this player

        players[current] = {
            "battles": compact,
            "fetched_battles": len(compact),
        }

        visited.add(current)

        discovered = set(iter_tags_in_obj(trimmed))
        discovered.discard(current)

        for t in discovered:
            edges.add((current, t))
            if t not in visited:
                queue.append(t)

        if request_delay_s > 0:
            time.sleep(request_delay_s)

        if checkpoint_every and (len(visited) % checkpoint_every == 0):
            checkpoint()

    checkpoint()
    return {
        "seed": seed,
        "players_fetched": len(visited),
        "edges_recorded": len(edges),
        "out_path": out_path,
    }

def battles_json_to_df(battles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for b in battles:
        p_cards = b["playerCards"]      # list of 9 ints
        o_cards = b["opponentCards"]    # list of 9 ints

        row: Dict[str, Any] = {
            "battleTime": b.get("battleTime"),
            "gameModeId": b.get("gameModeId"),
            "playerTag": b.get("playerTag"),
            "opponentTag": b.get("opponentTag"),
            "playerCrowns": b.get("playerCrowns", 0) or 0,
            "opponentCrowns": b.get("opponentCrowns", 0) or 0,
            "crown_diff": b.get("crown_diff", 0) or 0,
            "result": b.get("result"),
        }

        # Add 9 card-id features per side
        for i in range(9):
            row[f"p_card_{i+1}"] = int(p_cards[i])
            row[f"o_card_{i+1}"] = int(o_cards[i])

        rows.append(row)

    return pd.DataFrame(rows)

def load_battles_file_to_df(path: str) -> pd.DataFrame:
    """
    Loads the NEW crawl JSON format that stores compact battles:
      {
        "meta": {...},
        "players": {
          "<tag>": {
            "battles": [ {compact battle}, ... ],
            "fetched_battles": n
          },
          ...
        }
      }

    Returns a single DataFrame containing all compact battles across all players,
    expanded to p_card_1..8 and o_card_1..8 columns.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not (isinstance(payload, dict) and isinstance(payload.get("players"), dict)):
        raise ValueError(f"Expected crawl JSON with top-level 'players' dict: {path}")

    all_battles: List[Dict[str, Any]] = []

    for _tag, pdata in payload["players"].items():
        if not isinstance(pdata, dict):
            continue
        battles = pdata.get("battles")
        if isinstance(battles, list) and battles:
            all_battles.extend(battles)

    return battles_json_to_df(all_battles)

def plot_card_usage_and_win_rates(
    df: pd.DataFrame,
    cards_df: pd.DataFrame,
    *,
    top_n_usage: int = 30,
    top_n_win: int = 30,
    min_uses_for_winrate: int = 200,
) -> pd.DataFrame:
    """
    Works with one-hot encoded results:
        result_win, result_loss, result_draw
    """

    usage_counts = defaultdict(int)
    win_counts = defaultdict(int)

    p_cols = [f"p_card_{i}" for i in range(1, 9)]
    o_cols = [f"o_card_{i}" for i in range(1, 9)]

    total_battles = len(df)
    total_card_slots = total_battles * 16 if total_battles else 1

    for _, row in df.iterrows():
        player_cards = [row[c] for c in p_cols if pd.notna(row[c])]
        opponent_cards = [row[c] for c in o_cols if pd.notna(row[c])]

        # Count usage
        for card in player_cards:
            usage_counts[card] += 1
        for card in opponent_cards:
            usage_counts[card] += 1

        # Count wins using one-hot columns
        if row.get("result_win", 0) == 1:
            for card in player_cards:
                win_counts[card] += 1
        elif row.get("result_loss", 0) == 1:
            for card in opponent_cards:
                win_counts[card] += 1
        # draws ignored

    stats = []
    for card_id, uses in usage_counts.items():
        wins = win_counts.get(card_id, 0)
        stats.append({
            "card_id": int(card_id),
            "usage_count": uses,
            "usage_rate": uses / total_card_slots,
            "win_count": wins,
            "win_rate": (wins / uses) if uses else 0.0,
        })

    card_stats = pd.DataFrame(stats)

    # Map ID → name
    id_to_name = cards_df.set_index("id")["name"]
    card_stats["name"] = card_stats["card_id"].map(id_to_name).fillna(card_stats["card_id"].astype(str))

    card_stats = card_stats.sort_values("usage_rate", ascending=False).reset_index(drop=True)

    # --------------------------
    # Top Usage Plot
    # --------------------------
    top_usage = card_stats.head(top_n_usage).sort_values("usage_rate")

    plt.figure()
    plt.barh(top_usage["name"], top_usage["usage_rate"])
    plt.xlabel("Usage Rate")
    plt.title(f"Top {top_n_usage} Card Usage Rates")
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Top Win Rate Plot
    # --------------------------
    eligible = card_stats[card_stats["usage_count"] >= min_uses_for_winrate]
    top_win = (
        eligible
        .sort_values("win_rate", ascending=False)
        .head(top_n_win)
        .sort_values("win_rate")
    )

    plt.figure()
    plt.barh(top_win["name"], top_win["win_rate"])
    plt.xlabel(f"Win Rate (min uses = {min_uses_for_winrate})")
    plt.title(f"Top {top_n_win} Card Win Rates")
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Scatter Plot
    # --------------------------
    plt.figure()
    plt.scatter(eligible["usage_rate"], eligible["win_rate"])
    plt.xlabel("Usage Rate")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Usage Rate")
    plt.tight_layout()
    plt.show()

    return card_stats
