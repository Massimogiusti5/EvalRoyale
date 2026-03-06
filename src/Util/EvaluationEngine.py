import re
import numpy as np
import pandas as pd
from difflib import get_close_matches
import ModelDevelopment

def _norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def build_name_to_id(cards_df: pd.DataFrame) -> dict:
    """
    cards_df columns expected:
      name, id
    """
    if not {"name", "id"}.issubset(cards_df.columns):
        raise ValueError("cards_df must contain columns ['name', 'id'].")

    m = {}
    for _, r in cards_df.iterrows():
        m[_norm_name(r["name"])] = int(r["id"])
    return m

def resolve_ids(deck_names, name_to_id: dict, *, deck_label="deck"):
    """
    deck_names: list[str] length 9 (8 cards + tower troop)
    Returns list[int]
    """
    if len(deck_names) != 9:
        raise ValueError(f"{deck_label} must have length 9 (8 cards + tower troop). Got {len(deck_names)}")

    known_keys = list(name_to_id.keys())
    ids = []

    for nm in deck_names:
        k = _norm_name(nm)

        # allow raw numeric ids
        if k.isdigit():
            ids.append(int(k))
            continue

        if k in name_to_id:
            ids.append(name_to_id[k])
            continue

        sug = get_close_matches(k, known_keys, n=1, cutoff=0.78)
        if sug:
            ids.append(name_to_id[sug[0]])
            continue

        raise ValueError(
            f"Unknown card/tower troop name '{nm}' in {deck_label}. "
            "Check spelling vs cards_df['name']."
        )

    return ids


def build_matchup_row(
    deckA_names,
    deckB_names,
    *,
    cards_df: pd.DataFrame,
    feature_columns,
    gameModeId: int = 72000006,
    hour: int = 12,
    day_of_week: int = 2,
    strict_missing_cols: bool = False,
):
    """
    Creates a 1-row DataFrame aligned to feature_columns.
    Uses p_has_<id>, o_has_<id> one-hot and your compute_deck_metadata().

    strict_missing_cols:
      - False (default): silently ignores features not present in feature_columns
      - True: raises if any expected feature columns are missing
    """
    name_to_id = build_name_to_id(cards_df)
    p_ids = resolve_ids(deckA_names, name_to_id, deck_label="deckA (player)")
    o_ids = resolve_ids(deckB_names, name_to_id, deck_label="deckB (opponent)")

    # init zeros
    feature_columns = list(feature_columns)
    x = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)

    def _set_if_present(col, val):
        if col in x.columns:
            x.loc[0, col] = float(val)
        elif strict_missing_cols:
            raise KeyError(f"Feature column '{col}' missing from feature_columns")

    # base features
    _set_if_present("gameModeId", gameModeId)
    _set_if_present("hour", hour)
    _set_if_present("day_of_week", day_of_week)

    # one-hot presence
    for cid in p_ids:
        col = f"p_has_{int(cid)}"
        if col in x.columns:
            x.loc[0, col] = 1.0
        elif strict_missing_cols:
            raise KeyError(f"Feature column '{col}' missing from feature_columns")

    for cid in o_ids:
        col = f"o_has_{int(cid)}"
        if col in x.columns:
            x.loc[0, col] = 1.0
        elif strict_missing_cols:
            raise KeyError(f"Feature column '{col}' missing from feature_columns")

    # metadata
    p_meta = ModelDevelopment.compute_deck_metadata(p_ids, cards_df)
    o_meta = ModelDevelopment.compute_deck_metadata(o_ids, cards_df)

    def _nan_to_0(v):
        return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    p_avg = _nan_to_0(p_meta["avg_elixir"])
    p_tot = _nan_to_0(p_meta["total_elixir"])
    o_avg = _nan_to_0(o_meta["avg_elixir"])
    o_tot = _nan_to_0(o_meta["total_elixir"])

    # write p_ and o_ numeric features
    _set_if_present("p_avg_elixir", p_avg)
    _set_if_present("p_total_elixir", p_tot)
    _set_if_present("p_troop_count", p_meta["troop_count"])
    _set_if_present("p_spell_count", p_meta["spell_count"])
    _set_if_present("p_building_count", p_meta["building_count"])

    _set_if_present("o_avg_elixir", o_avg)
    _set_if_present("o_total_elixir", o_tot)
    _set_if_present("o_troop_count", o_meta["troop_count"])
    _set_if_present("o_spell_count", o_meta["spell_count"])
    _set_if_present("o_building_count", o_meta["building_count"])

    # diffs
    _set_if_present("avg_elixir_diff", p_avg - o_avg)
    _set_if_present("spell_diff", p_meta["spell_count"] - o_meta["spell_count"])
    _set_if_present("building_diff", p_meta["building_count"] - o_meta["building_count"])
    _set_if_present("troop_diff", p_meta["troop_count"] - o_meta["troop_count"])

    return x

def predict_matchup_win_pct(
    model,
    deckA_names,
    deckB_names,
    *,
    cards_df: pd.DataFrame,
    feature_columns,
    gameModeId: int = 72000006,
    hour: int = 12,
    day_of_week: int = 2,
):
    """
    Returns (win_pct_for_deckA, X_row)
    """
    X_row = build_matchup_row(
        deckA_names,
        deckB_names,
        cards_df=cards_df,
        feature_columns=feature_columns,
        gameModeId=gameModeId,
        hour=hour,
        day_of_week=day_of_week,
    )
    prob = float(model.predict_proba(X_row)[0, 1])
    return prob * 100.0, X_row

def evaluate_matchup(
    model,
    deckA,
    deckB,
    *,
    cards_df,
    feature_columns,
    gameModeId=72000006,
    hour=12,
    day_of_week=2
):

    # A vs B
    p_ab, _ = predict_matchup_win_pct(
        model,
        deckA,
        deckB,
        cards_df=cards_df,
        feature_columns=feature_columns,
        gameModeId=gameModeId,
        hour=hour,
        day_of_week=day_of_week
    )

    # B vs A
    p_ba, _ = predict_matchup_win_pct(
        model,
        deckB,
        deckA,
        cards_df=cards_df,
        feature_columns=feature_columns,
        gameModeId=gameModeId,
        hour=hour,
        day_of_week=day_of_week
    )

    advantage = p_ab - p_ba

    return {
        "A_beats_B": p_ab,
        "B_beats_A": p_ba,
        "advantage": advantage
    }