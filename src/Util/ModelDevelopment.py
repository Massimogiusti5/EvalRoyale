from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd

import numpy as np

def compute_deck_metadata(deck_ids, cards_df):
    """
    Compute deck metadata (avg/total elixir + troop/spell/building counts)
    using cards_df with columns: ['id', 'name', 'elixirCost', ...].

    Assumptions:
      - 260xxxxx = troop/champion
      - 270xxxxx = building
      - 280xxxxx = spell
      - 159xxxxx = tower troop (ignored in counts by default)
    """
    deck_ids = list(map(int, deck_ids))

    # Fast lookup: id -> elixirCost
    id_to_elixir = dict(zip(cards_df["id"].astype(int), cards_df["elixirCost"]))

    troop_count = 0
    spell_count = 0
    building_count = 0
    unknown_ids = []
    tower_ids = []

    elixirs = []

    for cid in deck_ids:
        # Type inference from ID bands
        if 26000000 <= cid < 27000000:
            troop_count += 1
        elif 27000000 <= cid < 28000000:
            building_count += 1
        elif 28000000 <= cid < 29000000:
            spell_count += 1
        elif 159000000 <= cid < 160000000:
            tower_ids.append(cid)  # not part of 8-card deck usually
        else:
            unknown_ids.append(cid)

        # Elixir
        if cid in id_to_elixir:
            cost = id_to_elixir[cid]
            if cost is not None and not (isinstance(cost, float) and np.isnan(cost)):
                elixirs.append(float(cost))
        else:
            unknown_ids.append(cid)

    total_elixir = float(np.sum(elixirs)) if elixirs else np.nan
    avg_elixir = float(np.mean(elixirs)) if elixirs else np.nan

    return {
        "avg_elixir": avg_elixir,
        "total_elixir": total_elixir,
        "troop_count": troop_count,
        "spell_count": spell_count,
        "building_count": building_count,
        "missing_card_ids": sorted(set(unknown_ids)),
        "tower_card_ids": tower_ids,
        "elixir_cards_counted": len(elixirs),
    }


def build_matchup_features(player_deck, opponent_deck, cards_df, model_columns):
    """
    Create a full feature row matching training data format.
    """

    # Start empty row
    feature_row = pd.DataFrame(0, index=[0], columns=model_columns)

    # Fill deck presence
    for card in player_deck:
        col = f"p_has_{card}"
        if col in feature_row.columns:
            feature_row[col] = 1

    for card in opponent_deck:
        col = f"o_has_{card}"
        if col in feature_row.columns:
            feature_row[col] = 1

    # Compute metadata
    p_meta = compute_deck_metadata(player_deck, cards_df)
    o_meta = compute_deck_metadata(opponent_deck, cards_df)

    # Add player metadata
    feature_row["p_avg_elixir"] = p_meta["avg_elixir"]
    feature_row["p_total_elixir"] = p_meta["total_elixir"]
    feature_row["p_troop_count"] = p_meta["troop_count"]
    feature_row["p_spell_count"] = p_meta["spell_count"]
    feature_row["p_building_count"] = p_meta["building_count"]

    # Add opponent metadata
    feature_row["o_avg_elixir"] = o_meta["avg_elixir"]
    feature_row["o_total_elixir"] = o_meta["total_elixir"]
    feature_row["o_troop_count"] = o_meta["troop_count"]
    feature_row["o_spell_count"] = o_meta["spell_count"]
    feature_row["o_building_count"] = o_meta["building_count"]

    # Difference features
    feature_row["avg_elixir_diff"] = p_meta["avg_elixir"] - o_meta["avg_elixir"]
    feature_row["troop_diff"] = p_meta["troop_count"] - o_meta["troop_count"]
    feature_row["spell_diff"] = p_meta["spell_count"] - o_meta["spell_count"]
    feature_row["building_diff"] = p_meta["building_count"] - o_meta["building_count"]

    feature_row["gameModeId"] = 72000450  # Ladder
    feature_row["hour"] = 12              # Midday
    feature_row["day_of_week"] = 3        # Wednesday

    return feature_row


@dataclass(frozen=True)
class FeatureArtifacts:
    """Useful objects you may want to reuse (e.g., for inference/sample rows later)."""
    all_cards: np.ndarray
    card_to_index: Dict[int, int]
    card_elixir: Dict[int, float]
    card_types: Dict[int, str]
    p_cols: List[str]
    o_cols: List[str]


def _card_type_from_id(cid: int) -> str:
    s = str(int(cid))
    if s.startswith("26"):
        return "troop"
    if s.startswith("27"):
        return "building"
    if s.startswith("28"):
        return "spell"
    return "unknown"


def _build_presence_matrix(
    df_cards: pd.DataFrame,
    cols: Sequence[str],
    *,
    all_cards: np.ndarray,
    card_to_index: Dict[int, int],
    prefix: str,
) -> pd.DataFrame:
    """
    Presence one-hot (deck-as-set), ignoring slot order.
    Columns become e.g. 'p_has_26000000', etc.
    """
    n = len(df_cards)
    m = len(all_cards)

    mat = np.zeros((n, m), dtype=np.uint8)

    # allow NaNs -> use pandas nullable Int64 first, then fill
    vals = df_cards.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.int64).to_numpy()

    row_idx = np.repeat(np.arange(n), vals.shape[1])
    flat = vals.ravel()

    col_idx = np.array([card_to_index.get(int(cid), -1) for cid in flat], dtype=np.int64)
    valid = col_idx >= 0

    mat[row_idx[valid], col_idx[valid]] = 1

    return pd.DataFrame(
        mat,
        columns=[f"{prefix}{int(cid)}" for cid in all_cards],
        index=df_cards.index,
    )


def _deck_features_from_cols(
    df_cards: pd.DataFrame,
    cols: Sequence[str],
    *,
    card_elixir: Dict[int, float],
    card_types: Dict[int, str],
    prefix: str,
) -> pd.DataFrame:
    vals = df_cards.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.int64).to_numpy()

    # Elixir features
    elixir = np.full(vals.shape, np.nan, dtype=float)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        elixir[:, j] = [card_elixir.get(int(cid), np.nan) if cid >= 0 else np.nan for cid in col]

    avg_elixir = np.nanmean(elixir, axis=1)
    total_elixir = np.nansum(elixir, axis=1)

    # Type counts
    troop_count = np.zeros(len(df_cards), dtype=np.int16)
    spell_count = np.zeros(len(df_cards), dtype=np.int16)
    building_count = np.zeros(len(df_cards), dtype=np.int16)

    for j in range(vals.shape[1]):
        cids = vals[:, j]
        troop_count += np.array([(1 if card_types.get(int(cid)) == "troop" else 0) for cid in cids], dtype=np.int16)
        spell_count += np.array([(1 if card_types.get(int(cid)) == "spell" else 0) for cid in cids], dtype=np.int16)
        building_count += np.array([(1 if card_types.get(int(cid)) == "building" else 0) for cid in cids], dtype=np.int16)

    return pd.DataFrame(
        {
            f"{prefix}avg_elixir": avg_elixir,
            f"{prefix}total_elixir": total_elixir,
            f"{prefix}troop_count": troop_count,
            f"{prefix}spell_count": spell_count,
            f"{prefix}building_count": building_count,
        },
        index=df_cards.index,
    )


def build_features_and_target(
    df: pd.DataFrame,
    cards_df: pd.DataFrame,
    *,
    n_cards_per_side: int = 9,
    p_prefix: str = "p_card_",
    o_prefix: str = "o_card_",
    base_cols: Sequence[str] = ("gameModeId", "hour", "day_of_week"),
    target_col: str = "result_win",
    leakage_cols: Sequence[str] = ("playerCrowns", "opponentCrowns", "crown_diff", "result_loss"),
    id_cols_to_drop: Sequence[str] = ("playerTag", "opponentTag"),
    drop_raw_time: bool = True,
    return_artifacts: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[FeatureArtifacts]]:
    """
    Build ML features X and target y from your battle dataframe + cards metadata.

    Returns:
        X: feature matrix
        y: target series
        artifacts (optional): lookup tables you can reuse for inference
    """
    df = df.copy()

    # card columns
    p_cols = [f"{p_prefix}{i}" for i in range(1, n_cards_per_side + 1)]
    o_cols = [f"{o_prefix}{i}" for i in range(1, n_cards_per_side + 1)]

    missing = [c for c in (list(p_cols) + list(o_cols)) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected card columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # time features
    df["battleTime"] = pd.to_datetime(df["battleTime"], errors="coerce", utc=True)
    df["hour"] = df["battleTime"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["battleTime"].dt.dayofweek.fillna(0).astype(int)

    # target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df.")
    y = df[target_col].astype(int)

    # cards lookup tables
    cards_df = cards_df.copy()
    cards_df["id"] = pd.to_numeric(cards_df["id"], errors="coerce").astype(int)

    card_elixir: Dict[int, float] = dict(zip(cards_df["id"], cards_df["elixirCost"]))
    card_types: Dict[int, str] = {int(cid): _card_type_from_id(int(cid)) for cid in cards_df["id"].values}

    all_cards = np.array(sorted(cards_df["id"].unique()), dtype=np.int64)
    card_to_index = {int(cid): i for i, cid in enumerate(all_cards)}

    # presence one-hot
    p_presence = _build_presence_matrix(df, p_cols, all_cards=all_cards, card_to_index=card_to_index, prefix="p_has_")
    o_presence = _build_presence_matrix(df, o_cols, all_cards=all_cards, card_to_index=card_to_index, prefix="o_has_")

    # deck engineered features
    p_deck = _deck_features_from_cols(df, p_cols, card_elixir=card_elixir, card_types=card_types, prefix="p_")
    o_deck = _deck_features_from_cols(df, o_cols, card_elixir=card_elixir, card_types=card_types, prefix="o_")

    # base
    X_base = df.loc[:, list(base_cols)].copy()

    # combine
    X = pd.concat([X_base, p_presence, o_presence, p_deck, o_deck], axis=1)

    # difference features
    X["avg_elixir_diff"] = X["p_avg_elixir"] - X["o_avg_elixir"]
    X["spell_diff"] = X["p_spell_count"] - X["o_spell_count"]
    X["building_diff"] = X["p_building_count"] - X["o_building_count"]
    X["troop_diff"] = X["p_troop_count"] - X["o_troop_count"]

    # drop leakage + ids + raw time (if present)
    drop_cols = [c for c in leakage_cols if c in df.columns] + [c for c in id_cols_to_drop if c in df.columns]
    if drop_raw_time:
        drop_cols.append("battleTime")

    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=c)

    artifacts = None
    if return_artifacts:
        artifacts = FeatureArtifacts(
            all_cards=all_cards,
            card_to_index=card_to_index,
            card_elixir=card_elixir,
            card_types=card_types,
            p_cols=p_cols,
            o_cols=o_cols,
        )

    return X, y, artifacts
