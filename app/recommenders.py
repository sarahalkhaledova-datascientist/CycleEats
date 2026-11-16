"""CycleEats recommenders.

All five recommenders plus the Hybrid used by the Streamlit app, built on the
files written by notebook 10 (``app/data``). Same logic as notebooks 07-09:

- Popularity      : Bayesian weighted rating (all interactions)
- Content-Based   : TF-IDF on ingredients, max-similarity to liked recipes ("B")
- Item-Based CF   : Jaccard, top-10 neighbours
- User-Based CF   : Jaccard, top-100 neighbours, computed live from the user's hearts
- Knowledge-Based : cycle-phase score (notebook 05 rules)
- Hybrid          : weighted sum of per-component ranks (0-1) + phase + pantry match

The app user is new (not in Food.com), so her "hearts" (favourites) act as
likes. With no hearts, Popularity stands in for the personal systems.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

DATA_DIR = Path(__file__).resolve().parent / "data"
UNITS = ["g", "oz", "cup", "tbsp", "tsp", "piece"]
PHASE_LABELS = {"menstrual": "Menstrual", "follicular": "Follicular", "ovulatory": "Ovulatory", "luteal": "Luteal"}


# ----------------------------------------------------------------------------- helpers
def rank01(x: np.ndarray) -> np.ndarray:
    """Relative rank in [0, 1]: highest = 1, lowest = 0, ties share a value (as notebook 09)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.ones_like(x)
    return ((rankdata(x, method="average") - 1) / (len(x) - 1)).astype(np.float32)


def percentile_against(sorted_ref: np.ndarray, values) -> np.ndarray:
    """Where values would rank inside a reference distribution, on the same 0-1 scale as rank01."""
    values = np.atleast_1d(np.asarray(values, dtype=float))
    lo = np.searchsorted(sorted_ref, values, side="left")
    hi = np.searchsorted(sorted_ref, values, side="right")
    return (((lo + hi) / 2) / max(len(sorted_ref), 1)).astype(np.float32)


def cycle_phase(last_period_start: date, cycle_length: int = 28, period_length: int = 5,
                today: date | None = None) -> tuple[str, int]:
    """Simple average-cycle estimate. Returns (phase, cycle day 1..cycle_length).

    menstrual: days 1..period_length; ovulatory: ovulation day +-1 (ovulation = cycle_length - 14);
    follicular: between them; luteal: after ovulation.
    """
    today = today or date.today()
    day = (today - last_period_start).days % cycle_length + 1
    ovulation = cycle_length - 14
    if day <= period_length:
        return "menstrual", day
    if ovulation - 1 <= day <= ovulation + 1:
        return "ovulatory", day
    if day < ovulation - 1:
        return "follicular", day
    return "luteal", day


@dataclass
class OwnRecipe:
    """A recipe the user typed in. ingredients = [(name, quantity, unit), ...]."""
    name: str
    servings: float
    ingredients: list
    meal_type: str = "Main"
    instructions: str = ""
    id: str = ""
    info: dict = field(default_factory=dict)   # filled by Recommender.analyse_own


# ----------------------------------------------------------------------------- recommender
class Recommender:
    def __init__(self, data_dir: str | Path = DATA_DIR):
        d = Path(data_dir)
        self.config = json.loads((d / "app_config.json").read_text(encoding="utf-8"))
        self.weights_default = dict(self.config["weights"])
        self.user_k = int(self.config["user_k"])
        self.phases = list(self.config["phases"])

        r = pd.read_parquet(d / "recipes_app.parquet")
        self.recipes = r.reset_index(drop=True)
        self.ids = self.recipes["RecipeId"].to_numpy(np.int64)
        self.idx = pd.Series(np.arange(len(self.ids)), index=self.ids)
        self.n = len(self.ids)
        self.meal = self.recipes["meal_type"].to_numpy()
        self.parts = [set(p) for p in self.recipes["parts_norm"]]

        # Knowledge-Based
        self.phase_score = {p: self.recipes[f"score_{p}"].fillna(0).to_numpy(float) for p in self.phases}
        self.phase_rank = {p: rank01(v) for p, v in self.phase_score.items()}
        self.phase_sorted = {p: np.sort(v) for p, v in self.phase_score.items()}

        # Popularity
        st = pd.read_csv(d / "onboarding_ids.csv")
        self.onboarding_ids = st["RecipeId"].astype("int64").tolist()
        self.pop_rank = self._load_popularity(d)

        # Content-Based
        self.X = sparse.load_npz(d / "tfidf_catalog.npz").tocsr()
        voc = json.loads((d / "tfidf_vocab.json").read_text(encoding="utf-8"))
        self.vocab = {w: i for i, w in enumerate(voc["vocab"])}
        self.idf = np.asarray(voc["idf"], dtype=float)
        self.XT = self.X.T.tocsc()

        # Item-CF (rows = neighbour j, cols = candidate i)
        self.S_item = sparse.load_npz(d / "itemknn_catalog.npz").tocsr()

        # User-CF
        self.B = sparse.load_npz(d / "usercf_interactions.npz").tocsc()
        self.L = sparse.load_npz(d / "usercf_liked.npz").tocsr()
        self.user_total = np.load(d / "usercf_counts.npy").astype(float)

        # pantry + own recipes
        self.staples = set(pd.read_csv(d / "pantry_staples.csv")["ingredient"])
        self.ingredients = pd.read_csv(d / "ingredients_app.csv").set_index("ingredient")
        self.dv = pd.read_csv(d / "daily_values.csv").set_index("nutrient")
        self.rules = pd.read_csv(d / "phase_rules.csv")
        self.params = pd.read_csv(d / "phase_params.csv").set_index("phase")
        self.thresholds = pd.read_csv(d / "phase_thresholds.csv").set_index("phase")["threshold"].to_dict()
        self.scoring = self.config["phase_scoring"]

    def _load_popularity(self, d: Path) -> np.ndarray:
        """Popularity rank saved by notebook 10 (Bayesian weighted rating, all interactions)."""
        if (d / "popularity.npy").exists():
            return np.load(d / "popularity.npy").astype(np.float32)
        # fallback if the file is missing: smoothed share of likes, weighted by the number of interactions
        L = sparse.load_npz(d / "usercf_liked.npz")
        B = sparse.load_npz(d / "usercf_interactions.npz")
        likes = np.asarray(L.sum(axis=0)).ravel()
        inter = np.asarray(B.sum(axis=0)).ravel()
        share = (likes + 1) / (inter + 2)                    # smoothed share of interactions that are likes
        conf = np.log1p(inter)                               # more interactions = more trust
        return rank01(share * conf + 1e-6 * inter)

    # ------------------------------------------------------------------ components
    def _liked_idx(self, hearts) -> np.ndarray:
        return np.array([self.idx[h] for h in hearts if h in self.idx.index], dtype=np.int64)

    def tfidf_row(self, parts) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for p in parts:
            i = self.vocab.get(p)
            if i is not None:
                v[i] += 1
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def content_scores(self, liked: np.ndarray, own_rows: list[np.ndarray] | None = None) -> np.ndarray:
        """Max cosine similarity of each catalog recipe to any liked recipe (or own recipe)."""
        s = np.zeros(self.n)
        if len(liked):
            s = np.maximum(s, (self.X[liked] @ self.XT).max(axis=0).toarray().ravel())
        for row in own_rows or []:
            s = np.maximum(s, self.X @ row)
        return s

    def item_cf_scores(self, liked: np.ndarray) -> np.ndarray:
        if not len(liked):
            return np.zeros(self.n)
        v = np.zeros(self.n)
        v[liked] = 1
        return np.asarray(self.S_item.T @ v).ravel()

    def user_cf_scores(self, liked: np.ndarray) -> np.ndarray:
        if not len(liked):
            return np.zeros(self.n)
        co = np.asarray(self.B[:, liked].sum(axis=1)).ravel()          # shared recipes with each user
        denom = len(liked) + self.user_total - co
        sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
        k = min(self.user_k, int((sim > 0).sum()))
        if k == 0:
            return np.zeros(self.n)
        top = np.argpartition(-sim, k - 1)[:k]
        return np.asarray(self.L[top].T @ sim[top]).ravel()

    def pantry_match(self, pantry: set[str] | None, use_staples: bool = True):
        """Share of each recipe's ingredients the user has, and the missing ones."""
        have = set(pantry or []) | (self.staples if use_staples else set())
        share = np.array([len(p & have) / len(p) if p else 0.0 for p in self.parts])
        return share, have

    def missing(self, recipe_idx: int, have: set[str]) -> list[str]:
        return sorted(self.parts[recipe_idx] - have)

    # ------------------------------------------------------------------ hybrid
    def recommend(self, hearts=(), phase: str | None = None, pantry=None, meal_types=None,
                  own_recipes: list[OwnRecipe] | None = None, weights: dict | None = None,
                  only_cookable: bool = False, max_missing: int | None = None,
                  use_staples: bool = True, k: int = 10, exclude=()) -> pd.DataFrame:
        """Top-k recommendations as a DataFrame (catalog recipes and, if suitable, own recipes)."""
        w = dict(self.weights_default)
        w.update(weights or {})
        max_missing = self.config["max_missing_default"] if max_missing is None else max_missing
        liked = self._liked_idx(hearts)
        own_recipes = [o for o in (own_recipes or []) if o.info]
        own_rows = [o.info["tfidf"] for o in own_recipes]

        # personal components (ranks over the catalog)
        raw = {"popularity": None,
               "content": self.content_scores(liked, own_rows) if (len(liked) or own_rows) else None,
               "item_cf": self.item_cf_scores(liked) if len(liked) else None,
               "user_cf": self.user_cf_scores(liked) if len(liked) else None}
        avail = {c: w.get(c, 0) for c in raw if c == "popularity" or raw[c] is not None}
        if sum(v for c, v in avail.items() if c != "popularity") <= 0:     # cold start (or all personal sliders at 0)
            avail = {"popularity": 1.0}
        total_w = sum(avail.values())
        comp_rank = {c: (self.pop_rank if c == "popularity" else rank01(raw[c])) for c in avail}
        personal = sum(avail[c] / total_w * comp_rank[c] for c in avail)

        score = personal.astype(float).copy()
        if phase:
            score += w.get("phase", 0) * self.phase_rank[phase]
        share, have = self.pantry_match(pantry, use_staples)
        use_pantry = bool(pantry)
        if use_pantry:
            score += w.get("pantry", 0) * share
        score += 1e-6 * self.pop_rank                                       # tie-break

        ok = np.ones(self.n, dtype=bool)
        if meal_types:
            ok &= np.isin(self.meal, list(meal_types))
        ok[liked] = False
        if len(exclude):
            ok[self._liked_idx(exclude)] = False
        if only_cookable and use_pantry:
            n_missing = np.array([len(p - have) for p in self.parts])
            ok &= n_missing <= max_missing
        cand = np.where(ok)[0]
        order = cand[np.argsort(-score[cand])][:k]

        rows = []
        for i in order:
            rec = self.recipes.iloc[i]
            rows.append({
                "id": int(self.ids[i]), "is_own": False, "name": rec["Name"], "meal_type": rec["meal_type"],
                "minutes": rec["TotalTime_min"], "calories": rec["Calories"], "score": float(score[i]),
                "phase_fit": bool(rec[f"label_{phase}"]) if phase else False,
                "why": rec[f"why_{phase}"] if phase else "",
                "match": float(share[i]) if use_pantry else None,
                "missing": self.missing(i, have) if use_pantry else [],
                "image_url": rec["image_url"],
            })

        # own recipes: only when suitable for the phase, at most own_top10_cap of the top k
        own_rows_out = []
        if phase:
            sorted_personal = {c: np.sort(comp_rank[c] if c == "popularity" else raw[c]) for c in avail}
            for o in own_recipes:
                if not o.info["labels"].get(phase):
                    continue
                if meal_types and o.meal_type not in meal_types:
                    continue
                others = [x.info["tfidf"] for x in own_recipes if x is not o]
                content_raw = 0.0
                if len(liked):
                    content_raw = float((self.X[liked] @ o.info["tfidf"]).max())
                for row in others:
                    content_raw = max(content_raw, float(row @ o.info["tfidf"]))
                own_raw = {"popularity": 0.0, "content": content_raw, "item_cf": 0.0, "user_cf": 0.0}
                pers = sum(avail[c] / total_w * percentile_against(sorted_personal[c], own_raw[c])[0] for c in avail)
                s = pers + w.get("phase", 0) * percentile_against(self.phase_sorted[phase], o.info["scores"][phase])[0]
                own_parts = set(o.info["parts"])
                o_share = len(own_parts & have) / len(own_parts) if own_parts else 0.0
                if use_pantry:
                    if only_cookable and len(own_parts - have) > max_missing:
                        continue
                    s += w.get("pantry", 0) * o_share
                own_rows_out.append({
                    "id": o.id, "is_own": True, "name": o.name, "meal_type": o.meal_type, "minutes": None,
                    "calories": o.info["kcal"], "score": float(s), "phase_fit": True, "why": o.info["why"][phase],
                    "match": o_share if use_pantry else None,
                    "missing": sorted(own_parts - have) if use_pantry else [], "image_url": "",
                })
        # suitable own recipes are guaranteed a place (priority), up to the cap; catalog fills the rest
        cap = int(self.config["own_top10_cap"])
        own_rows_out = sorted(own_rows_out, key=lambda r: -r["score"])[:cap]
        out = pd.DataFrame(rows[:max(k - len(own_rows_out), 0)] + own_rows_out)
        if out.empty:
            return out
        return out.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ own recipes
    def phase_core(self, pdv: dict, kcal: float):
        """Phase score, gate and contributions for one recipe (identical to notebook 05)."""
        sc = self.scoring
        dens = {n: v * 100 / max(kcal, sc["min_kcal_density"]) for n, v in pdv.items()}
        out = {}
        for ph in self.phases:
            r = self.rules[self.rules["phase"] == ph]
            prm = self.params.loc[ph]
            W = r["weight"].to_numpy(float)
            contrib = np.minimum([np.nan_to_num(dens.get(n, 0.0)) for n in r["nutrient"]], sc["density_cap"]) / sc["density_cap"] * W
            score = float(contrib.sum() / W.sum())
            gate_cols = r.loc[r["gate"] == "Y", "nutrient"]
            gate = sum(pdv.get(n, 0.0) >= prm["gate_pct"] for n in gate_cols) >= prm["gate_min_count"]
            out[ph] = (score, bool(gate), contrib, r["nutrient"].tolist())
        return out

    def analyse_own(self, own: OwnRecipe) -> OwnRecipe:
        """Nutrients per serving, phase scores/labels/reasons and a TF-IDF row for an own recipe."""
        nutr_cols = ["kcal"] + list(self.dv.index)
        total = dict.fromkeys(nutr_cols, 0.0)
        known, unknown, parts = [], [], []
        for name, qty, unit in own.ingredients:
            name = str(name).strip().lower()
            if name not in self.ingredients.index:
                unknown.append(name)
                continue
            row = self.ingredients.loc[name]
            grams = qty if unit == "g" else qty * row.get(f"g_per_{unit}", np.nan)
            if not np.isfinite(grams):
                unknown.append(name)
                continue
            known.append(name)
            parts.append(name)
            for c in nutr_cols:
                total[c] += grams / 100 * np.nan_to_num(row[c])
        n_all = max(len(own.ingredients), 1)
        servings = max(float(own.servings or 1), 1.0)
        per = {c: v / servings for c, v in total.items()}
        kcal = per["kcal"]
        pdv = {n: per[n] / self.dv.loc[n, "dv"] * 100 for n in self.dv.index}
        sc = self.scoring
        plausible = (max(pdv.values()) <= sc["max_pct_dv"]) and (sc["min_kcal_eligible"] <= kcal <= sc["max_kcal"])
        unknown_share = len(unknown) / n_all
        core = self.phase_core(pdv, kcal)
        scores, labels, why = {}, {}, {}
        for ph, (score, gate, contrib, nl) in core.items():
            scores[ph] = score
            labels[ph] = bool(gate and plausible and unknown_share <= self.config["own_unknown_max"]
                              and score >= self.thresholds[ph])
            order = np.argsort(-contrib)[:3]
            why[ph] = "; ".join(f"{self.dv.loc[nl[j], 'name']} {pdv[nl[j]]:.0f}%" for j in order if pdv[nl[j]] >= 10) if labels[ph] else ""
        own.info = {"kcal": kcal, "pdv": pdv, "scores": scores, "labels": labels, "why": why,
                    "plausible": plausible, "unknown": unknown, "unknown_share": unknown_share,
                    "parts": parts, "tfidf": self.tfidf_row(parts)}
        return own

    # ------------------------------------------------------------------ lookups
    def recipe(self, recipe_id: int) -> pd.Series:
        return self.recipes.iloc[self.idx[recipe_id]]

    def onboarding(self) -> pd.DataFrame:
        return self.recipes.iloc[[self.idx[i] for i in self.onboarding_ids if i in self.idx.index]]

    def ingredient_names(self) -> list[str]:
        return self.ingredients.index.tolist()
