"""CycleEats chatbot: retrieval-augmented answers about the app's recipes.

Flow for each question
1. understand(): the LLM turns the question into a search plan (clear search text + simple filters).
2. retrieve(): cosine similarity on the recipe embeddings (notebook 12), filters, then a small
   re-ranking (+ phase fit, + popularity, + pantry match). The best 5 go on.
3. answer(): the LLM answers ONLY from those recipes and names the ones it recommends.

The embeddings model runs locally through fastembed (ONNX, no torch). The LLM runs on Groq.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

LLM_MODEL = "openai/gpt-oss-20b"
N_CANDIDATES = 40        # nearest recipes looked at before re-ranking
N_CONTEXT = 5            # recipes sent to the LLM
MAX_HISTORY = 6          # last 3 questions + answers kept for follow-ups
BONUS_PHASE = 0.10       # re-ranking bonuses on top of cosine similarity (which is ~0.3 to 0.8)
BONUS_POPULAR = 0.03
BONUS_PANTRY = 0.05

ALWAYS_THERE = {"water", "ice", "ice cube", "boiling water", "cold water", "hot water", "warm water"}

MEAT_FISH = re.compile(
    r"chicken|beef|pork|lamb|mutton|turkey|bacon|\bham\b|sausage|salami|pepperoni|prosciutto|pancetta|chorizo|"
    r"steak|veal|venison|duck|goose|meat|mince|gelatin|lard|anchov|fish|salmon|tuna|cod\b|tilapia|halibut|trout|"
    r"sardine|mackerel|haddock|snapper|catfish|mahi|swordfish|shrimp|prawn|crab|lobster|clam|mussel|oyster|scallop|"
    r"squid|calamari|octopus", re.I)

PLAN_PROMPT = """You turn a user's message to a recipe app into a search plan. Reply with JSON only:
{
  "search_query": "short English description of the dishes to look for (dish types, main ingredients, style). Turn needs into foods, e.g. 'iron' -> 'iron-rich lentils spinach beef', 'cold evening' -> 'warm hearty soup stew'.",
  "meal_types": [],            // zero or more of: MEAL_TYPES
  "vegetarian": false,         // true only if the user wants no meat or fish
  "max_minutes": null,         // number if the user gives a time limit
  "max_calories": null,        // number if the user gives a calorie limit per serving
  "include": [],               // ingredients the user has or wants in the dish, simple singular names
                               // (e.g. 'I have chicken breast and mushrooms' -> ["chicken", "mushroom"])
  "exclude": [],               // ingredients the user does not want (allergies, dislikes), simple singular names
  "intent": "recipes"          // one of:
                               //  "recipes"   - wants recipe ideas or food/cooking help
                               //  "followup"  - asks about recipes already shown (e.g. 'how do I make the first one?')
                               //  "chitchat"  - greeting, thanks, goodbye, how are you, small talk
                               //  "off_topic" - not about food, cooking or eating
                               //  "unclear"   - cannot be understood (random letters, wrong keyboard)
}
Use the conversation for context. 'dinner' or 'lunch' means Main; 'side' means Salads & sides; 'snack' means Snacks & appetizers."""

ANSWER_PROMPT = """You are CycleEats, a friendly cooking assistant in a recipe app for women.
The user is in her PHASE phase. Nutrients that matter now: FOCUS.

Rules:
- Use ONLY the recipes in the context. Never invent recipes, ingredients, amounts or steps.
- Recommend 2 or 3 recipes that fit the question, so she has a choice (only 1 if just one really fits).
  Put recipes that fit her phase first. Use each recipe's exact name in **bold**. For each, one or two short sentences:
  time, calories, and why it fits (the question, and the phase if the recipe fits it).
- If the user asks how to make a recipe, give its ingredients and steps as written in the context.
  Amounts in the ingredient list are Food.com quantities without units; do not add units.
- If none of the recipes fits, say so honestly and suggest how to rephrase.
- Do not recommend decorations, garnishes, sauces or drinks on their own unless she asks for them.
- You are not a doctor: no medical advice, diagnosis or promises about symptoms.
- Keep it short and warm. Answer in the user's language.
- End with one line exactly like this, with the ids of the recipes you recommended or described:
  RECIPES: 123, 456"""


SMALLTALK_PROMPT = """You are CycleEats, a warm cooking assistant in a recipe app for women.
The user's name is NAME. She is in her PHASE phase (good nutrients now: FOCUS).
Her message is not a recipe request. Its type: KIND.
Reply in one or two short, friendly sentences, in the same language as her message:
- greeting / small talk: greet her back (use her name if you know it) and ask what she would like to eat today,
  maybe with one short example idea that suits her phase.
- thanks / goodbye: reply kindly and say you're here whenever she wants a meal idea.
- off_topic: say kindly that you can only help with food, recipes and cooking, and invite a food question.
- unclear: say you didn't quite understand and ask her to write it again.
Do not name specific recipes. No medical advice."""


def _singular(word: str) -> str:
    """'mushrooms' -> 'mushroom', 'tomatoes' -> 'tomato', 'berries' -> 'berr' (substring match on ingredients)."""
    for end, new in (("ies", ""), ("oes", "o"), ("s", "")):
        if word.endswith(end) and len(word) > len(end) + 2 and not word.endswith("ss"):
            return word[: -len(end)] + new
    return word


class Chatbot:
    def __init__(self, recommender, data_dir: str | Path | None = None, embedder=None):
        self.R = recommender
        d = Path(data_dir) if data_dir else Path(recommender.data_dir)
        cfg = json.loads((d / "chat_config.json").read_text())
        self.model_name = cfg["model"]
        emb = np.load(d / "chat_embeddings.npy").astype(np.float32)
        ids = np.load(d / "chat_ids.npy")
        # align the embeddings with the recommender's recipe order
        pos = {int(r): i for i, r in enumerate(ids)}
        order = np.array([pos[int(r)] for r in self.R.recipes["RecipeId"]])
        self.E = emb[order]
        self._embedder = embedder
        rec = self.R.recipes
        self.veg_ok = np.array([not any(MEAT_FISH.search(p) for p in parts) for parts in rec["parts_norm"]])
        self.meal_types = sorted(rec["meal_type"].unique())

    # ------------------------------------------------------------------ embeddings
    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(self.model_name)
        return self._embedder

    def embed(self, text: str) -> np.ndarray:
        v = np.array(list(self.embedder.embed([text])), dtype=np.float32)[0]
        return v / (np.linalg.norm(v) or 1.0)

    # ------------------------------------------------------------------ LLM
    @staticmethod
    def _chat(client, messages, json_mode=False) -> str:
        kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": 0.3}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
        except TypeError:          # older SDK without reasoning_effort
            resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def understand(self, client, question: str, history: list[dict]) -> dict:
        prompt = PLAN_PROMPT.replace("MEAL_TYPES", ", ".join(self.meal_types))
        msgs = [{"role": "system", "content": prompt}] + history[-MAX_HISTORY:] + [{"role": "user", "content": question}]
        plan = {"search_query": question}
        try:
            raw = self._chat(client, msgs, json_mode=True)
            m = re.search(r"\{.*\}", raw, re.S)
            plan.update(json.loads(m.group(0) if m else raw))
        except Exception:
            pass                   # fall back to searching the question as written
        plan["meal_types"] = [t for t in (plan.get("meal_types") or []) if t in self.meal_types]
        for k in ("include", "exclude"):
            plan[k] = [_singular(str(x).lower().strip()) for x in (plan.get(k) or []) if str(x).strip()]
        for k in ("max_minutes", "max_calories"):
            try:
                plan[k] = int(plan[k]) if plan.get(k) else None
            except (TypeError, ValueError):
                plan[k] = None
        return plan

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, plan: dict, phase: str | None, pantry: set[str] | None = None,
                 staples: set[str] | None = None, k: int = N_CONTEXT) -> list[int]:
        rec = self.R.recipes
        sim = self.E @ self.embed(plan.get("search_query") or "")
        ok = np.ones(len(rec), bool)
        if plan.get("meal_types"):
            ok &= rec["meal_type"].isin(plan["meal_types"]).to_numpy()
        if plan.get("vegetarian"):
            ok &= self.veg_ok
        if plan.get("max_minutes"):
            ok &= rec["TotalTime_min"].to_numpy() <= plan["max_minutes"]
        if plan.get("max_calories"):
            ok &= rec["Calories"].to_numpy() <= plan["max_calories"]
        for x in plan.get("exclude", []):
            ok &= np.array([not any(x in p for p in parts) for parts in rec["parts_norm"]])
        # ingredients she has or wants: recipes with all of them if there are enough, else with most of them
        inc = plan.get("include") or []
        if inc:
            n_has = np.array([sum(any(x in p for p in parts) for x in inc) for parts in rec["parts_norm"]])
            for need in range(len(inc), 0, -1):
                if (ok & (n_has >= need)).sum() >= 3:
                    ok &= n_has >= need
                    break
        if not ok.any():
            return []
        sim = np.where(ok, sim, -np.inf)
        cand = np.argsort(-sim)[:N_CANDIDATES]
        cand = cand[np.isfinite(sim[cand])]
        score = sim[cand] + BONUS_POPULAR * self.R.pop_rank[cand]
        if phase:
            score += BONUS_PHASE * rec[f"label_{phase}"].to_numpy()[cand]
        if pantry:
            share, _ = self.R.pantry_match(pantry, True, staples)
            score += BONUS_PANTRY * share[cand]
        return [int(i) for i in cand[np.argsort(-score)][:k]]

    def recipe_doc(self, idx: int, phase: str | None, have: set[str] | None) -> str:
        r = self.R.recipes.iloc[idx]
        steps = [str(s) for s in r["instructions"]]
        steps_txt = " ".join(f"{i}. {s}" for i, s in enumerate(steps[:15], 1))[:1500]
        lines = [f"id: {int(r['RecipeId'])}",
                 f"name: {r['Name']}",
                 f"meal type: {r['meal_type']}; time: {int(r['TotalTime_min'])} min; "
                 f"calories: {r['Calories']:.0f} per serving; servings: {r['servings_final']:.0f}",
                 "ingredients: " + "; ".join(str(x) for x in r["ingredients_display"]),
                 "steps: " + steps_txt]
        if phase:
            fit = bool(r[f"label_{phase}"])
            lines.append(f"fits the {phase} phase: {'yes' if fit else 'no'}"
                         + (f" (rich in {r[f'why_{phase}']} of daily value)" if fit and r[f"why_{phase}"] else ""))
        if have:
            missing = sorted(set(r["parts_norm"]) - have - ALWAYS_THERE)
            lines.append("missing from her kitchen: " + (", ".join(missing) if missing else "nothing"))
        return "\n".join(lines)

    def answer(self, client, question: str, history: list[dict], idxs: list[int], phase: str | None,
               focus: str, have: set[str] | None) -> tuple[str, list[int]]:
        system = ANSWER_PROMPT.replace("PHASE", phase or "unknown").replace("FOCUS", focus or "a balanced diet")
        context = "\n\n---\n\n".join(self.recipe_doc(i, phase, have) for i in idxs) or "No matching recipes found."
        msgs = ([{"role": "system", "content": system}] + history[-MAX_HISTORY:]
                + [{"role": "user", "content": f"Recipes:\n{context}\n\nQuestion: {question}"}])
        text = self._chat(client, msgs)
        # which recipes to show as cards: the RECIPES line, else names mentioned in the text
        shown = []
        m = re.search(r"^\s*RECIPES:\s*(.*)$", text, re.M | re.I)
        if m:
            text = (text[:m.start()] + text[m.end():]).strip()
            ids = [int(x) for x in re.findall(r"\d+", m.group(1))]          # keep the order of the answer
            by_id = {int(self.R.recipes.iloc[i]["RecipeId"]): i for i in idxs}
            shown = [by_id[r] for r in dict.fromkeys(ids) if r in by_id]
        if not shown:
            low = text.lower()
            found = [(low.find(str(self.R.recipes.iloc[i]["Name"]).lower()), i) for i in idxs]
            shown = [i for pos, i in sorted(found) if pos >= 0]
        return text, shown[:3]

    def small_talk(self, client, question: str, history: list[dict], kind: str, phase: str | None,
                   focus: str, name: str) -> str:
        system = (SMALLTALK_PROMPT.replace("NAME", name or "unknown").replace("PHASE", phase or "unknown")
                  .replace("FOCUS", focus or "a balanced diet").replace("KIND", kind))
        msgs = [{"role": "system", "content": system}] + history[-MAX_HISTORY:] + [{"role": "user", "content": question}]
        try:
            return self._chat(client, msgs).strip()
        except Exception:
            return "Hi! 🌿 What would you like to eat today?"

    # ------------------------------------------------------------------ one turn
    def ask(self, client, question: str, history: list[dict], last_idxs: list[int], phase: str | None,
            focus: str = "", pantry: set[str] | None = None, staples: set[str] | None = None, name: str = "") -> dict:
        plan = self.understand(client, question, history)
        intent = str(plan.get("intent") or "recipes").lower()
        if plan.get("off_topic"):                     # older plan format
            intent = "off_topic"
        if intent in ("chitchat", "off_topic", "unclear"):
            text = self.small_talk(client, question, history, intent, phase, focus, name)
            return {"text": text, "shown": [], "context": last_idxs, "plan": plan}
        if (intent == "followup" or plan.get("followup")) and last_idxs:
            idxs = last_idxs
        else:
            idxs = self.retrieve(plan, phase, pantry, staples)
        have = (set(pantry or []) | set(staples or [])) if pantry else None
        text, shown = self.answer(client, question, history, idxs, phase, focus, have)
        return {"text": text, "shown": shown, "context": idxs or last_idxs, "plan": plan}
