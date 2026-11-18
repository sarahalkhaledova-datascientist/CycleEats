"""CycleEats: recipe suggestions for your cycle phase, your taste and your kitchen.

Run locally:  streamlit run app/app.py
"""
from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import recommenders
from recommenders import PHASE_LABELS, UNITS, OwnRecipe, Recommender, cycle_phase

st.set_page_config(page_title="CycleEats", page_icon="🌿", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
:root { --ce-text: #2b2230; --ce-muted: #6b6470; }
.block-container { padding-top: 1.4rem; max-width: 1200px; }
[data-testid="stHeader"] { background: transparent; }
.ce-hello { font-size: 2.1rem !important; line-height: 1.2; font-weight: 700; color: var(--ce-accent); margin: 0; }
.ce-sub { color: var(--ce-muted); margin: 0.1rem 0 0; font-size: 1.1rem !important; }
.ce-phase { background: #ffffff; border-left: 6px solid var(--ce-accent); color: var(--ce-text);
            border-radius: 12px; padding: 0.85rem 1.1rem; margin: 0.8rem 0 1rem; }
.ce-phase .ce-phase-name { color: var(--ce-accent); font-weight: 700; font-size: 1.1rem; }
.ce-phase .ce-phase-why { color: var(--ce-muted); font-size: 0.9rem; margin-top: 0.35rem; }
.ce-badge { display: inline-block; background: var(--ce-soft); color: var(--ce-accent); border-radius: 999px;
            padding: 0.1rem 0.6rem; font-size: 0.8rem; font-weight: 600; margin: 0 0.3rem 0.3rem 0; }
.ce-badge.own { background: #fbeedd; color: #8a5a12; }
.ce-meta { color: var(--ce-muted); font-size: 0.85rem; }
.ce-why { color: var(--ce-accent); font-size: 0.85rem; }
.ce-missing { color: #9a5b00; font-size: 0.85rem; }
.ce-have { color: #2f6b4f; } .ce-lack { color: #9a5b00; }
.ce-noimg { height: 150px; border-radius: 10px; background: var(--ce-soft);
            display: flex; align-items: center; justify-content: center; font-size: 2.6rem; }
.ce-section-title { font-size: 1.4rem !important; font-weight: 700; color: var(--ce-text); margin: 0.2rem 0 0; }
.ce-step { color: var(--ce-accent); font-weight: 700; letter-spacing: 0.04em; font-size: 0.85rem; text-transform: uppercase; }
.ce-wiz-title { font-size: 1.7rem !important; line-height: 1.25; font-weight: 700; color: var(--ce-text); margin: 0.2rem 0 0.2rem; }
.ce-photo { width: 100%; aspect-ratio: 3 / 2; object-fit: cover; border-radius: 10px; display: block; }
.ce-credit { color: var(--ce-muted); font-size: 0.68rem; margin: 0.15rem 0 0.4rem; }
.ce-credit a { color: var(--ce-muted) !important; text-decoration: underline; }
.ce-foot { color: var(--ce-muted); font-size: 0.8rem; margin-top: 2rem; }
[class*="st-key-card_"], .st-key-wizard, .st-key-about_box { background: #ffffff; }
.st-key-wizard { border-radius: 18px; padding: 1.6rem 1.8rem; box-shadow: 0 4px 18px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading recipes…")
def get_recommender(code_version: str):
    # reload the module too: Streamlit Cloud keeps the old recommenders.py in memory after a push
    importlib.reload(recommenders)
    return recommenders.Recommender()


def code_version() -> str:
    """Changes when recommenders.py or the photo list changes, so the cached recommender is rebuilt."""
    h = hashlib.md5(Path(recommenders.__file__).read_bytes())
    photos = Path(recommenders.DATA_DIR) / "photos.csv"
    if photos.exists():
        h.update(photos.read_bytes())
    return h.hexdigest()


# the cache is rebuilt whenever the code or photos change (otherwise an update keeps the old object)
R = get_recommender(code_version())
MIN_ONBOARD = 3            # recipes to like before leaving the start screen
PAGE = 12                  # suggestions per page ("Show more" adds another page)
MAX_SHOWN = 48
N_RECENT = 5
SEARCH_LIMIT = 24
CFG = R.config
WEIGHT_LABELS = {"content": "Similar ingredients (Content-Based)", "item_cf": "Recipes liked together (Item-CF)",
                 "user_cf": "People with your taste (User-CF)", "popularity": "Popular overall",
                 "phase": "Fit for my cycle phase", "pantry": "Uses what I have"}
PHASE_THEME = {   # accent, page background, soft tint, sidebar
    "menstrual":  ("#9b2c5a", "#fbf0f4", "#f5dbe6", "#f6e4ec"),
    "follicular": ("#2e7d5b", "#eff8f3", "#d6eee1", "#e2f2e9"),
    "ovulatory":  ("#c4573a", "#fdf3ee", "#f8ddd1", "#f9e6dd"),
    "luteal":     ("#6b4fa0", "#f4f0fa", "#e3daf3", "#ebe4f6"),
    None:         ("#7a3e65", "#faf6f1", "#f1e4ec", "#f3ebe3"),   # set-up screens, before the phase is known
}
PHASE_WHY = {
    "menstrual": "Why: women need more iron than men (18 mg vs 8 mg a day) and periods lose iron. "
                 "Vitamin C helps your body absorb iron from plant foods.",
    "follicular": "Why: no nutrient has strong evidence for this phase, so we aim for a balanced plate. "
                  "Zinc has some evidence for cycle symptoms; protein, fibre and folate support overall health.",
    "ovulatory": "Why: no nutrient has strong evidence for this phase, so we aim for a balanced plate. "
                 "Zinc has some evidence for cycle symptoms; protein, fibre and folate support overall health.",
    "luteal": "Why: calcium and vitamin B6 have the best evidence for easing PMS symptoms. "
              "Magnesium may help too, but the evidence is mixed.",
}
ENCOURAGE = {
    "menstrual": ["Be gentle with yourself today.", "Rest counts as progress.", "A warm meal is a small act of care.",
                  "Listen to your body, it knows what it needs.", "Slow mornings are allowed.",
                  "You're doing better than you think.", "Cosy food, cosy evening.",
                  "Take the easy recipe today. That's fine.", "Drink some water and breathe.",
                  "Your body is working hard. Feed it well.", "Comfort and nourishment can go together.",
                  "One kind thing for yourself today is enough."],
    "follicular": ["Energy is coming back. Enjoy it!", "A great time to try a new recipe.", "Fresh start, fresh plate.",
                   "Small good choices add up.", "Feeling curious? Cook something new.",
                   "Your energy is rising, so ride the wave.", "Plan a few meals while you feel motivated.",
                   "Green, fresh and simple works today.", "Today is a good day to start something.",
                   "Bring a friend into the kitchen.", "Bold flavours, bright mood.", "You've got this!"],
    "ovulatory": ["You're glowing. Keep it going!", "Share a meal with someone you like today.",
                  "Feed the energy you have.", "Bright days call for colourful plates.",
                  "Good day for a dinner with friends.", "You're at your best, so treat yourself well.",
                  "Say yes to something fun today.", "Light, fresh and full of colour.",
                  "Your confidence looks good on you.", "Celebrate the small wins.",
                  "Cook something you're proud of.", "Keep the momentum going!"],
    "luteal": ["Slow down a little, you deserve it.", "Comfort food can be nourishing too.", "Steady meals, steady mood.",
               "Be kind to yourself this week.", "Cravings are normal. Be kind to them.",
               "Regular meals help keep your mood steady.", "It's okay to say no and rest.",
               "Warm bowls and early nights.", "You don't have to do it all today.",
               "A little planning now saves stress later.", "Treat yourself, and add some greens.",
               "You're stronger than this week feels."],
}
PHASE_INFO = {
    "menstrual": "Focus on <b>iron</b> and <b>vitamin C</b> (helps iron absorption).",
    "follicular": "Focus on <b>zinc</b>, <b>protein</b>, <b>fibre</b> and <b>folate</b>.",
    "ovulatory": "Focus on <b>zinc</b>, <b>protein</b>, <b>fibre</b> and <b>folate</b>.",
    "luteal": "Focus on <b>calcium</b> and <b>vitamin B6</b>, plus magnesium.",
}

# ----------------------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    ss.setdefault("user_name", "")
    ss.setdefault("setup_step", 0)                # 0 name · 1 cycle · 2 recipes
    ss.setdefault("hearts", [])
    ss.setdefault("recent", [])                   # recently opened catalogue recipes, newest first
    ss.setdefault("n_shown", PAGE)
    ss.setdefault("own", [])                      # list of dicts (serialisable)
    ss.setdefault("onboarding_done", False)
    ss.setdefault("last_start", date.today() - timedelta(days=3))
    ss.setdefault("cycle_len", 28)
    ss.setdefault("period_len", 5)
    ss.setdefault("manual_phase", False)
    ss.setdefault("phase_pick", "menstrual")
    ss.setdefault("pantry", [])
    ss.setdefault("use_staples", True)
    ss.setdefault("only_cookable", False)
    ss.setdefault("max_missing", int(CFG["max_missing_default"]))
    ss.setdefault("meal_types", list(CFG["meal_type_default"]))
    for k, v in R.weights_default.items():
        ss.setdefault(f"w_{k}", float(v))


init_state()
ss = st.session_state
# keep values when their widget is not on screen (e.g. set-up pages vs. sidebar)
for _k in ["user_name", "last_start", "cycle_len", "period_len", "manual_phase", "phase_pick", "pantry", "use_staples",
           "only_cookable", "max_missing", "meal_types"] + [f"w_{k}" for k in R.weights_default]:
    if _k in ss:
        ss[_k] = ss[_k]


def toggle_heart(rid: int):
    if rid in ss.hearts:
        ss.hearts.remove(rid)
    else:
        ss.hearts.append(rid)


def own_objects() -> list[OwnRecipe]:
    out = []
    for d in ss.own:
        o = OwnRecipe(name=d["name"], servings=d["servings"], ingredients=[tuple(x) for x in d["ingredients"]],
                      meal_type=d.get("meal_type", "Main"), instructions=d.get("instructions", ""), id=d["id"])
        out.append(R.analyse_own(o))
    return out


def export_data() -> str:
    keys = ["user_name", "hearts", "recent", "own", "last_start", "cycle_len", "period_len", "manual_phase", "phase_pick", "pantry",
            "use_staples", "only_cookable", "max_missing", "meal_types", "onboarding_done"] + [f"w_{k}" for k in R.weights_default]
    data = {k: ss[k] for k in keys}
    data["last_start"] = ss.last_start.isoformat()
    data["hearts"] = [int(h) for h in ss.hearts]
    data["recent"] = [int(h) for h in ss.recent]
    return json.dumps({"app": "CycleEats", "version": 1, "data": data}, indent=2)


def import_data(raw: bytes):
    payload = json.loads(raw.decode("utf-8"))
    data = payload.get("data", payload)
    for k, v in data.items():
        if k == "last_start":
            v = date.fromisoformat(v)
        ss[k] = v
    if len(ss.get("hearts", [])) >= MIN_ONBOARD:
        ss["onboarding_done"] = True


def reset_weights():
    for k, v in R.weights_default.items():
        ss[f"w_{k}"] = float(v)


def show_more():
    ss.n_shown = min(ss.n_shown + PAGE, MAX_SHOWN)


def clear_search():
    ss.search_q = ""


def restore_from_upload():
    up = ss.get("restore_file")
    if up is None:
        return
    try:
        import_data(up.getvalue())
        ss["_restore_msg"] = "Data restored."
    except Exception as e:                          # noqa: BLE001
        ss["_restore_msg"] = f"Could not read this file: {e}"


# ----------------------------------------------------------------------------- sidebar
def redo_setup():
    ss.onboarding_done = False
    ss.setup_step = 0


if ss.onboarding_done:
  with st.sidebar:
    st.header("👤 You")
    st.text_input("Your name", key="user_name")
    st.button("Redo set-up", on_click=redo_setup)

    st.header("🌙 Your cycle")
    if not ss.manual_phase:
        st.date_input("First day of your last period", key="last_start", max_value=date.today())
        c1, c2 = st.columns(2)
        c1.number_input("Cycle length (days)", 21, 40, key="cycle_len")
        c2.number_input("Period length (days)", 2, 9, key="period_len")
    st.toggle("Set my phase manually", key="manual_phase")
    if ss.manual_phase:
        st.selectbox("Phase", R.phases, key="phase_pick", format_func=lambda p: PHASE_LABELS[p])

    st.header("🧺 Your kitchen")
    st.checkbox("Basics are always at home", key="use_staples",
                help="Salt, pepper, oil, butter, sugar, flour, baking powder, garlic and common dried spices "
                     "count as available, so you don't need to pick them.")
    if ss.use_staples:   # basics are hidden from the list (keep any already picked, so the widget stays valid)
        pantry_options = [n for n in R.ingredient_names() if n not in R.staples or n in ss.pantry]
    else:
        pantry_options = R.ingredient_names()
    st.multiselect("Other ingredients you have", pantry_options, key="pantry",
                   placeholder="Type to search, e.g. chicken breast")
    if ss.use_staples:
        with st.expander(f"Basics counted automatically ({len(R.staples)})"):
            st.caption(", ".join(sorted(R.staples)))
    st.toggle("Only recipes I can cook now", key="only_cookable", disabled=not ss.pantry)
    if ss.only_cookable and ss.pantry:
        st.slider("Missing ingredients allowed", 0, 5, key="max_missing")

    st.header("🍽️ Meal type")
    st.multiselect("Show", CFG["meal_types"], key="meal_types", label_visibility="collapsed")

    with st.expander("⚙️ Advanced: how recommendations are blended"):
        st.caption("Defaults come from the offline evaluation (notebook 09).")
        for k in ["content", "item_cf", "user_cf", "popularity", "phase", "pantry"]:
            st.slider(WEIGHT_LABELS[k], 0.0, 1.0, step=0.05, key=f"w_{k}")
        st.button("Reset to defaults", on_click=reset_weights)

    st.header("💾 My data")
    st.caption("Nothing is stored on a server. Save a file to keep your favourites, recipes and cycle settings.")
    st.download_button("Save my data", export_data(), file_name="cycleeats_my_data.json", mime="application/json",
                       width="stretch")
    st.file_uploader("Restore my data", type="json", label_visibility="collapsed", key="restore_file",
                     on_change=restore_from_upload)
    if ss.get("_restore_msg"):
        (st.success if ss["_restore_msg"].startswith("Data") else st.error)(ss["_restore_msg"])

# ----------------------------------------------------------------------------- phase
if ss.manual_phase:
    phase, cycle_day = ss.phase_pick, None
else:
    phase, cycle_day = cycle_phase(ss.last_start, int(ss.cycle_len), int(ss.period_len))
weights = {k: ss[f"w_{k}"] for k in R.weights_default}



def apply_theme(ph):
    accent, bg, soft, side = PHASE_THEME[ph]
    st.markdown(f"""<style>
:root {{ --ce-accent: {accent}; --ce-soft: {soft}; }}
[data-testid="stApp"] {{ background: {bg}; }}
[data-testid="stSidebar"] {{ background: {side}; }}
button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background: {accent} !important; border-color: {accent} !important; color: #fff !important; }}
span[data-baseweb="tag"] {{ background: {accent} !important; }}
[data-baseweb="tab-highlight"] {{ background: {accent} !important; }}
button[role="tab"][aria-selected="true"] p {{ color: {accent} !important; }}
[data-testid="stProgress"] [role="progressbar"] div div div {{ background: {accent} !important; }}
</style>""", unsafe_allow_html=True)


def greeting() -> str:
    hour = None
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        tz = st.context.timezone
        hour = datetime.now(ZoneInfo(tz)).hour if tz else None
    except Exception:                               # noqa: BLE001
        pass
    part = ("Good morning" if hour is not None and 5 <= hour < 12 else "Good afternoon" if hour is not None and 12 <= hour < 18
            else "Good evening" if hour is not None else "Hi")
    return f"{part}, {ss.user_name.strip()} 🌿" if ss.user_name.strip() else f"{part} 🌿"


apply_theme(phase if ss.onboarding_done else (phase if ss.setup_step >= 2 else None))



# ----------------------------------------------------------------------------- cards
HAVE = R.pantry_match(set(ss.pantry), ss.use_staples)[1] if ss.pantry else set()


def section_title(title: str, sub: str = ""):
    st.markdown(f'<p class="ce-section-title">{title}</p>', unsafe_allow_html=True)
    if sub:
        st.caption(sub)


UTM = "?utm_source=cycleeats&utm_medium=referral"


def photo_html(ph: dict, w: int = 600, h: int = 400, note: str = "") -> str:
    """Unsplash photo (hotlinked) with the credit the Unsplash guidelines ask for."""
    url = f'{ph["url_raw"]}&w={w}&h={h}&fit=crop&auto=format&q=75'
    alt = str(ph.get("alt") or "food photo").replace('"', "'")
    credit = (f'{note}Photo by <a href="{ph["author_url"]}{UTM}" target="_blank">{ph["author"]}</a> on '
              f'<a href="https://unsplash.com/{UTM}" target="_blank">Unsplash</a>')
    return f'<img class="ce-photo" src="{url}" alt="{alt}" loading="lazy"><div class="ce-credit">{credit}</div>'


def nutrient_bars(pdv: dict, n: int = 6):
    top = sorted(pdv.items(), key=lambda kv: -kv[1])[:n]
    for k, v in top:
        st.progress(min(float(v) / 100, 1.0), text=f"{R.dv.loc[k, 'name']}: {v:.0f}% of daily value")


@st.dialog("Recipe", width="large")
def recipe_dialog(item: dict):
    if item.get("photo"):
        c_img, c_head = st.columns([2, 3])
        c_img.markdown(photo_html(item["photo"], 720, 480, note="For illustration. "), unsafe_allow_html=True)
        head = c_head
    else:
        head = st.container()
    head.markdown(f"### {item['name']}")
    head.markdown(f'<span class="ce-meta">{item["meta"]}</span>', unsafe_allow_html=True)
    if item.get("badges"):
        head.markdown(item["badges"], unsafe_allow_html=True)
    if item.get("why"):
        head.markdown(f'<span class="ce-why">Rich in: {item["why"]} of daily value</span>', unsafe_allow_html=True)
    if item.get("description"):
        head.caption(item["description"])
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown("**Ingredients**")
        lines = []
        for line in item["ingredients"]:
            if HAVE:
                low = line.lower()
                ok = any(h in low for h in HAVE)
                lines.append(f'<span class="{"ce-have" if ok else "ce-lack"}">{"✓" if ok else "✗"}</span> {line}')
            else:
                lines.append(f"• {line}")
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
        if item.get("unit_note"):
            st.caption(item["unit_note"])
    with c2:
        st.markdown("**Steps**")
        if item["steps"]:
            st.markdown("\n".join(f"{i}. {x}" for i, x in enumerate(item["steps"], 1)))
        else:
            st.caption("No steps written.")
    st.markdown("**Nutrients per serving**")
    nutrient_bars(item["pdv"])


def open_recipe(row):
    meta = card_meta(row)
    if row.get("is_own"):
        o = next(x for x in own_objects() if x.id == row["id"])
        item = {"name": o.name, "meta": meta, "badges": '<span class="ce-badge own">My recipe</span>',
                "why": o.info["why"].get(phase, ""), "description": "", "photo": R.photo(o.name, o.meal_type, o.id),
                "ingredients": [f"{q:g} {u} · {n}" for n, q, u in o.ingredients],
                "steps": [x for x in o.instructions.split("\n") if x.strip()], "pdv": o.info["pdv"]}
    else:
        rid = int(row["id"])
        ss.recent = ([rid] + [r for r in ss.recent if r != rid])[:N_RECENT]
        rec = R.recipe(rid)
        badge = (f'<span class="ce-badge">Good for your {PHASE_LABELS[phase].lower()} phase</span>'
                 if rec[f"label_{phase}"] else "")
        item = {"name": rec["Name"], "meta": meta, "badges": badge, "why": rec[f"why_{phase}"],
                "description": rec["description"], "photo": R.photo(rec["Name"], rec["meal_type"], rid),
                "ingredients": list(rec["ingredients_display"]), "steps": list(rec["instructions"]),
                "unit_note": "Food.com lists quantities without units (e.g. 2 · butter).",
                "pdv": {n: float(rec[f"pdv_{n}"]) for n in R.dv.index}}
    recipe_dialog(item)


def card_meta(row) -> str:
    meta = [row["meal_type"]]
    if row.get("minutes") is not None and pd.notna(row.get("minutes")):
        meta.append(f"{int(row['minutes'])} min")
    if pd.notna(row.get("calories")):
        meta.append(f"{int(row['calories'])} kcal/serving")
    return " · ".join(meta)


def show_card(row, key_prefix: str):
    with st.container(border=True, key=f"card_{key_prefix}_{str(row['id']).replace(':', '_')}"):
        ph = R.photo(row["name"], row["meal_type"], row["id"])
        if ph:
            st.markdown(photo_html(ph), unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ce-noimg">{"📝" if row.get("is_own") else "🍽️"}</div>', unsafe_allow_html=True)
        st.markdown(f"**{row['name']}**")
        st.markdown(f'<span class="ce-meta">{card_meta(row)}</span>', unsafe_allow_html=True)
        badges = ""
        if row.get("is_own"):
            badges += '<span class="ce-badge own">My recipe</span>'
        if row.get("phase_fit"):
            badges += f'<span class="ce-badge">Good for your {PHASE_LABELS[phase].lower()} phase</span>'
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
        if row.get("match") is not None:
            st.progress(float(row["match"]), text=f"You have {row['match']:.0%} of the ingredients")
            if row.get("missing"):
                more = len(row["missing"]) - 4
                st.markdown(f'<span class="ce-missing">Missing: {", ".join(row["missing"][:4])}'
                            f'{f" +{more} more" if more > 0 else ""}</span>', unsafe_allow_html=True)
        if not row.get("is_own"):
            rid = int(row["id"])
            liked = rid in ss.hearts
            st.button("♥ Liked" if liked else "♡ I like this", key=f"{key_prefix}_h_{rid}", on_click=toggle_heart,
                      args=(rid,), type="primary" if liked else "secondary", width="stretch")
        if st.button("View recipe", key=f"{key_prefix}_v_{row['id']}", width="stretch"):
            open_recipe(row)


def show_grid(df: pd.DataFrame, key_prefix: str, n_cols: int = 3):
    for start in range(0, len(df), n_cols):
        cols = st.columns(n_cols)
        for c, (_, row) in zip(cols, df.iloc[start:start + n_cols].iterrows()):
            with c:
                show_card(row, key_prefix)


# ----------------------------------------------------------------------------- set-up (first visit)
def go(step: int):
    ss.setup_step = step


def finish_setup():
    ss.onboarding_done = True


if not ss.onboarding_done:
    step = ss.setup_step
    st.markdown('<p class="ce-hello">CycleEats 🌿</p>', unsafe_allow_html=True)
    st.markdown('<p class="ce-sub">Recipes for your cycle phase, your taste and your kitchen.</p>', unsafe_allow_html=True)
    st.progress((step + 1) / 3, text=f"Step {step + 1} of 3")
    if step in (0, 1):
        _, mid, _ = st.columns([1, 2, 1])
        with mid, st.container(key="wizard"):
            if step == 0:
                st.markdown('<p class="ce-step">Welcome</p><p class="ce-wiz-title">What should we call you?</p>',
                            unsafe_allow_html=True)
                st.text_input("Your name", key="user_name", placeholder="e.g. Sarah", label_visibility="collapsed")
                st.caption("Only used to greet you. It stays in this browser session.")
                st.button("Next →", type="primary", on_click=go, args=(1,), width="stretch")
            else:
                hi = f", {ss.user_name.strip()}" if ss.user_name.strip() else ""
                st.markdown(f'<p class="ce-step">Your cycle</p><p class="ce-wiz-title">Nice to meet you{hi}!<br>'
                            'When did your last period start?</p>', unsafe_allow_html=True)
                if not ss.manual_phase:
                    st.date_input("First day of your last period", key="last_start", max_value=date.today())
                    c1, c2 = st.columns(2)
                    c1.number_input("Cycle length (days)", 21, 40, key="cycle_len", help="Usually 21 to 35 days. 28 if unsure.")
                    c2.number_input("Period length (days)", 2, 9, key="period_len")
                st.checkbox("I'm not sure, let me choose my phase", key="manual_phase")
                if ss.manual_phase:
                    st.selectbox("Phase", R.phases, key="phase_pick", format_func=lambda p: PHASE_LABELS[p])
                ph_now, day_now = ((ss.phase_pick, None) if ss.manual_phase
                                   else cycle_phase(ss.last_start, int(ss.cycle_len), int(ss.period_len)))
                acc, _, soft, _ = PHASE_THEME[ph_now]
                st.markdown(f'<div style="background:{soft};color:{acc};border-radius:12px;padding:0.7rem 1rem;font-weight:600">'
                            f"{'Day ' + str(day_now) + ' · ' if day_now else ''}You're in your {PHASE_LABELS[ph_now].lower()} phase</div>",
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.button("← Back", on_click=go, args=(0,), width="stretch")
                c2.button("Next →", type="primary", on_click=go, args=(2,), width="stretch")
    else:
        n_liked = len(ss.hearts)
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            st.markdown('<p class="ce-step">Your taste</p><p class="ce-wiz-title">Pick at least 3 recipes you like</p>',
                        unsafe_allow_html=True)
            st.caption("Tap ♡ on the ones you like. Your suggestions will learn your taste from them.")
            st.progress(min(n_liked / MIN_ONBOARD, 1.0), text=f"{n_liked} of {MIN_ONBOARD} liked")
        with c2:
            st.button("Continue →", type="primary", width="stretch", disabled=n_liked < MIN_ONBOARD, on_click=finish_setup)
            st.button("Skip", width="stretch", on_click=finish_setup)
            st.button("← Back", width="stretch", on_click=go, args=(1,))
        ob = R.onboarding()
        df = pd.DataFrame({"id": ob["RecipeId"], "name": ob["Name"], "meal_type": ob["meal_type"],
                           "minutes": ob["TotalTime_min"], "calories": ob["Calories"], "image_url": ob["image_url"],
                           "is_own": False, "phase_fit": False, "why": "", "match": None, "missing": [[]] * len(ob)})
        show_grid(df, "ob", n_cols=4)
    st.stop()

# ----------------------------------------------------------------------------- main page header
day_txt = f"Day {cycle_day} · " if cycle_day else ""
lines = ENCOURAGE[phase]
st.markdown(f'<p class="ce-hello">{greeting()}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="ce-sub">{lines[date.today().toordinal() % len(lines)]}</p>', unsafe_allow_html=True)
st.markdown(f'<div class="ce-phase"><span class="ce-phase-name">{day_txt}{PHASE_LABELS[phase]} phase</span><br>'
            f'{PHASE_INFO[phase]}<div class="ce-phase-why">{PHASE_WHY[phase]}</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- tabs
tab_for_you, tab_fav, tab_own, tab_about = st.tabs(["✨ For you", "♥ Favourites", "📝 My recipes", "ℹ️ About"])

with tab_for_you:
  if True:
    with st.container(key="sec_foryou"):
        basis = [f"{len(ss.hearts)} recipe{'s' if len(ss.hearts) != 1 else ''} you like" if ss.hearts else "popular recipes (no likes yet)",
                 f"your {PHASE_LABELS[phase].lower()} phase"]
        if ss.pantry:
            basis.append("your kitchen")
        section_title("What would you like to eat today?",
                      "Based on " + ", ".join(basis) + ". Tap ♡ on recipes you like and the list learns your taste. "
                      "It updates automatically when you change anything in the sidebar.")
        c_q, c_x = st.columns([6, 1], vertical_alignment="bottom")
        query = c_q.text_input("Search recipes", key="search_q", placeholder="🔍 Search a recipe by name, e.g. banana bread",
                               label_visibility="collapsed")
        if query.strip():
            c_x.button("Clear", on_click=clear_search, width="stretch")
            found = R.search(query, phase=phase, limit=SEARCH_LIMIT)
            if found.empty:
                st.info(f"No recipes found for “{query.strip()}”. Try another word.")
            else:
                st.caption(f"{len(found)}{'+' if len(found) == SEARCH_LIMIT else ''} recipes match “{query.strip()}”. "
                           "Recipes that fit your phase come first among similar matches.")
                show_grid(found, "srch")
        elif not ss.meal_types:
            st.warning("Pick at least one meal type in the sidebar.")
        else:
            if len(ss.hearts) < MIN_ONBOARD:
                st.info(f"{len(ss.hearts)} liked so far. Like a few more for more personal suggestions.")
            # back to one page whenever a setting changes (liking a recipe does not count)
            sig = (phase, tuple(sorted(ss.pantry)), tuple(ss.meal_types), ss.only_cookable, int(ss.max_missing),
                   ss.use_staples, tuple(sorted(weights.items())))
            if ss.get("_rec_sig") != sig:
                ss._rec_sig, ss.n_shown = sig, PAGE
            recs = R.recommend(hearts=ss.hearts, phase=phase, pantry=set(ss.pantry), meal_types=ss.meal_types,
                               own_recipes=own_objects(), weights=weights, only_cookable=ss.only_cookable,
                               max_missing=int(ss.max_missing), use_staples=ss.use_staples, k=ss.n_shown, page=PAGE)
            if recs.empty:
                st.warning("No recipes match. Allow more missing ingredients or add meal types.")
            else:
                show_grid(recs, "rec")
                if len(recs) >= ss.n_shown and ss.n_shown < MAX_SHOWN:
                    _, mid, _ = st.columns([1, 2, 1])
                    mid.button(f"Show more ({ss.n_shown} of {MAX_SHOWN} shown)", on_click=show_more, width="stretch")

with tab_fav, st.container(key="sec_fav"):
    section_title("Recipes you like", "Everything you tapped ♡ on. Tap ♥ again to remove one.")
    if not ss.hearts:
        st.info("No favourites yet. Tap ♡ I like this on any recipe.")
    else:
        fav = R.recipes.set_index("RecipeId").loc[[h for h in ss.hearts if h in R.idx.index]].reset_index()
        df = pd.DataFrame({"id": fav["RecipeId"], "name": fav["Name"], "meal_type": fav["meal_type"],
                           "minutes": fav["TotalTime_min"], "calories": fav["Calories"], "image_url": fav["image_url"],
                           "is_own": False, "phase_fit": fav[f"label_{phase}"], "why": fav[f"why_{phase}"],
                           "match": None, "missing": [[]] * len(fav)})
        show_grid(df, "fav")

    recent = [r for r in ss.recent if r in R.idx.index]
    if recent:
        st.markdown("")
        section_title("Recently viewed", "The last recipes you opened, in case you forgot to tap ♡.")
        rv = R.recipes.set_index("RecipeId").loc[recent].reset_index()
        show_grid(pd.DataFrame({"id": rv["RecipeId"], "name": rv["Name"], "meal_type": rv["meal_type"],
                                "minutes": rv["TotalTime_min"], "calories": rv["Calories"], "image_url": rv["image_url"],
                                "is_own": False, "phase_fit": rv[f"label_{phase}"], "why": rv[f"why_{phase}"],
                                "match": None, "missing": [[]] * len(rv)}), "recent", n_cols=5)

with tab_own, st.container(key="sec_own"):
    section_title("My recipes", "Your own recipes get the same nutrient and cycle-phase check as the catalogue.")
    st.markdown("**Add a recipe**")
    st.caption("Search each ingredient in the list so its nutrients are counted. If it is not in the list, "
               "type it anyway: it is kept but not counted.")
    names = R.ingredient_names()
    ss.setdefault("own_rows", 5)
    ss.setdefault("own_form_id", 0)          # changes after saving, so all fields start empty again
    fid = ss.own_form_id
    c1, c2, c3 = st.columns([3, 1, 2])
    o_name = c1.text_input("Recipe name", key=f"o_name_{fid}")
    o_serv = c2.number_input("Servings", 1, 20, 2, key=f"o_serv_{fid}")
    o_meal = c3.selectbox("Meal type", [m for m in CFG["meal_types"] if m != "Other"], key=f"o_meal_{fid}")
    h1, h2, h3 = st.columns([4, 1, 1])
    h1.caption("Ingredient"); h2.caption("Quantity"); h3.caption("Unit")
    rows = []
    for r in range(ss.own_rows):
        c1, c2, c3 = st.columns([4, 1, 1])
        ing = c1.selectbox("Ingredient", names, index=None, key=f"o_ing_{fid}_{r}", accept_new_options=True,
                           placeholder="Search, e.g. lentil", label_visibility="collapsed")
        qty = c2.number_input("Quantity", 0.0, 5000.0, 1.0, 0.25, key=f"o_qty_{fid}_{r}", label_visibility="collapsed")
        default_unit = R.ingredients.at[ing, "default_unit"] if ing in R.ingredients.index else "cup"
        unit = c3.selectbox("Unit", UNITS, index=UNITS.index(default_unit) if default_unit in UNITS else 2,
                            key=f"o_unit_{fid}_{r}_{ing}", label_visibility="collapsed")
        if ing:
            rows.append((str(ing).strip().lower(), float(qty), unit))
    if st.button("＋ Add ingredient"):
        ss.own_rows += 1
        st.rerun()
    o_steps = st.text_area("Steps (optional, one per line)", key=f"o_steps_{fid}")
    if st.button("Save recipe", type="primary"):
        if not o_name or not rows:
            st.error("Add a name and at least one ingredient.")
        else:
            d = {"id": f"own:{uuid.uuid4().hex[:8]}", "name": o_name, "servings": o_serv, "meal_type": o_meal,
                 "ingredients": rows, "instructions": o_steps}
            info = R.analyse_own(OwnRecipe(name=o_name, servings=o_serv, ingredients=rows)).info
            ss.own.append(d)
            msg = f"Saved “{o_name}”."
            if info["unknown"]:
                tips = []
                for u in info["unknown"]:
                    close = difflib.get_close_matches(u, names, n=2)
                    tips.append(f"**{u}**" + (f" (did you mean {' / '.join(close)}?)" if close else ""))
                msg += " Not counted in nutrients: " + ", ".join(tips)
            ss["_own_msg"] = msg
            ss.own_form_id += 1
            ss.own_rows = 5
            st.rerun()
    if ss.get("_own_msg"):
        st.success(ss.pop("_own_msg"))

    if ss.own:
        st.markdown("**Saved recipes**")
        for o in own_objects():
            i = o.info
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{o.name}** · {o.meal_type} · {o.servings:g} servings · {i['kcal']:.0f} kcal/serving")
                if c2.button("View", key=f"view_{o.id}", width="stretch"):
                    open_recipe({"id": o.id, "is_own": True, "meal_type": o.meal_type, "minutes": None,
                                 "calories": i["kcal"]})
                if c2.button("Delete", key=f"del_{o.id}", width="stretch"):
                    ss.own = [d for d in ss.own if d["id"] != o.id]
                    st.rerun()
                fits = [PHASE_LABELS[p] for p, ok in i["labels"].items() if ok]
                if fits:
                    st.markdown("".join(f'<span class="ce-badge">{f}</span>' for f in fits), unsafe_allow_html=True)
                    if i["why"].get(phase):
                        st.markdown(f'<span class="ce-why">This phase: {i["why"][phase]} of daily value</span>',
                                    unsafe_allow_html=True)
                else:
                    reason = ("too many ingredients not in the list" if i["unknown_share"] > CFG["own_unknown_max"]
                              else "outside the usual calorie / nutrient range" if not i["plausible"]
                              else "not rich enough in this phase's nutrients")
                    st.caption(f"No phase badge: {reason}.")
                if i["unknown"]:
                    st.caption("Not counted: " + ", ".join(i["unknown"]))

with tab_about, st.container(key="sec_about"):
    section_title("About CycleEats")
    st.markdown("""
**How CycleEats chooses recipes**

Five recommenders, blended into one ranking (weights in *Advanced* in the sidebar):

| System | Idea |
|---|---|
| Popularity | Recipes rated highly by many people. Used when you have no favourites yet. |
| Content-Based | Recipes whose ingredients look like the ones you saved. |
| Item-Based CF | Recipes that people who liked yours also liked. |
| User-Based CF | Recipes liked by people with taste similar to yours. |
| Knowledge-Based | Nutrition rules for each cycle phase (e.g. iron in the menstrual phase). |

Your kitchen adds a sixth signal: how many of a recipe's ingredients you already have.

**Data:** 12,375 Food.com recipes (public domain) with nutrients estimated from USDA FoodData Central.
Nutrient values are estimates.

**Photos:** from [Unsplash](https://unsplash.com/?utm_source=cycleeats&utm_medium=referral), chosen by dish type
(e.g. *banana bread*, *lentil soup*). They illustrate the kind of dish, not the exact recipe.

**Privacy:** your cycle data, favourites and recipes stay in this browser session; use *Save my data* to keep them.
""")

st.markdown('<p class="ce-foot">CycleEats gives general food suggestions, not medical or nutritional advice. '
            'Talk to a doctor or dietitian about your own needs.</p>', unsafe_allow_html=True)
