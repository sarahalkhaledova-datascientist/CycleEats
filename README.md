# CycleEats

A Streamlit app that recommends recipes based on two things: **the ingredients you have at home**, and **the phase of your menstrual cycle** (menstrual, follicular, ovulatory, luteal).

> Suggestions are general nutrition information, not medical advice. Cycle data stays on the user's device.

## Status

| Stage | Status |
|---|---|
| Data pipeline (notebooks 01–06) | Done |
| Recommender systems + evaluation | Next |
| Streamlit app | Planned |

## Data pipeline

Run the notebooks in this order, from inside the `notebooks/` folder.

| # | Notebook | What it does | Main output |
|---|---|---|---|
| 01 | `01_cleaning.ipynb` | Strict ingredient/quantity filter, missing values, time ≤ 3 h | `recipes_clean.pkl` (84,905 → 78,779 recipes) |
| 02 | `02_ingredients.ipynb` | Quantities to numbers, ingredient name unification, top-500 ingredients | `recipes_candidates.pkl` (46,145) |
| 03a | `03a_mapping_fixes.ipynb` | Manual fixes to the USDA matching (run **before** 03b) | `manual_overrides.csv` |
| 03b | `03b_usda_mapping.ipynb` | Matches each ingredient to a USDA food (automatic → hints → manual) | `ingredient_usda_map.csv`, `usda_nutrients_per100g.pkl` |
| 04 | `04_nutrients.ipynb` | Grams per ingredient (unit calibration by calories, 70/30 train/test), nutrients per serving, quality flag | `recipes_nutrients.pkl` (42,736) |
| 05 | `05_phase_labels.ipynb` | Phase scores and labels from evidence-weighted nutrient rules (% FDA daily value) | `recipes_labeled.pkl` |
| 06 | `06_interactions.ipynb` | Food.com reviews: cleaning, transformation, train/test split | `interactions_train.pkl`, `interactions_test.pkl` |

### Key numbers

- 522,517 raw recipes → **42,736** recipes with estimated nutrients
- Quality (estimated vs. original calories): good 9,782 · fair 4,677 · poor 10,590 · unverified 17,687
- Phase labels (good + fair only): menstrual 2,134 · follicular 1,910 · ovulatory 1,910 · luteal 1,611
- Interactions: 164,235 from 60,183 users on 24,452 recipes (density 0.011%)
- Evaluation: 4,852 users with ≥ 5 interactions; 20% of their liked recipes held out

## Recommender systems (planned)

1. **Popularity**: Bayesian weighted rating (baseline)
2. **Content-based**: ingredient similarity (TF-IDF, embeddings)
3. **Collaborative filtering, user-based**
4. **Collaborative filtering, item-based**
5. **Knowledge-based**: cycle-phase nutrient rules

Combined in a **hybrid** ranking with adjustable weights. Evaluated with HitRate@10, NDCG@10 and catalog coverage.

## Repository structure

```
├── data/
│   ├── raw/          original datasets (not in the repo, see below)
│   └── processed/    generated data (not in the repo; later: the app's data file)
├── notebooks/        notebooks + hand-made files (manual_*.csv, phase_*.csv)
├── src/              shared code (recommenders, app)
├── README.md
└── requirements.txt
```

Generated files (`.pkl` and intermediate `.csv`) are not in the repo. Running the notebooks recreates them.

## Data sources

Download and place in `data/raw/`:

- **Food.com Recipes and Reviews** (Kaggle): `recipes.csv`, `reviews.csv`
  https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews
- **USDA FoodData Central, SR Legacy** (CSV, April 2018): folder `FoodData_Central_sr_legacy_food_csv_2018-04`
  https://fdc.nal.usda.gov/download-datasets.html

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Limitations

- Food.com quantities have no units, so grams are estimated by calibration (typical calorie error around 46%). Nutrients are shown as categories, not exact numbers.
- 41% of recipes have no servings count, so their calories cannot be verified.
- The phase rules need review by a nutritionist. Vitamin D and omega-3 are not included.
- Ratings are sparse and skewed (72.6% five stars; the top 1% of recipes hold 26% of interactions).
