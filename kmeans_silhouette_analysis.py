import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(r"C:\Users\^_^\Downloads\dimensionfoodconsumption.xlsx")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}

ORDINAL_MAPS = {
    "(B7) EDUCATION LEVEL": {
        "Primary School": 1,
        "Junior High School": 2,
        "Senior High School": 3,
        "Diploma": 4,
        "Bachelor degree": 5,
        "Master and doctoral degree": 6,
    },
    "(B10) SOCIAL CLAS": {
        "Lower Class": 1,
        "Middle Class": 2,
        "Upper Class": 3,
    },
    "(C13) TIME OF INCOME RECEIPT": {
        "Daily": 1,
        "Weekly": 2,
        "Monthly": 3,
    },
}


def parse_million_idr(value):
    """Convert Excel time-like values such as 03:08:00 into 3.8 million IDR."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"^(\d{1,2}):(\d{2}):\d{2}$", text)
    if match:
        whole = int(match.group(1))
        decimal = int(match.group(2))
        return whole + decimal / 10
    try:
        return float(text)
    except ValueError:
        return np.nan


def standardize(frame):
    values = frame.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1
    return (values - mean) / std


def pca_two_components(values):
    centered = values - values.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    scores = centered @ components
    variance = np.var(scores, axis=0, ddof=0)
    explained = variance / np.var(centered, axis=0, ddof=0).sum()
    return scores, explained


def kmeans(values, k, n_init=10, max_iter=150, seed=42):
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_inertia = math.inf
    n = len(values)

    for _ in range(n_init):
        centers = values[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_id in range(k):
                members = values[labels == cluster_id]
                if len(members):
                    centers[cluster_id] = members.mean(axis=0)

        inertia = ((values - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def silhouette_score(values, labels):
    n = len(values)
    clusters = sorted(np.unique(labels))
    if len(clusters) < 2 or len(clusters) == n:
        return np.nan

    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    score = np.zeros(n)

    for i in range(n):
        own = labels == labels[i]
        own_count = own.sum()
        a = distances[i, own].sum() / (own_count - 1) if own_count > 1 else 0
        b = min(distances[i, labels == c].mean() for c in clusters if c != labels[i])
        score[i] = (b - a) / max(a, b) if max(a, b) else 0

    return float(score.mean())


def add_preprocessed_columns(df):
    clean = df.copy()
    clean["age"] = clean["(B4) AGE"]
    clean["urban_years"] = clean["(B8) LIVED IN URBAN AREA (YEARS)"]
    clean["education_score"] = clean["(B7) EDUCATION LEVEL"].map(ORDINAL_MAPS["(B7) EDUCATION LEVEL"])
    clean["social_class_score"] = clean["(B10) SOCIAL CLAS"].map(ORDINAL_MAPS["(B10) SOCIAL CLAS"])
    clean["household_member"] = clean["(B11) HOUSEHOLD MEMBER"]
    clean["income_million_idr"] = clean["(C12) HOUSEHOLD INCOME (MONTHLY IN MILLION IDR)"].map(parse_million_idr)
    clean["income_receipt_score"] = clean["(C13) TIME OF INCOME RECEIPT"].map(ORDINAL_MAPS["(C13) TIME OF INCOME RECEIPT"])
    clean["expenditure_million_idr"] = clean["(C14) HOUSEHOLD EXPENDITURE (MONTHLY IN MILLION IDR)"].map(parse_million_idr)
    clean["food_expenditure_pct"] = clean["(C15) % MONTHLY EXPENDITURE FOR FOOD"]

    clean["income_per_capita"] = clean["income_million_idr"] / clean["household_member"]
    clean["expenditure_per_capita"] = clean["expenditure_million_idr"] / clean["household_member"]
    clean["food_spending_million_idr"] = clean["expenditure_million_idr"] * clean["food_expenditure_pct"] / 100
    clean["non_food_spending_million_idr"] = clean["expenditure_million_idr"] - clean["food_spending_million_idr"]
    clean["saving_gap_million_idr"] = clean["income_million_idr"] - clean["expenditure_million_idr"]
    clean["urban_life_ratio"] = clean["urban_years"] / clean["age"].clip(lower=1)

    for col in clean.columns:
        if clean[col].dtype == "object" and clean[col].dropna().isin(LIKERT_MAP).all():
            clean[col + "_score"] = clean[col].map(LIKERT_MAP)

    return clean


def evaluate_feature_set(clean, feature_set, k_values=(2, 3, 4, 5, 6)):
    data = clean[list(feature_set)].dropna()
    scaled = standardize(data)
    pca_scores, explained = pca_two_components(scaled)

    rows = []
    for k in k_values:
        labels, centers, inertia = kmeans(pca_scores, k)
        silhouette = silhouette_score(pca_scores, labels)
        rows.append(
            {
                "features": list(feature_set),
                "n_features": len(feature_set),
                "k": k,
                "silhouette": silhouette,
                "inertia": float(inertia),
                "pca_explained_pc1": float(explained[0]),
                "pca_explained_pc2": float(explained[1]),
                "pca_explained_total": float(explained.sum()),
                "labels": labels,
                "pca_scores": pca_scores,
            }
        )
    return rows


def main():
    raw = pd.read_excel(SOURCE)
    clean = add_preprocessed_columns(raw)

    feature_sets = {
        "full_socioeconomic": [
            "age",
            "urban_years",
            "education_score",
            "social_class_score",
            "household_member",
            "income_million_idr",
            "income_receipt_score",
            "expenditure_million_idr",
            "food_expenditure_pct",
        ],
        "economic_core": [
            "income_million_idr",
            "expenditure_million_idr",
            "food_expenditure_pct",
            "social_class_score",
        ],
        "economic_capacity": [
            "income_million_idr",
            "expenditure_million_idr",
            "income_per_capita",
            "expenditure_per_capita",
            "saving_gap_million_idr",
        ],
        "food_budget_profile": [
            "expenditure_million_idr",
            "food_expenditure_pct",
            "food_spending_million_idr",
            "non_food_spending_million_idr",
        ],
        "demographic_economic": [
            "age",
            "urban_life_ratio",
            "education_score",
            "social_class_score",
            "household_member",
            "income_per_capita",
            "expenditure_per_capita",
            "food_expenditure_pct",
        ],
    }

    all_results = []
    for name, features in feature_sets.items():
        for row in evaluate_feature_set(clean, features):
            row["feature_set_name"] = name
            all_results.append(row)

    targeted_sets = {
        "target_income_expenditure_food": [
            "income_million_idr",
            "expenditure_million_idr",
            "food_expenditure_pct",
        ],
        "target_per_capita_food": [
            "income_per_capita",
            "expenditure_per_capita",
            "food_expenditure_pct",
        ],
        "target_budget_capacity": [
            "income_million_idr",
            "expenditure_million_idr",
            "saving_gap_million_idr",
        ],
        "target_food_spending_capacity": [
            "income_million_idr",
            "food_spending_million_idr",
            "non_food_spending_million_idr",
        ],
        "target_household_food_burden": [
            "household_member",
            "income_per_capita",
            "food_expenditure_pct",
            "food_spending_million_idr",
        ],
    }

    for name, features in targeted_sets.items():
        for row in evaluate_feature_set(clean, features):
            row["feature_set_name"] = name
            all_results.append(row)

    summary_rows = []
    for row in all_results:
        summary_rows.append({k: v for k, v in row.items() if k not in {"labels", "pca_scores"}})
    summary = pd.DataFrame(summary_rows).sort_values("silhouette", ascending=False)

    best = max(all_results, key=lambda row: row["silhouette"])
    labels = best["labels"]
    pca_scores = best["pca_scores"]

    exported = clean.copy()
    exported["PC1"] = pca_scores[:, 0]
    exported["PC2"] = pca_scores[:, 1]
    exported["cluster"] = labels + 1

    features = best["features"]
    profile = exported.groupby("cluster")[features + ["PC1", "PC2"]].mean().round(3)
    counts = exported["cluster"].value_counts().sort_index().rename("count")
    profile.insert(0, "count", counts)

    summary_for_file = summary.copy()
    summary_for_file["features"] = summary_for_file["features"].map(lambda x: ", ".join(x))
    summary_for_file.head(50).to_csv(OUTPUT_DIR / "clustering_experiment_summary.csv", index=False)
    exported.to_csv(OUTPUT_DIR / "food_consumption_clustered_for_dashboard.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "cluster_profile_unlabelled.csv")

    with (OUTPUT_DIR / "best_clustering_result.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_set_name": best["feature_set_name"],
                "features": best["features"],
                "k": best["k"],
                "silhouette": best["silhouette"],
                "pca_explained_total": best["pca_explained_total"],
                "pca_explained_pc1": best["pca_explained_pc1"],
                "pca_explained_pc2": best["pca_explained_pc2"],
            },
            f,
            indent=2,
        )

    print("BEST")
    print(json.dumps(
        {
            "feature_set_name": best["feature_set_name"],
            "features": best["features"],
            "k": best["k"],
            "silhouette": round(best["silhouette"], 4),
            "pca_explained_total": round(best["pca_explained_total"], 4),
        },
        indent=2,
    ))
    print("\nTOP 15")
    print(summary_for_file.head(15).to_string(index=False))
    print("\nPROFILE")
    print(profile.to_string())


if __name__ == "__main__":
    main()
