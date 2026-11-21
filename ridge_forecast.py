"""Train the feed-free Ridge model and fill the bioreactor test template."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
KEYS = ["batch_id", "elapsed_time"]
ALPHAS = [1.0, 10.0, 0.1, 10.0]
RIDGE_WEIGHTS = np.array([0.75, 1.0, 0.75, 0.75])


def features(initial, query):
    values = initial[TARGETS + ["volume"]].to_numpy(float)
    return np.column_stack([
        np.broadcast_to(values, (len(query), len(values))),
        query["volume"].to_numpy(float) - initial["volume"],
        query["elapsed_time"].to_numpy(float),
    ])


def constrain(predictions):
    predictions = np.maximum(predictions, 0)
    predictions[:, 1] = np.maximum(predictions[:, 1], predictions[:, 0])
    return predictions


def baseline(initial, query, references):
    times = query["elapsed_time"].to_numpy(float)
    profiles, changes = [], []
    for reference in references:
        values = reference[TARGETS].to_numpy(float)
        profile = np.column_stack([
            np.interp(times, reference["elapsed_time"], series)
            for series in values.T
        ])
        profiles.append(profile)
        changes.append(profile - values[0])

    changes = np.stack(changes)
    weights = 1 / np.maximum(np.abs(np.stack(profiles)), 1e-8)
    order = np.argsort(changes, axis=0)
    cumulative = np.take_along_axis(weights, order, axis=0).cumsum(axis=0)
    median_index = (cumulative >= cumulative[-1] / 2).argmax(axis=0)
    sorted_changes = np.take_along_axis(changes, order, axis=0)
    delta = np.take_along_axis(sorted_changes, median_index[None], axis=0)[0]
    return constrain(initial[TARGETS].to_numpy(float) + delta)


def fit_models(references):
    inputs = np.stack([features(batch.iloc[0], batch.iloc[1:]) for batch in references])
    truth = np.stack([batch.iloc[1:][TARGETS].to_numpy(float) for batch in references])
    if not np.isfinite(inputs).all() or not (np.isfinite(truth) & (truth > 0)).all():
        raise ValueError("Training inputs must be finite and targets must be positive.")

    step_models = []
    for step in range(inputs.shape[1]):
        values = inputs[:, step]
        center = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (values - center) / scale
        models = [
            Ridge(alpha=alpha).fit(normalized, np.log(truth[:, step, column]))
            for column, alpha in enumerate(ALPHAS)
        ]
        step_models.append((center, scale, models))
    return step_models


def forecast(initial, query, references, step_models):
    query_x = features(initial, query)
    if len(query_x) != len(step_models):
        raise ValueError("Forecast and training horizons differ.")
    ridge = []
    for row, (center, scale, models) in zip(query_x, step_models):
        normalized = ((row - center) / scale)[None, :]
        ridge.append([np.exp(model.predict(normalized)[0]) for model in models])
    median = baseline(initial, query, references)
    return constrain((1 - RIDGE_WEIGHTS) * median + RIDGE_WEIGHTS * np.array(ridge))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--output", type=Path, default=Path("bioreaction_submission_model_comparison.csv"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    template = pd.read_csv(args.template, dtype=str)
    batches = {
        batch_id: batch.sort_values("elapsed_time")
        for batch_id, batch in data.groupby("batch_id", sort=False)
    }
    test_ids = set(data.loc[data["to_predict"], "batch_id"])
    if template.columns.tolist() != KEYS + TARGETS or set(template["batch_id"]) != test_ids:
        raise ValueError("Template columns or test batch IDs do not match.")

    references = [
        batch.dropna(subset=TARGETS)
        for batch_id, batch in batches.items()
        if batch_id not in test_ids
    ]
    step_models = fit_models(references)
    submission = template.copy()
    submission[TARGETS] = np.nan
    for batch_id, rows in template.groupby("batch_id", sort=False):
        batch = batches[batch_id]
        initial = batch.dropna(subset=TARGETS).iloc[0]
        query = batch.loc[batch["to_predict"]]
        predictions = forecast(initial, query, references, step_models)

        times = rows["elapsed_time"].to_numpy(float)
        distances = np.abs(times[:, None] - query["elapsed_time"].to_numpy(float))
        matches = distances.argmin(axis=1)
        if (len(rows) != len(query) or len(np.unique(matches)) != len(query)
                or np.any(distances[np.arange(len(rows)), matches] >= 1e-8)):
            raise ValueError(f"Template timestamps do not match for {batch_id}.")
        submission.loc[rows.index, TARGETS] = predictions[matches]

    if not np.isfinite(submission[TARGETS].to_numpy(float)).all():
        raise ValueError("Submission contains missing or non-finite predictions.")
    submission.to_csv(args.output, index=False)
    print(f"Trained on {len(references)} batches. Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
