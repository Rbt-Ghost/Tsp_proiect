from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def _load_20newsgroups(dataset: str):
    if dataset == "20newsgroups_small":
        categories = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    elif dataset == "20newsgroups_medium":
        categories = None
    elif dataset == "20newsgroups_full":
        categories = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    kwargs: Dict[str, Any] = {
        "remove": ("headers", "footers", "quotes"),
    }

    if categories is not None:
        train = fetch_20newsgroups(subset="train", categories=categories, **kwargs)
        test = fetch_20newsgroups(subset="test", categories=categories, **kwargs)
    else:
        train = fetch_20newsgroups(subset="train", **kwargs)
        test = fetch_20newsgroups(subset="test", **kwargs)

    return train, test


def run_lab10_model(
    *,
    train,
    test,
    clf,
    ngram_range: Tuple[int, int],
    max_features: Optional[int],
    no_plots: bool,
    title_prefix: str,
    log_fn=None,
) -> Dict[str, Any]:
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    clf_name = type(clf).__name__
    mf_str = str(max_features) if max_features is not None else "all"
    _log(f"🔧 Pipeline: TF-IDF(ngram={ngram_range}, max_features={mf_str}) → {clf_name}")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )

    _log(f"⏳ Training on {len(train.data):,} documents...")
    start = time.time()
    pipeline.fit(train.data, train.target)
    duration_s = time.time() - start
    _log(f"   Training complete — {duration_s:.2f}s")

    _log(f"📊 Predicting on {len(test.data):,} test documents...")
    pred = pipeline.predict(test.data)
    acc = accuracy_score(test.target, pred)
    report = classification_report(test.target, pred, target_names=train.target_names)
    cm = confusion_matrix(test.target, pred)
    _log(f"✔️ Accuracy: {acc:.4f}")

    fig_cm = None
    if not no_plots:
        _log("📈 Generating confusion matrix...")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train.target_names).plot(
            ax=ax, cmap="Blues", colorbar=True, values_format="d"
        )
        ax.set_title(f"{title_prefix} - Confusion Matrix")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig_cm = fig

    top_confusions = None
    if cm.shape[0] > 1:
        pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[0]):
                if i == j:
                    continue
                pairs.append((int(cm[i, j]), i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        top_confusions = [
            {
                "real": train.target_names[i],
                "pred": train.target_names[j],
                "count": c,
            }
            for (c, i, j) in pairs[:5]
        ]

    return {
        "accuracy": float(acc),
        "duration_s": float(duration_s),
        "pred": pred,
        "report": report,
        "cm": cm,
        "plot_cm": fig_cm,
        "top_confusions": top_confusions,
    }


def run_lab10_task(
    *,
    task_id: int,
    dataset: str,
    no_plots: bool,
    seed: int,
    log_fn=None,
) -> Dict[str, Any]:
    """Run Lab10 NLP tasks 1..5 for the UI."""

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    _log(f"📂 Loading dataset ‘{dataset}’...")
    train, test = _load_20newsgroups(dataset)
    _log(f"✔️ {len(train.data):,} training docs | {len(test.data):,} test docs | {len(train.target_names)} classes")

    # Task 1: Naive Bayes + TF-IDF
    if task_id == 1:
        _log("🔬 Task 1 — Naive Bayes with default TF-IDF")
        return run_lab10_model(
            train=train, test=test, clf=MultinomialNB(),
            ngram_range=(1, 1), max_features=None,
            no_plots=no_plots, title_prefix="NaiveBayes", log_fn=log_fn,
        )

    # Task 2: compare classifiers
    if task_id == 2:
        classifiers = {
            "Naive Bayes": MultinomialNB(),
            "LinearSVC": LinearSVC(max_iter=2000),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="saga"),
            "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        }
        total = len(classifiers)
        table = []
        best = None
        best_name = None
        best_res = None

        for idx, (name, clf) in enumerate(classifiers.items(), 1):
            _log(f"🔬 [{idx}/{total}] Training {name}...")
            res = run_lab10_model(
                train=train, test=test, clf=clf,
                ngram_range=(1, 1), max_features=None,
                no_plots=no_plots, title_prefix=name, log_fn=log_fn,
            )
            table.append({"Model": name, "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})
            if best is None or res["accuracy"] > best:
                best = res["accuracy"]
                best_name = name
                best_res = res

        _log(f"🏆 Winner: {best_name} — accuracy={best:.4f}")
        plot_cm = best_res.get("plot_cm") if best_res is not None else None
        report = best_res.get("report") if best_res is not None else ""
        cm = best_res.get("cm") if best_res is not None else None
        return {
            "task": 2, "table": table, "best": best_name,
            "best_accuracy": float(best) if best is not None else None,
            "plot_cm": plot_cm, "report": report, "cm": cm,
        }

    # Task 3: study ngram_range (SVM)
    if task_id == 3:
        configs = [(1, 1), (1, 2), (2, 2), (1, 3)]
        rows = []
        for idx, ng in enumerate(configs, 1):
            _log(f"📐 [{idx}/{len(configs)}] SVM with ngram_range={ng}...")
            res = run_lab10_model(
                train=train, test=test, clf=LinearSVC(max_iter=2000),
                ngram_range=ng, max_features=None,
                no_plots=True, title_prefix=f"SVM_ngram_{ng}", log_fn=log_fn,
            )
            rows.append({"ngram_range": str(ng), "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})

        fig = None
        if not no_plots:
            _log("📈 Generating graph for ngram_range...")
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [r["ngram_range"] for r in rows]
            accs = [r["Accuracy"] for r in rows]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, accs, color=colors, edgecolor="black")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("ngram_range")
            ax.set_title("Impact of ngram_range (SVM)")
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

        return {"task": 3, "table": rows, "plot": fig}

    # Task 4: study max_features (SVM)
    if task_id == 4:
        values = [100, 500, 1000, 5000, 10000, None]
        rows = []
        for idx, mf in enumerate(values, 1):
            label = str(mf) if mf is not None else "all"
            _log(f"📐 [{idx}/{len(values)}] SVM with max_features={label}...")
            res = run_lab10_model(
                train=train, test=test, clf=LinearSVC(max_iter=2000),
                ngram_range=(1, 1), max_features=mf,
                no_plots=True, title_prefix=f"SVM_mf_{label}", log_fn=log_fn,
            )
            rows.append({"max_features": label, "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})

        fig = None
        if not no_plots:
            _log("📈 Generating graph for max_features...")
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [r["max_features"] for r in rows]
            accs = [r["Accuracy"] for r in rows]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, accs, color=colors, edgecolor="black")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("max_features")
            ax.set_title("Impact of max_features (SVM)")
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

        return {"task": 4, "table": rows, "plot": fig}

    # Task 5: grid ngram × max_features (SVM)
    if task_id == 5:
        try:
            import seaborn as sns
        except ImportError:
            return {"task": 5, "error": "seaborn is not installed; cannot render heatmap.", "heatmap": None}

        ngrams = [(1, 1), (1, 2), (1, 3)]
        features = [500, 2000, 5000, 10000]
        data = np.zeros((len(ngrams), len(features)))
        total_combos = len(ngrams) * len(features)
        combo = 0

        for i, ng in enumerate(ngrams):
            for j, mf in enumerate(features):
                combo += 1
                _log(f"🔲 [{combo}/{total_combos}] Grid: ngram={ng}, max_features={mf}...")
                res = run_lab10_model(
                    train=train, test=test, clf=LinearSVC(max_iter=2000),
                    ngram_range=ng, max_features=mf,
                    no_plots=True, title_prefix=f"grid_{ng}_{mf}", log_fn=log_fn,
                )
                data[i, j] = res["accuracy"]

        fig = None
        if not no_plots:
            _log("📈 Generating heatmap grid...")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                data, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=features, yticklabels=[str(ng) for ng in ngrams], ax=ax,
            )
            ax.set_xlabel("max_features")
            ax.set_ylabel("ngram_range")
            ax.set_title("Accuracy (SVM) – Grid ngram × max_features")
            plt.tight_layout()

        _log("✅ Grid search finished!")
        return {"task": 5, "heatmap": fig, "table": {"ngrams": ngrams, "features": features, "data": data.tolist()}}

    raise ValueError(f"Unknown task_id: {task_id}")


def run_nlp_experiment(
    *,
    dataset: str,
    model_name: str,
    ngram_range: Tuple[int, int],
    max_features: Optional[int],
    no_plots: bool = False,
    seed: int = 42,
    log_fn=None,
) -> Dict[str, Any]:
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    _log(f"📂 Loading dataset '{dataset}'...")
    train, test = _load_20newsgroups(dataset)
    _log(f"✔️ {len(train.data):,} training docs | {len(test.data):,} test docs | {len(train.target_names)} classes")

    if model_name == "Naive Bayes":
        clf = MultinomialNB()
        title = "NaiveBayes"
    elif model_name == "LinearSVC":
        clf = LinearSVC(max_iter=2000, random_state=seed)
        title = "LinearSVC"
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, solver="saga", random_state=seed)
        title = "LogisticRegression"
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        title = "RandomForest"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    _log(f"🔬 Selected model: {model_name}")
    return run_lab10_model(
        train=train, test=test, clf=clf,
        ngram_range=ngram_range, max_features=max_features,
        no_plots=no_plots, title_prefix=title, log_fn=log_fn,
    )


