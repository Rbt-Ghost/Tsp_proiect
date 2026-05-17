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
) -> Dict[str, Any]:
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

    start = time.time()
    pipeline.fit(train.data, train.target)
    duration_s = time.time() - start

    pred = pipeline.predict(test.data)
    acc = accuracy_score(test.target, pred)
    report = classification_report(test.target, pred, target_names=train.target_names)
    cm = confusion_matrix(test.target, pred)

    fig_cm = None
    if not no_plots:
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
) -> Dict[str, Any]:
    """Run Lab10 NLP tasks 1..5 for the UI.

    dataset is mapped to the same categories as in the lab scripts.
    """

    train, test = _load_20newsgroups(dataset)

    # Task 1: Naive Bayes + TF-IDF with implied parameters:
    # ngram_range=(1,1), max_features=None, stop_words='english', sublinear_tf=True
    if task_id == 1:
        return run_lab10_model(
            train=train,
            test=test,
            clf=MultinomialNB(),
            ngram_range=(1, 1),
            max_features=None,
            no_plots=no_plots,
            title_prefix="NaiveBayes",
        )

    # Task 2: compare classifiers (NB, LinearSVC, LogisticRegression, RandomForest)
    if task_id == 2:
        classifiers = {
            "Naive Bayes": MultinomialNB(),
            "LinearSVC": LinearSVC(max_iter=2000),
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="saga"),
            "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        }

        table = []
        best = None
        best_name = None
        best_res = None

        for name, clf in classifiers.items():
            res = run_lab10_model(
                train=train,
                test=test,
                clf=clf,
                ngram_range=(1, 1),
                max_features=None,
                no_plots=no_plots,
                title_prefix=name,
            )
            table.append({"Model": name, "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})
            if best is None or res["accuracy"] > best:
                best = res["accuracy"]
                best_name = name
                best_res = res

        # best plot already available if no_plots=False
        plot_cm = best_res.get("plot_cm") if best_res is not None else None
        report = best_res.get("report") if best_res is not None else ""
        cm = best_res.get("cm") if best_res is not None else None

        # for UI: return table + best confusion matrix
        return {
            "task": 2,
            "table": table,
            "best": best_name,
            "best_accuracy": float(best) if best is not None else None,
            "plot_cm": plot_cm,
            "report": report,
            "cm": cm,
        }

    # Task 3: study ngram_range (SVM)
    if task_id == 3:
        configs = [(1, 1), (1, 2), (2, 2), (1, 3)]
        rows = []
        for ng in configs:
            res = run_lab10_model(
                train=train,
                test=test,
                clf=LinearSVC(max_iter=2000),
                ngram_range=ng,
                max_features=None,
                no_plots=True,  # keep it light; we’ll show bar only
                title_prefix=f"SVM_ngram_{ng}",
            )
            rows.append({"ngram_range": str(ng), "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})

        fig = None
        if not no_plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [r["ngram_range"] for r in rows]
            accs = [r["Accuracy"] for r in rows]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, accs, color=colors, edgecolor="black")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("ngram_range")
            ax.set_title("Influența ngram_range (SVM)")
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

        return {"task": 3, "table": rows, "plot": fig}

    # Task 4: study max_features (SVM, ngram_range=(1,1))
    if task_id == 4:
        values = [100, 500, 1000, 5000, 10000, None]
        rows = []
        for mf in values:
            label = str(mf) if mf is not None else "toate"
            res = run_lab10_model(
                train=train,
                test=test,
                clf=LinearSVC(max_iter=2000),
                ngram_range=(1, 1),
                max_features=mf,
                no_plots=True,
                title_prefix=f"SVM_mf_{label}",
            )
            rows.append({"max_features": label, "Accuracy": res["accuracy"], "Time_s": res["duration_s"]})

        fig = None
        if not no_plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = [r["max_features"] for r in rows]
            accs = [r["Accuracy"] for r in rows]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, accs, color=colors, edgecolor="black")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("max_features")
            ax.set_title("Influența max_features (SVM)")
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()

        return {"task": 4, "table": rows, "plot": fig}

    # Task 5 (optional): grid ngram × max_features (SVM)
    if task_id == 5:
        try:
            import seaborn as sns
        except ImportError:
            return {
                "task": 5,
                "error": "seaborn is not installed; cannot render heatmap.",
                "heatmap": None,
            }

        ngrams = [(1, 1), (1, 2), (1, 3)]
        features = [500, 2000, 5000, 10000]
        data = np.zeros((len(ngrams), len(features)))

        for i, ng in enumerate(ngrams):
            for j, mf in enumerate(features):
                res = run_lab10_model(
                    train=train,
                    test=test,
                    clf=LinearSVC(max_iter=2000),
                    ngram_range=ng,
                    max_features=mf,
                    no_plots=True,
                    title_prefix=f"grid_{ng}_{mf}",
                )
                data[i, j] = res["accuracy"]

        fig = None
        if not no_plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                data,
                annot=True,
                fmt=".3f",
                cmap="YlOrRd",
                xticklabels=features,
                yticklabels=[str(ng) for ng in ngrams],
                ax=ax,
            )
            ax.set_xlabel("max_features")
            ax.set_ylabel("ngram_range")
            ax.set_title("Acuratețe (SVM) – Grid ngram × max_features")
            plt.tight_layout()

        return {
            "task": 5,
            "heatmap": fig,
            "table": {"ngrams": ngrams, "features": features, "data": data.tolist()},
        }

    raise ValueError(f"Unknown task_id: {task_id}")


def run_nlp_experiment(
    *,
    dataset: str,
    model_name: str,
    ngram_range: Tuple[int, int],
    max_features: Optional[int],
    no_plots: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    # kept for compatibility with the existing Streamlit NLP page
    train, test = _load_20newsgroups(dataset)

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

    return run_lab10_model(
        train=train,
        test=test,
        clf=clf,
        ngram_range=ngram_range,
        max_features=max_features,
        no_plots=no_plots,
        title_prefix=title,
    )


