from matplotlib import pyplot as plt
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score
import pandas as pd
import numpy as np
from src.models.glm_model import train_glm_baseline
from src.models.rf_model import train_random_forest


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series | list, exposure: float) -> tuple[float, float]:
    """
    Calculates Mean Squared Error (mpd) and Deviance (D^2) for the predictions.
    """
    mpd = mean_poisson_deviance(y_true, y_pred, sample_weight=exposure)
    d2 = d2_tweedie_score(y_true, y_pred, sample_weight = exposure, power = 1)
    return float(mpd), float(d2)


def generate_comparison_table(glm_mpd: float, glm_d2: float, rf_mpd: float, rf_d2: float) -> str:
    """
    Generates a formatted markdown table comparing the model performance.
    """
    markdown_table = (
        "| Model | Mean Poisson Deviance (MPD) | D-squared (D^2) |\n"
        "| :--- | :---: | :---: |\n"
        f"| Poisson GLM (Baseline) | {glm_mpd:.6f} | {glm_d2:.6f} |\n"
        f"| Random Forest (ML) | {rf_mpd:.6f} | {rf_d2:.6f} |\n"
    )
    return markdown_table

def plot_calibration(y_true_freq: pd.Series, y_pred_freq: pd.Series, exposure: float, n_bins: int = 50, title: str = "Calibration Plot"):
    cal = pd.DataFrame(dict(y_true = y_true_freq, y_pred = y_pred_freq, exposure = exposure))
    cal["bin"] = pd.qcut(cal["y_pred"],q=n_bins,duplicates="drop")

    cal["y_true_x_exposure"] = cal["y_true"] * cal["exposure"]
    cal["y_pred_x_exposure"] = cal["y_pred"] * cal["exposure"]

    grouped = (
        cal.groupby("bin", observed=True)
        .agg(
            y_true_x_exposure=("y_true_x_exposure", "sum"),
            y_pred_x_exposure=("y_pred_x_exposure", "sum"),
            exposure=("exposure", "sum")
        )
        .reset_index()
    )

    grouped["y_true"] = grouped["y_true_x_exposure"] / grouped["exposure"]
    grouped["y_pred"] = grouped["y_pred_x_exposure"] / grouped["exposure"]

    plt.figure()
    plt.scatter(grouped["y_true"], grouped["y_pred"])
    min_val = min(grouped["y_true"].min(), grouped["y_pred"].min())
    max_val = min(grouped["y_true"].max(), grouped["y_pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.xlabel("Predicted frequency")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()





def main():
    print("Training Poisson GLM model...")
    _, glm_y_test, glm_y_pred, glm_exp = train_glm_baseline()
    glm_mpd, glm_d2 = evaluate_predictions(glm_y_test, glm_y_pred, glm_exp)

    print("Training Random Forest model...")
    _, rf_y_test, rf_y_pred, rf_exp = train_random_forest()
    rf_mpd, rf_d2 = evaluate_predictions(rf_y_test, rf_y_pred, rf_exp)

    comparison_table = generate_comparison_table(glm_mpd, glm_d2, rf_mpd, rf_d2)
    print("\n### Model Evaluation Results")
    print(comparison_table)

    plot_calibration(glm_y_test, glm_y_pred, glm_exp, title="Calibration Plot GLM Model")
    plot_calibration(rf_y_test, rf_y_pred, rf_exp, title="Calibration Plot RF Model")

if __name__ == "__main__":
    main()
