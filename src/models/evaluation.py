from sklearn.metrics import mean_squared_error, d2_tweedie_score
import pandas as pd
from src.models.glm_model import train_glm_baseline
from src.models.rf_model import train_random_forest


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series | list, exposure: float) -> tuple[float, float]:
    """
    Calculates Mean Squared Error (MSE) and R-squared (R^2) for the predictions.
    """
    mse = mean_squared_error(y_true, y_pred)
    d2 = d2_tweedie_score(y_true, y_pred, sample_weight = exposure, power = 1)
    return float(mse), float(d2)


def generate_comparison_table(glm_mse: float, glm_d2: float, rf_mse: float, rf_d2: float) -> str:
    """
    Generates a formatted markdown table comparing the model performance.
    """
    markdown_table = (
        "| Model | Mean Squared Error (MSE) | D-squared (D^2) |\n"
        "| :--- | :---: | :---: |\n"
        f"| Poisson GLM (Baseline) | {glm_mse:.6f} | {glm_d2:.6f} |\n"
        f"| Random Forest (ML) | {rf_mse:.6f} | {rf_d2:.6f} |\n"
    )
    return markdown_table


def main():
    print("Training Poisson GLM model...")
    _, glm_y_test, glm_y_pred, glm_exp = train_glm_baseline()
    glm_mse, glm_d2 = evaluate_predictions(glm_y_test, glm_y_pred, glm_exp)

    print("Training Random Forest model...")
    _, rf_y_test, rf_y_pred, rf_exp = train_random_forest()
    rf_mse, rf_d2 = evaluate_predictions(rf_y_test, rf_y_pred, rf_exp)

    comparison_table = generate_comparison_table(glm_mse, glm_d2, rf_mse, rf_d2)
    print("\n### Model Evaluation Results")
    print(comparison_table)


if __name__ == "__main__":
    main()
