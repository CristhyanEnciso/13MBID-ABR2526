import pandas as pd
import json
from pathlib import Path

RAW = Path("data/raw/bank-additional-full.csv")
OUT = Path("docs/test_results")
OUT.mkdir(parents=True, exist_ok=True)

VALID_MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}
VALID_DOW = {"mon","tue","wed","thu","fri"}
VALID_CONTACT = {"cellular","telephone"}

def test_great_expectations_sim():
    """
    Simulación de expectativas estilo Great Expectations con evidencia JSON.
    Falla si alguna expectativa no se cumple.
    """
    df = pd.read_csv(RAW, sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(name: str, condition: bool, message: str = ""):
        results["statistics"]["total_count"] += 1
        if bool(condition):
            results["statistics"]["success_count"] += 1
            results["expectations"].append({"expectation": name, "success": True})
        else:
            results["success"] = False
            results["expectations"].append({"expectation": name, "success": False, "message": message})

    # 1) Dominio de la variable objetivo
    add_expectation(
        "y_in_yes_no",
        df["y"].isin(["yes","no"]).all(),
        "La columna 'y' contiene valores fuera de {'yes','no'}."
    )

    # 2) Rango de edad
    add_expectation(
        "age_in_[17,100]",
        df["age"].between(17, 100).all(),
        "Hay filas con 'age' fuera del rango [17,100]."
    )

    # 3) No debería haber NaN en crudo (UCI usa 'unknown' en vez de NaN)
    add_expectation(
        "no_nan_in_raw",
        df.isna().sum().sum() == 0,
        "Existen NaN; en crudo deberían venir como 'unknown'."
    )

    # 4) Dominio de 'month'
    if "month" in df.columns:
        add_expectation(
            "month_in_valid_set",
            df["month"].isin(VALID_MONTHS).all(),
            f"Valores de 'month' fuera de {sorted(VALID_MONTHS)}."
        )

    # 5) Dominio de 'day_of_week'
    if "day_of_week" in df.columns:
        add_expectation(
            "day_of_week_in_valid_set",
            df["day_of_week"].isin(VALID_DOW).all(),
            f"Valores de 'day_of_week' fuera de {sorted(VALID_DOW)}."
        )

    # 6) Dominio de 'contact'
    if "contact" in df.columns:
        add_expectation(
            "contact_in_valid_set",
            df["contact"].isin(VALID_CONTACT).all(),
            f"Valores de 'contact' fuera de {sorted(VALID_CONTACT)}."
        )

    # 7) duration >= 0 (en segundos)
    if "duration" in df.columns:
        add_expectation(
            "duration_ge_0",
            (df["duration"] >= 0).all(),
            "Existen valores negativos en 'duration'."
        )

    # 8) campaign >= 1
    if "campaign" in df.columns:
        add_expectation(
            "campaign_ge_1",
            (df["campaign"] >= 1).all(),
            "Existen valores < 1 en 'campaign'."
        )

    # 9) pdays: 999 (nunca contactado) o >= 0
    if "pdays" in df.columns:
        pdays_ok = (df["pdays"] == 999) | (df["pdays"] >= 0)
        add_expectation(
            "pdays_in_{999}U[0,inf)",
            pdays_ok.all(),
            "Valores no válidos en 'pdays' (esperado 999 o >= 0)."
        )

    # 10) previous >= 0
    if "previous" in df.columns:
        add_expectation(
            "previous_ge_0",
            (df["previous"] >= 0).all(),
            "Existen valores negativos en 'previous'."
        )

    # 11) education no vacía
    if "education" in df.columns:
        add_expectation(
            "education_non_empty",
            (df["education"].astype(str).str.len() > 0).all(),
            "Existen valores vacíos en 'education'."
        )

    # Evidencia JSON
    (OUT / "ge_expectations_result.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Aserción final
    assert results["success"], f"Falló alguna expectativa: {results}"
