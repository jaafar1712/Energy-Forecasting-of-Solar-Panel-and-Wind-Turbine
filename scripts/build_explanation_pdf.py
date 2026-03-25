from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem


def build_pdf(output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 15
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=9.5,
        leading=12,
    )

    story = []
    story.append(Paragraph("Hybrid Solar/Wind Digital Twin: Model Explanation", title_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Project Goal", heading_style))
    story.append(
        Paragraph(
            "Create a synthetic digital-twin dataset for a hybrid Solar/Wind system and train baseline and forecast AI models for power estimation.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Pipeline Outputs", heading_style))
    outputs = [
        "energy_dataset.csv (8760 hourly rows)",
        "Static plots in plots folder",
        "Live dashboard plots during execution (optional)",
        "Baseline model artifact: power_model_rf.joblib",
        "Forecast model artifact: power_forecast_1h_rf.joblib",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style)) for item in outputs],
            bulletType="1",
            start="1",
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Data Synthesis", heading_style))
    synthesis_points = [
        "Lux follows a diurnal sine pattern, seasonal modulation, and cloud noise.",
        "Wind speed is sampled from a Weibull distribution.",
        "Temperature combines annual cycle, daily cycle, lux correlation, and noise.",
        "Power combines solar and wind terms; current uses I = P / V.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style)) for item in synthesis_points],
            bulletType="bullet",
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Physics-Inspired Equations", heading_style))
    equations = [
        "P_solar = P_rated * (lux / lux_max)",
        "P_wind = k * v^3",
        "P_total = P_solar + P_wind",
        "I = P / V",
    ]
    for eq in equations:
        story.append(Paragraph(eq, code_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Model Training", heading_style))
    model_points = [
        "Baseline model: RandomForestRegressor predicts same-hour power_produced_w.",
        "Forecast model: RandomForestRegressor predicts next-hour power_next_1h_w.",
        "Baseline split is random 80/20; forecast split is chronological 80/20.",
        "Metrics: MAE (W) and R2.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, body_style)) for item in model_points],
            bulletType="bullet",
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Metric Interpretation", heading_style))
    story.append(
        Paragraph(
            "MAE is average absolute error in watts. For hourly data, MAE in watts is approximately equivalent to Wh error per hour interval.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("How To Run", heading_style))
    story.append(Paragraph('.\\.venv\\Scripts\\python.exe .\\train_energy_model.py', code_style))
    story.append(Paragraph('.\\.venv\\Scripts\\python.exe .\\train_energy_model.py --no-live-plots', code_style))

    doc.build(story)


def main() -> None:
    output = Path("docs") / "energy_model_explanation.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(output)
    print(f"Generated PDF: {output}")


if __name__ == "__main__":
    main()
