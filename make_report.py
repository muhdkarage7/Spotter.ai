"""Build the assessment report as a PDF (needs only matplotlib, which score.py already uses).

Run AFTER Spotter.py and score.py:
    python make_report.py
Defaults (all relative to the current folder):
    --results      results.json                        written by Spotter.py
    --chart        scorer_results/candidate_december.png   written by score.py
    --diagnostics  validation_diagnostics.png          written by Spotter.py
    --out          Freight_Rate_Report.pdf
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE_W, PAGE_H = 8.5, 11.0
LEFT, RIGHT, TOP, BOTTOM = 0.09, 0.91, 0.94, 0.06
INK, MUTED, ACCENT = "#161616", "#555555", "#064A56"


class Report:
    def __init__(self, path: Path):
        self.pdf = PdfPages(path)
        self.fig = self.ax = None
        self.y = TOP
        self._new_page()

    # -- page handling -----------------------------------------------------
    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.y = TOP

    def _need(self, height: float):
        if self.y - height < BOTTOM:
            self._new_page()

    @staticmethod
    def _line_height(size: float) -> float:
        return size * 1.5 / 72 / PAGE_H

    # -- content -----------------------------------------------------------
    def title(self, text: str, subtitle: str = ""):
        self._need(0.09)
        self.ax.text(LEFT, self.y, text, fontsize=20, fontweight="bold", color=INK, va="top")
        self.y -= 0.035
        if subtitle:
            self.ax.text(LEFT, self.y, subtitle, fontsize=10.5, color=MUTED, va="top")
            self.y -= 0.03
        self.y -= 0.01

    def heading(self, text: str):
        self._need(0.22)   # keep the heading with the content that follows it
        self.y -= 0.012
        self.ax.text(LEFT, self.y, text, fontsize=13, fontweight="bold", color=ACCENT, va="top")
        self.y -= 0.026
        self.ax.plot([LEFT, RIGHT], [self.y + 0.004, self.y + 0.004], color="#CCCCCC", linewidth=0.8)
        self.y -= 0.008

    def paragraph(self, text: str, size: float = 10, color: str = INK, indent: float = 0.0, bullet: bool = False):
        text = text.replace("$", r"\$")
        chars = int((RIGHT - LEFT - indent) * PAGE_W * 72 / (size * 0.56))
        lines = textwrap.wrap(text, width=chars) or [""]
        lh = self._line_height(size)
        self._need(lh * len(lines) + 0.008)
        for i, line in enumerate(lines):
            if bullet and i == 0:
                self.ax.text(LEFT + indent - 0.015, self.y, "\u2022", fontsize=size, color=color, va="top")
            self.ax.text(LEFT + indent, self.y, line, fontsize=size, color=color, va="top")
            self.y -= lh
        self.y -= 0.008

    def bullets(self, items, size: float = 10):
        for item in items:
            self.paragraph(item, size=size, indent=0.025, bullet=True)

    def table(self, header, rows, col_x):
        lh = self._line_height(10) * 1.15
        self._need(lh * (len(rows) + 2))
        for x, text in zip(col_x, header):
            self.ax.text(x, self.y, text, fontsize=10, fontweight="bold", color=INK, va="top")
        self.y -= lh
        self.ax.plot([LEFT, RIGHT], [self.y + 0.004, self.y + 0.004], color=INK, linewidth=0.8)
        for row in rows:
            for x, text in zip(col_x, row):
                self.ax.text(x, self.y, text, fontsize=10, color=INK, va="top")
            self.y -= lh
        self.y -= 0.012

    def image(self, path: Path, caption: str = ""):
        if not path.exists():
            self.paragraph(f"[missing image: {path}]", color="#B00020")
            return
        img = plt.imread(path)
        width_in = (RIGHT - LEFT) * PAGE_W
        height = width_in * img.shape[0] / img.shape[1] / PAGE_H
        self._need(height + 0.04)
        axis = self.fig.add_axes([LEFT, self.y - height, RIGHT - LEFT, height])
        axis.imshow(img)
        axis.axis("off")
        self.y -= height + 0.008
        if caption:
            self.paragraph(caption, size=8.5, color=MUTED)

    def close(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.pdf.close()


def money(value: float) -> str:
    return f"${value:,.1f}"


def build(results: dict, chart: Path, diagnostics: Path, out: Path):
    r = results
    rows, dates = r["rows"], r["dates"]
    pct = lambda x: f"{x:.0f}%"
    report = Report(out)

    report.title("Freight Rate Prediction: Approach and Results",
                 "Machine Learning Engineer assessment  |  Muhammad Karage")

    report.heading("Summary")
    report.paragraph(
        f"A LightGBM gradient-boosting model predicts posted_rate from load features and calendar features. "
        f"On a chronological holdout it beats a lane-median baseline by {pct(r['mae_gain_pct'])} on MAE "
        f"({r['holdout']['MAE']:.1f} vs {r['baseline']['MAE']:.1f}) and {pct(r['rmse_gain_pct'])} on RMSE "
        f"({r['holdout']['RMSE']:.1f} vs {r['baseline']['RMSE']:.1f}) across all holdout rows. On typical loads "
        f"(rate per mile inside {r['rate_per_mile_fences'][0]:.2f} to {r['rate_per_mile_fences'][1]:.2f}, "
        f"{rows['typical_holdout']:,} of {rows['holdout']:,} holdout rows) MAE is "
        f"{r['typical_holdout']['MAE']:.1f} vs {r['typical_baseline']['MAE']:.1f} for the baseline. "
        f"A small number of extreme rates dominate the overall RMSE; see the limitations section.")

    report.heading("Train / test split and validation approach")
    report.bullets([
        f"Labelled data ({rows['train_total']:,} rows, {dates['train'][0]} to {dates['train'][1]}) was sorted by date. "
        f"The first {rows['fit']:,} rows fit the model; the most recent {rows['holdout']:,} rows "
        f"({dates['holdout'][0]} to {dates['holdout'][1]}) form the holdout.",
        f"Why chronological: the supplied validation set is a later period ({dates['validation'][0]} to "
        f"{dates['validation'][1]}), so a random split would leak future information and overstate accuracy.",
        "Encoders and imputers are fitted on the fit partition only, then applied to the holdout, the validation set "
        "and the December inputs.",
        f"Boosting rounds are chosen by early stopping on the holdout (best round {r['best_round']}, "
        f"{r['rounds_trained']} trained). The final model is refit on all labelled rows with that round count.",
        "A lane-median baseline (median rate per pickup-to-delivery lane, overall median for unseen lanes) is scored "
        "on the same holdout as the reference the model must beat.",
    ])

    report.heading("Data quality findings and how they were handled")
    report.bullets([line.strip() for line in r["notes"] if line.strip()], size=8.5)

    report.heading("Model choice")
    report.paragraph(
        "LightGBM was chosen because the data is tabular, mixes numeric and categorical inputs, and rates depend "
        "non-linearly on distance, equipment and lane. Trees need no feature scaling and handle these interactions "
        "directly. Categories are one-hot encoded. Features are the supplied columns plus calendar features "
        "(day of week, day of month, month, days to and since the nearest US federal holiday). The December chart "
        "varies only by date, so calendar features are what allow the model to respond to it.")

    report.heading("Holdout results")
    fmt = lambda d: (f"{d['RMSE']:.1f}", f"{d['MAE']:.1f}")
    table_rows = [
        ("LightGBM", f"All holdout rows ({rows['holdout']:,})", *fmt(r["holdout"])),
        ("Lane-median baseline", f"All holdout rows ({rows['holdout']:,})", *fmt(r["baseline"])),
        ("LightGBM", f"Typical rows ({rows['typical_holdout']:,})", *fmt(r["typical_holdout"])),
        ("Lane-median baseline", f"Typical rows ({rows['typical_holdout']:,})", *fmt(r["typical_baseline"])),
    ]
    report.table(("Model", "Rows", "RMSE", "MAE"), table_rows, (LEFT, LEFT + 0.27, LEFT + 0.62, LEFT + 0.75))
    report.image(diagnostics, "Holdout diagnostics: predicted vs actual, error distribution, and training/holdout loss "
                              "by boosting round.")

    report.heading("December 2025 prediction chart (produced by score.py)")
    report.image(chart, "Fixed inputs: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes.")
    report.paragraph(
        f"Predicted rates range from {money(r['december_range'][0])} to {money(r['december_range'][1])} across "
        f"December. Missing December features (for example market signals) are filled with the typical value "
        f"from the last 30 days of labelled data.")

    report.heading("Limitations")
    report.bullets([
        f"Training data ends {dates['train'][1]}, while validation runs to {dates['validation'][1]}. The model has "
        "never seen November or December, so it cannot learn their seasonality directly. Holiday-distance features "
        "let it reuse patterns from earlier holidays, but that is an assumption, not something the data proves.",
        f"About {100 * (1 - rows['typical_holdout'] / rows['holdout']):.1f}% of holdout rows have an extreme rate per "
        "mile. They produce most of the squared error, so overall RMSE stays high while typical-load error is far lower. "
        "Both are reported rather than hiding the difference.",
        "The holdout covers the most recent labelled months, which may not match conditions in the validation period.",
    ])
    report.close()


def main():
    parser = argparse.ArgumentParser(description="Build the assessment report PDF.")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--chart", default="scorer_results/candidate_december.png")
    parser.add_argument("--diagnostics", default="validation_diagnostics.png")
    parser.add_argument("--out", default="Freight_Rate_Report.pdf")
    args, _ = parser.parse_known_args()  # ignore extra arguments (e.g. the "-f kernel.json" a notebook adds)

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Cannot find {results_path}. Run Spotter.py first.")
    build(json.loads(results_path.read_text(encoding="utf-8")), Path(args.chart), Path(args.diagnostics), Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
