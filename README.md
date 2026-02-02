# Bond Yield Calculator

MAT1856/APM466 Assignment 1: **The Value of Time** — yield curves, spot and forward rates, covariance, and PCA for Canadian government bonds.

## Contents

- **`yield_curve.ipynb`** — Main notebook: bond selection, YTM, bootstrapped spot curves, 1-year forward curves, covariance matrices, and PCA (eigenvalues/eigenvectors).
- **`bonds_master_all.csv`** — Bond data (prices, coupons, maturities) used for the analysis.
- **`figures/`** — Plots: yield, spot, forward curves; covariance heatmaps (spot and forward).
- **`main.tex`** — LaTeX report (3-page limit); compile with `pdflatex main.tex`.
- **`answers.md`** — Draft answers and notes for the assignment.

## Setup

```bash
# Optional: use a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install pandas numpy matplotlib
```

Open and run `yield_curve.ipynb` in Jupyter or VS Code to regenerate figures and results.

## Data

Bond data is from [Markets Insider](https://www.marketsinsider.com/) (Frankfurt listing). The scraper used for collection is the one supplied by Jaspreet Khela ([Colab link](https://colab.research.google.com/drive/1kCYtYmExgO7-iXjSc_2Pj87BsRBJGZnp?usp=sharing)).

## Report

Build the PDF:

```bash
pdflatex main.tex
pdflatex main.tex   # second run for references
```

## Author

David Goh — University of Toronto
