# Neural Synchrony, Anxiety, & Temperament — Public Repository

This repository hosts the public, reproducible pieces of the project examining parent–child neural synchrony during a mildly stressful interaction (DB-DOS: BioSync) and its links with child temperament and anxiety-relevant processes.

## What’s here

- **End-to-end analysis scripts**
  - `step 1_fnirs-synchrony-eDOC.ipynb` — pre-processing/feature building for synchrony on eDOC data (Jupyter Notebook).
  - `step 1b_fnirs-synchrony-real-dyads only.ipynb` — same pipeline constrained to real dyads only (Jupyter Notebook).
  - `step_2_tidy_imputation.Rmd` (+ rendered `step_2_tidy_imputation.html`) — tidy data assembly and imputation (R Markdown).
  - `step_3_synchrony-analysis.Rmd` — statistical modeling and figure/table generation (R Markdown).

- **Project assets**
  - `data/` — small example data, dictionaries, and/or derived public data artifacts.
  - `figures/` — rendered figures exported from analysis steps.
  - `tables/` — rendered tables (e.g., model summaries, descriptives).
  - `packages/` — helper functions and package scaffolding used across steps.
  - `LCBDtools/` — utility code (Local Common Best-Development tools) reused in notebooks/Rmds.

See the folder READMEs for details.

## Quick start

### Option A: R-first (to run the Rmd steps)

1. **Install R (≥4.x) & RStudio**  
2. **Install packages** (once per machine)
   ```r
   install.packages(c(
     "tidyverse","readr","here","janitor","knitr","rmarkdown",
     "broom","broom.mixed","lme4","lmerTest","glue","patchwork",
     "ggplot2","scales","mice"
   ))
   ```

3. **Render**  
   ```r
   rmarkdown::render("step_2_tidy_imputation.Rmd")
   rmarkdown::render("step_3_synchrony-analysis.Rmd")
   ```

### Option B: Python-first (to run the notebooks)

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  
   pip install -U pip jupyter numpy pandas matplotlib scipy
   ```

2. **Launch Jupyter**
   ```bash
   jupyter lab
   ```

## Reproducibility & order of operations

1. **Step 1/1b (notebooks)**: compute/prepare synchrony features.  
2. **Step 2 (Rmd)**: consolidate, tidy, and impute.  
3. **Step 3 (Rmd)**: fit models; export figures/tables to `figures/` and `tables/`.

## Data availability

Only de-identified/demo or derived data should live in `data/`. Any raw or sensitive data are **not** included.

## License

MIT License © 2024–present Joscelin Rocha-Hidalgo.
