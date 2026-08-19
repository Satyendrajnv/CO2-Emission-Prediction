# CO₂ Emission Dataset Schema & Feature Description

This directory contains the 36-sample vehicle emission dataset used for the small-sample regression study.

---

## Dataset Overview

- **Sample Size**: 36 vehicle observations.
- **Task**: Supervised Continuous Regression.
- **Target Variable**: `CO2` (Emissions in g/km).

---

## Column Schema

| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `Car` | String | - | Vehicle manufacturer brand (e.g., `Toyota`, `Audi`, `Volvo`) |
| `Model` | String | - | Vehicle model designation (e.g., `Aygo`, `A4`, `XC70`) |
| `Volume` | Integer | $\text{cm}^3$ / cc | Engine displacement volume |
| `Weight` | Integer | kg | Vehicle curb weight |
| `CO2` | Integer | g/km | **Target**: Carbon dioxide tailpipe emissions |

---

## Methodological Limitations

- **Small Sample Boundary**: With only 36 observations, sample variance in cross-validation estimates is inherently higher than in large automotive datasets.
- **Categorical Cardinality**: High cardinality in vehicle models (`Model`) limits one-hot encoding usefulness due to degrees of freedom constraints.
