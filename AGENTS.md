# Contributor instructions

Keep this a minimal digits/SVM example: three folds, fixed model parameters,
three Python commands and thin Bash/Slurm wrappers. Do not introduce workflow
frameworks, tracking services, custom image builds or automatic retries.

Preserve the explicit Python interfaces in PLAN.md. Python training is unaware
of folds and Slurm. Keep private settings outside the repository. Never connect
to a cluster or submit real jobs; use explicit local scheduler mocks for tests.
Compute jobs must not install packages or download data/images.

Run the local ML tests and per-file Bash syntax checks when relevant. Clearly
separate host checks, container checks, scheduler mocks and human-only Slurm
validation. Do not claim an unavailable check passed. The choice of agent skill
is a developer preference and is not required to run or maintain this project.
