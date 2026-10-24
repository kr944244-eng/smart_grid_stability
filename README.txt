GRIDSENSE - SMART GRID STABILITY PREDICTION (Flask + ML)
=========================================================

FOLDER
  gridsense/
    app.py                 <- Flask back end (login, predictions, history, batch, PDF)
    train.py               <- trains the models (only needed if the model fails to load)
    requirements.txt
    models/                <- trained neural network (model.pkl) + metrics.json
    templates/             <- HTML pages
    static/css, static/js  <- design and scripts
    static/img/            <- put college_logo.png here
    static/sample_grids.csv <- sample file for Batch Upload

HOW TO RUN (VS Code terminal, inside the gridsense folder)
  1. pip install flask scikit-learn==1.9.1 pandas numpy joblib reportlab
  2. python app.py
  3. Browser opens http://127.0.0.1:5000
  4. Login:  admin / admin123   (or click "Create an account")

IF THE MODEL GIVES AN ERROR
  Put smart_grid_stability_augmented.csv in this folder and run:  python train.py
  Then run python app.py again.

PERSONALISE
  - Open app.py and edit the PROJECT section at the top (college, department, title, team, guide).
  - Put your college logo in static/img/ as college_logo.png
  - Restart app.py (CTRL+C, then python app.py) after editing.

FILES CREATED WHILE RUNNING
  gridsense.db      <- users and prediction history (delete it to reset everything)
  batch_results/    <- results of CSV uploads
