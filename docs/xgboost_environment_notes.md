# XGBoost environment notes

The existing environment.yml does not list xgboost. In the existing activated
payment-guard-verify Conda environment, install the Python library via:

python -m pip install 'xgboost>=2.1,<4'

The research protocol records the installed xgboost version in local run.json.
A future environment.yml change should add pip and the above version constraint
once the Mac version is known and the research checkpoint is verified.
