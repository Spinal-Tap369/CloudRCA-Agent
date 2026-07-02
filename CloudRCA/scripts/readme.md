Download scenarios - hf download ibm-research/ITBench-Lite --repo-type dataset --include "snapshots/sre/v0.2-*/Scenario-1/*" --include "snapshots/sre/v0.2-*/Scenario-2/*" --include "snapshots/sre/v0.2-*/Scenario-5/*" --local-dir data/ITBench-Lite --dry-run

python .\CloudRCA\scripts\graph_check_all.py ".\data\ITBench-Lite\snapshots\sre\v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7"

python .\CloudRCA\scripts\download_missing_sre_scenarios.py --retries 4 --min-delay 10 --max-delay 25 --max-workers 1