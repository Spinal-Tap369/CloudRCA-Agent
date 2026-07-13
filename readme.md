# CLOUD RCA agent

This is an agent based RCA(Root Cause Analysis) project using the ITBench SRE dataset.

The project has two main parts:
1. Graph module - Ingests TSV/JSON scenario artifacts from ITBench snapshots and builds typed nodes for Kubernetes resources such as Pods, Deployments, Services, ConfigMaps, NetworkPolicies, HPAs, Namespaces, and Chaos Mesh objects. It also adds typed causal edges from ownership, selectors, service routing, Kubernetes events, telemetry symptoms, chaos schedules, and configuration changes. 

2. Agent - Takes the top 3 candidates from the graph module, looks at the graph evidence and runs a structured LangGraph RCA workflow: build case, normalize candidates, inspect root/rivals, audit causal path, draft diagnosis, validate, and repair/fallback if needed. The agent returns root cause entities, evidence, reasoning summary and remediation.

3. Evaluation focuses on raw RCA correctness and uses ITBench's ground_truth.yaml files. 

## SCRIPTS  - 

1. Download scenarios -
python .\CloudRCA\scripts\download_scenarios.py --retries 4 --min-delay 10 --max-delay 25 --max-workers 1

2. Evaluate graph coverage -
python .\CloudRCA\scripts\graph_check_all.py ".\data\ITBench-Lite\snapshots\sre\v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7"

3. Diagnose a scenario - 
python .\CloudRCA\scripts\run_scenario.py <scenario dir> --output-dir <outpit dir>

4. Evaluate a scenario diagnosis -
python .\CloudRCA\scripts\evaluate_result.py <scenario dir> <scenario-results.json path>

ITBench - https://arxiv.org/abs/2502.05352
ITBench Github repo - https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent

Work in Progress -> To extend the RCA agent into a remediation agent that can plan, execute, and verify corrective actions in an ITBench environment, enabling full ITBench-style SRE evaluation beyond offline diagnosis.