# CLOUD RCA agent

This is a small root cause analysis project using the ITBench SRE dataset.

The project has two main parts:
1. Graph module - Builds a graph from the scenario data and ranks the most likely root cause candidates.

2. Agent - Takes the top 3 candidates from the graph module, looks at the graph evidence and tries to choose the most likely root cause.

ITBench repo - https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent

