# Three-turn coding agent experiment
## Design
Based on the existing two-turn data from reports/csv/two_turn_nc_stats.csv, the idea is to push pareto frontier forward by adding an optional third turn. The metrics is mean latency vs accuracy. The idea is to identify points on existing pareto frontier and analyze if there's space to add a third model in between to reduce turn 2 latency significantly.

## Implementation
We need to create another copy of HumanEval AgentChatCopyNoCtx to enable 3-turn experiments on 3 models, and the config needs to be modified to run 3 sglang servers.

## Experiments
Model configs:
- 4nt -> 1.7t -> 14t
- 4nt -> 4t -> 14t
- 4nt -> 8t -> 14t
- 8nt -> 1.7t -> 14t
- 14nt -> 4t -> 14t
- 14nt -> 8t -> 14t

Number of trials: 3 per task

Hardware setup: 3 GPU, one per sglang server, using diferent port numbers
