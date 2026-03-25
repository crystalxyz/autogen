#!/bin/bash
# Rerun 11 two-turn-nc benchmarks after scenario.py timing fix
# (ChatCompletionClient.load_component moved before start_time)
# Run from: /home/xz957/autogen/python/packages/agbench
#
# Config 16 (4nt-14t) ports changed from 30021/30022 to 30039/30040
# to avoid conflict with config 15 (1.7nt-14t).

cd /home/xz957/autogen/python/packages/agbench

# Small turn1 models (0.6B, 1.7B) — can run concurrently, no port overlap
sbatch scripts/start.sh scripts/configs/config-twoturn-14.yaml outputs/two-turn-nc-1.7nt-1.7t-v2   # 1.7nt-1.7t  ports 30019/30020
sbatch scripts/start.sh scripts/configs/config-twoturn-15.yaml outputs/two-turn-nc-1.7nt-14t-v2   # 1.7nt-14t  ports 30021/30022

# Medium turn1 models (4B)
sbatch scripts/start.sh scripts/configs/config-twoturn-11.yaml outputs/two-turn-nc-4nt-8t-v2      # 4nt-8t     ports 30011/30012
sbatch scripts/start.sh scripts/configs/config-twoturn-12.yaml outputs/two-turn-nc-4nt-1.7t-v2    # 4nt-1.7t   ports 30013/30014
sbatch scripts/start.sh scripts/configs/config-twoturn-16.yaml outputs/two-turn-nc-4nt-14t-v2     # 4nt-14t    ports 30039/30040

# Large turn1 models (8B)
sbatch scripts/start.sh scripts/configs/config-twoturn-13.yaml outputs/two-turn-nc-8nt-1.7t-v2    # 8nt-1.7t   ports 30015/30016
sbatch scripts/start.sh scripts/configs/config-twoturn-17.yaml outputs/two-turn-nc-8nt-14t-v2     # 8nt-14t    ports 30023/30024

# 14B turn1 models
sbatch scripts/start.sh scripts/configs/config-twoturn-18.yaml outputs/two-turn-nc-14nt-1.7t-v2   # 14nt-1.7t  ports 30031/30032
sbatch scripts/start.sh scripts/configs/config-twoturn-19.yaml outputs/two-turn-nc-14nt-4t-v2     # 14nt-4t    ports 30033/30034
sbatch scripts/start.sh scripts/configs/config-twoturn-20.yaml outputs/two-turn-nc-14nt-8t-v2     # 14nt-8t    ports 30035/30036
sbatch scripts/start.sh scripts/configs/config-twoturn-21.yaml outputs/two-turn-nc-14nt-14t-v2    # 14nt-14t   ports 30037/30038
