#!/bin/bash
# progress for job 5523579: point-defect data generation gen_v1 (593 compounds x 5 cells)
while true; do
  n=$(ls displacement/gen_data/gen_v1/SIMULATIONS/*/gen.json 2>/dev/null | wc -l)
  ne=$(grep -h "^error" logs/jara-gen-5523579_*.out 2>/dev/null | wc -l)
  st=$(squeue -j 5523579 -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "gen_v1 $n/593 compounds | errors: $ne | slurm: ${st:-none} ($(date '+%d.%m %H:%M'))" >> displacement/gen_data/progress_gen.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 600
done
