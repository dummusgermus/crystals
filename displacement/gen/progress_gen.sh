#!/bin/bash
# progress for point-defect data generation gen_v1 (593 compounds x 5 cells): job 5523579 + continuation 5542084
JOBS=5523579,5542084
while true; do
  n=$(ls displacement/gen_data/gen_v1/SIMULATIONS/*/gen.json 2>/dev/null | wc -l)
  ne=$(cat logs/jara-gen-5523579_*.out logs/jara-gen-5542084_*.out 2>/dev/null | grep -c "^error")
  st=$(squeue -j $JOBS -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "gen_v1 $n/593 compounds | errors: $ne | slurm: ${st:-none} ($(date '+%d.%m %H:%M'))" >> displacement/gen_data/progress_gen.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 600
done
