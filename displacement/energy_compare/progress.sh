#!/bin/bash
# progress for job 5548969: CGCNN / GT energy models on our compound splits (24 runs)
while true; do
  n=$(find displacement/energy_compare/preds -name '*.json' -newer displacement/energy_compare/tasks.txt | wc -l)
  st=$(squeue -j 5548969 -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "energy-compare $n/24 runs done | slurm: ${st:-none} ($(date '+%d.%m %H:%M'))" >> displacement/energy_compare/progress.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 600
done
