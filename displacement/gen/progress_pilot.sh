#!/bin/bash
# progress for job 5485929: point-defect generation pilot (30 compounds, ~150 cells of 3000 atoms)
while true; do
  n=$(cat logs/jara-pilot-5485929_*.out 2>/dev/null | grep -c "^  [0-9]")
  st=$(squeue -j 5485929 -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "pilot: $n defect cells relaxed | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/gen_data/progress_pilot.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 120
done
