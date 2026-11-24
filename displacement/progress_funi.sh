#!/bin/bash
# progress for job 5482985: unified production candidates (runs_final_unified, 6)
J=5482985
while true; do
  n=$(ls displacement/runs_final_unified/results/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $J -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(grep -l -E "Traceback|out of memory" logs/*-${J}_*.err 2>/dev/null | wc -l)
  echo "final-unified $n/6 | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/progress_funi.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
