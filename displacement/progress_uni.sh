#!/bin/bash
# progress for job 5482173: unified point+planar+mixed study (runs_unified, 48)
J=5482173
while true; do
  n=$(ls displacement/runs_unified/results/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $J -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(grep -l -E "Traceback|out of memory" logs/*-${J}_*.err 2>/dev/null | wc -l)
  echo "unified $n/48 | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/progress_uni.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
