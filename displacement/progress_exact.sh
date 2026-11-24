#!/bin/bash
# progress for job 5495572: exact point energy labels (runs_exact, 8)
J=5495572
while true; do
  n=$(ls displacement/runs_exact/results/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $J -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(grep -l -E "Traceback|out of memory" logs/*-${J}_*.err 2>/dev/null | wc -l)
  echo "exact $n/8 | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/progress_exact.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
