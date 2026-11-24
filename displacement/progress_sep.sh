#!/bin/bash
# progress for job 5485710: separate vs unified heads (runs_separate, 18)
J=5485710
while true; do
  n=$(ls displacement/runs_separate/results/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $J -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(grep -l -E "Traceback|out of memory" logs/*-${J}_*.err 2>/dev/null | wc -l)
  echo "separate $n/18 | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/progress_sep.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
