#!/bin/bash
# progress for job 5477109: production (runs_final, 4) + energy study (runs_energy, 30)
while true; do
  nf=$(ls displacement/runs_final/results/*.json 2>/dev/null | wc -l); ne=$(ls displacement/runs_energy/results/*.json 2>/dev/null | wc -l)
  st=$(squeue -j 5477109 -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(grep -l -E "Traceback|out of memory" logs/*-5477109_*.err 2>/dev/null | wc -l)
  echo "production $nf/4 | energy $ne/30 | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/progress_fe.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
