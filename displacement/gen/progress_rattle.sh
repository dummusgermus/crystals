#!/bin/bash
# progress for job 5496541: rattle symmetry-breaking test (13 cells x 5 relaxations)
while true; do
  n=$(grep -h "dE " logs/jara-rattle-5496541_*.out 2>/dev/null | wc -l)
  st=$(squeue -j 5496541 -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "rattle $n/65 relaxations | slurm: ${st:-none} ($(date +%H:%M))" >> displacement/gen_data/progress_rattle.log
  [ -z "$st" ] && { echo finished; break; }
  sleep 300
done
