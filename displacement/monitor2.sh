#!/bin/bash
# usage: displacement/monitor2.sh <logfile> <results_dir> <n_runs> <jobid>...   (logs errors, exits when all jobs are gone)
LOG=$1; RES=$2; N=$3; shift 3; JOBS="$*"
while true; do
  n=$(ls $RES/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $(echo $JOBS | tr ' ' ',') -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  nerr=$(for j in $JOBS; do grep -l -E "Traceback|out of memory" logs/*-${j}_*.err 2>/dev/null; done | wc -l)
  echo "$n/$N runs | task logs with errors: $nerr | slurm: ${st:-none} ($(date +%H:%M))" >> $LOG
  [ -z "$st" ] && { echo "finished"; break; }
  sleep 300
done
