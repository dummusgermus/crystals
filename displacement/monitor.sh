#!/bin/bash
# usage: displacement/monitor.sh <jobid> <jobname> <n_runs> <results_dir> <logfile>
J=$1; NAME=$2; N=$3; RES=$4; LOG=$5
half=0
while true; do
  n=$(ls $RES/*.json 2>/dev/null | wc -l)
  st=$(squeue -j $J -h -o "%T" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "$n/$N runs done | slurm: ${st:-none} ($(date +%H:%M))" >> $LOG
  f=$(grep -l -E "Traceback|out of memory" logs/${NAME}-${J}_*.err 2>/dev/null | head -1)
  if [ -n "$f" ]; then echo "ERROR in $f"; break; fi
  if [ $half -eq 0 ] && [ "$n" -ge $((N/2)) ]; then half=1; echo "HALF $(date +%H:%M)"; fi
  [ -z "$(squeue -j $J -h 2>/dev/null)" ] && { echo "finished"; break; }
  sleep 120
done
