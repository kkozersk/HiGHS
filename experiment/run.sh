#!/bin/sh
#$ -N experiments
#$ -cwd
#$ -l h_rss=128G
#$ -l h_vmem=128G
#$ -l h_rt=08:00:00

algo="standard"
while IFS=, read -r c1 c2 c3 c4 c5 c6
do
        echo "I got:$c1|$c2|$c3|$c4|$c5|$c6"
        export COREFILE=$c1 STOCHFILE=$c3 TIMEFILE=$c2 NOTE=$c4 EXPECTED=$c5 SUBLB=$c6 ALGORITHM=$algo
        ../build/bin/experiments 2>&1
done < tests.csv
