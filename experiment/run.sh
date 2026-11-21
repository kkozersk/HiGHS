#!/bin/sh
#$ -N experiments
#$ -cwd
#$ -l h_rss=128G
#$ -l h_rt=48:00:00

cp ../build/bin/experiments .
export ALGORITHM="standard" EPS="1e-6" BASELOGFILE="../../results"
export RECENTRING="1e-5" GAMMA="0.5" OMEGA="0.9" TL="0.49" TU="0.51" D="1."
while IFS=, read -r c1 c2 c3 c4 c5 c6
do
        echo "I got:$c1|$c2|$c3|$c4|$c5|$c6"
        export COREFILE=$c1 STOCHFILE=$c3 TIMEFILE=$c2 NOTE=$c4 EXPECTED=$c5 SUBLB=$c6
        ./experiments 2>&1
done < tests.csv
