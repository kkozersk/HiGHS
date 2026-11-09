#!/bin/sh
# 
# for algo in "standard" "qp" "pd" "ipm"
# do
algo="standard"
    while IFS=, read -r c1 c2 c3 c4 c5 c6
    do
        echo "I got:$c1|$c2|$c3|$c4|$c5|$c6"
        export COREFILE=$c1 STOCHFILE=$c3 TIMEFILE=$c2 NOTE=$c4 EXPECTED=$c5 SUBLB=$c6 ALGORITHM=$algo
        build/main 2>&1
    done < tests.csv
echo $ALGORITHM
# done

