#!/bin/bash

executables_dir=$1

for filename in $executables_dir/*; do
    [ -e "$filename" ] || continue
    submission=$(sbatch $filename)
    echo $submission
done
