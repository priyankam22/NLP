#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Usage: $0 <seed>"
    exit 1
fi

# Call the python script

python3 ngram_language_model.py $1 RUN
