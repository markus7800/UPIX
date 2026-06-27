#!/bin/bash

N=$1
lists=$2
use_flags=$3

flags=""
if [ "$use_flags" = "true" ]; then
    flags="-num-recursive-calls -show-size"
fi

if [ "$lists" = "true" ]; then
    time dice -determinism -eager-eval -flip-lifting $flags -recursion-limit $((N + 1)) -max-list-length $((N + 1)) urn${N}_lists.dice
else
    time dice $flags urn${N}.dice
fi