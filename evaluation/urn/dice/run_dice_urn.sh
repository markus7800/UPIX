#!/bin/bash

N=$1

time dice -determinism -eager-eval -flip-lifting -num-recursive-calls -show-size -recursion-limit $((N + 1)) -max-list-length $((N + 1)) urn$N.dice