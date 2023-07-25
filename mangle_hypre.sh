#!/bin/bash

find src -type f -exec sed -i.bak 's/HYPRE_/NALU_HYPRE_/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find src -type f -exec sed -i.bak 's/hypre_/nalu_hypre_/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find src -type f -exec sed -i.bak 's/HYPRE.h/NALU_HYPRE.h/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
