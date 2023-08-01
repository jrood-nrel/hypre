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
find src -type f -exec sed -i.bak 's/HYPREf.h/NALU_HYPREf.h/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find src -type f -exec sed -i.bak 's/[[:<:]]HYPRE[[:>:]]/NALU_HYPRE/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find src -type f -name '*hypre*' -exec bash -c 'git mv $0 ${0/hypre/nalu_hypre}' {} \;
find src -type f -name '*HYPRE*' -exec bash -c 'git mv $0 ${0/HYPRE/NALU_HYPRE}' {} \;
git mv src/NALU_NALU_HYPRE.h src/NALU_HYPRE.h
git mv src/NALU_NALU_HYPREf.h src/NALU_HYPREf.h
find src -type f -exec sed -i.bak 's/NALU_NALU/NALU/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
