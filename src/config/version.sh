#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

nalu_hypre_version="2.26.0"
nalu_hypre_reldate="2022/10/14"

nalu_hypre_major=`echo $nalu_hypre_version | cut -d. -f 1`
nalu_hypre_minor=`echo $nalu_hypre_version | cut -d. -f 2`
nalu_hypre_patch=`echo $nalu_hypre_version | cut -d. -f 3`

let nalu_hypre_number="$nalu_hypre_major*10000 + $nalu_hypre_minor*100 + $nalu_hypre_patch"

