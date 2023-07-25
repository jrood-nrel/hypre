#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

#=============================================================================
# Check the NALU_HYPRE_DEVELOP variables
#=============================================================================

grep "Using NALU_HYPRE_DEVELOP_STRING" ${TNAME}.out.1 > ${TNAME}.testdata

echo -n > ${TNAME}.testdatacheck
if [ -d ../../../.git ]; then
  DEVSTRING=`git describe --match 'v*' --long --abbrev=9 2>/dev/null`
  DEVNUMBER=`echo $DEVSTRING | awk -F- '{print $2}'`
  DEVBRANCH=`git rev-parse --abbrev-ref HEAD`
  if [ -n "$DEVSTRING" ]; then
    if [ "$DEVBRANCH" != "master" ]; then
      echo "Using NALU_HYPRE_DEVELOP_STRING: $DEVSTRING (branch $DEVBRANCH; not the develop branch)" \
       > ${TNAME}.testdatacheck
    else
      echo "Using NALU_HYPRE_DEVELOP_STRING: $DEVSTRING (branch $DEVBRANCH; the develop branch)" \
       > ${TNAME}.testdatacheck
    fi
  fi
fi
diff ${TNAME}.testdata ${TNAME}.testdatacheck >&2

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
