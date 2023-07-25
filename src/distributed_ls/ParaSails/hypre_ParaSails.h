/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * nalu_hypre_ParaSails.h header file.
 *
 *****************************************************************************/

#include "NALU_HYPRE_distributed_matrix_protos.h"
#include "../../IJ_mv/NALU_HYPRE_IJ_mv.h"

typedef void *nalu_hypre_ParaSails;

NALU_HYPRE_Int nalu_hypre_ParaSailsCreate(MPI_Comm comm, nalu_hypre_ParaSails *obj);
NALU_HYPRE_Int nalu_hypre_ParaSailsDestroy(nalu_hypre_ParaSails ps);
NALU_HYPRE_Int nalu_hypre_ParaSailsSetup(nalu_hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels,
  NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal, NALU_HYPRE_Int logging);
NALU_HYPRE_Int nalu_hypre_ParaSailsSetupPattern(nalu_hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels, 
  NALU_HYPRE_Int logging);
NALU_HYPRE_Int nalu_hypre_ParaSailsSetupValues(nalu_hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal, 
  NALU_HYPRE_Int logging);
NALU_HYPRE_Int nalu_hypre_ParaSailsApply(nalu_hypre_ParaSails ps, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v);
NALU_HYPRE_Int nalu_hypre_ParaSailsApplyTrans(nalu_hypre_ParaSails ps, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v);
NALU_HYPRE_Int nalu_hypre_ParaSailsBuildIJMatrix(nalu_hypre_ParaSails obj, NALU_HYPRE_IJMatrix *pij_A);
