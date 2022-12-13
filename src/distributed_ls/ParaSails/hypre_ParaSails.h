/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_ParaSails.h header file.
 *
 *****************************************************************************/

#include "NALU_HYPRE_distributed_matrix_protos.h"
#include "../../IJ_mv/NALU_HYPRE_IJ_mv.h"

typedef void *hypre_ParaSails;

NALU_HYPRE_Int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj);
NALU_HYPRE_Int hypre_ParaSailsDestroy(hypre_ParaSails ps);
NALU_HYPRE_Int hypre_ParaSailsSetup(hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels,
  NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal, NALU_HYPRE_Int logging);
NALU_HYPRE_Int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Int sym, NALU_HYPRE_Real thresh, NALU_HYPRE_Int nlevels, 
  NALU_HYPRE_Int logging);
NALU_HYPRE_Int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
  NALU_HYPRE_DistributedMatrix distmat, NALU_HYPRE_Real filter, NALU_HYPRE_Real loadbal, 
  NALU_HYPRE_Int logging);
NALU_HYPRE_Int hypre_ParaSailsApply(hypre_ParaSails ps, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v);
NALU_HYPRE_Int hypre_ParaSailsApplyTrans(hypre_ParaSails ps, NALU_HYPRE_Real *u, NALU_HYPRE_Real *v);
NALU_HYPRE_Int hypre_ParaSailsBuildIJMatrix(hypre_ParaSails obj, NALU_HYPRE_IJMatrix *pij_A);
