/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "NALU_HYPRE_distributed_matrix_types.h"

#ifdef PETSC_AVAILABLE
/* NALU_HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
NALU_HYPRE_Int NALU_HYPRE_ConvertPETScMatrixToDistributedMatrix (Mat PETSc_matrix,
                                                       NALU_HYPRE_DistributedMatrix *DistributedMatrix );
#endif

/* NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
NALU_HYPRE_Int NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix (NALU_HYPRE_ParCSRMatrix parcsr_matrix,
                                                        NALU_HYPRE_DistributedMatrix *DistributedMatrix );

