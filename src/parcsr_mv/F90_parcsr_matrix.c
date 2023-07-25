/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParCSRMatrix Fortran interface to macros
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGlobalNumRows
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixglobalnumrows, NALU_HYPRE_PARCSRMATRIXGLOBALNUMROWS)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *num_rows,
  nalu_hypre_F90_Int *ierr      )
{
   *num_rows = (nalu_hypre_F90_BigInt)
               ( nalu_hypre_ParCSRMatrixGlobalNumRows(
                    (nalu_hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixRowStarts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixrowstarts, NALU_HYPRE_PARCSRMATRIXROWSTARTS)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *row_starts,
  nalu_hypre_F90_Int *ierr      )
{
   *row_starts = (nalu_hypre_F90_Obj)
                 ( nalu_hypre_ParCSRMatrixRowStarts(
                      (nalu_hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

#ifdef __cplusplus
}
#endif
