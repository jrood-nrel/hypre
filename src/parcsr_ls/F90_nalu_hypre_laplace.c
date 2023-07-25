/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*****************************************************************************
 *
 * NALU_HYPRE_par_laplace Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * GenerateLaplacian
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_generatelaplacian, NALU_HYPRE_GENERATELAPLACIAN)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Int *nx,
  nalu_hypre_F90_Int *ny,
  nalu_hypre_F90_Int *nz,
  nalu_hypre_F90_Int *P,
  nalu_hypre_F90_Int *Q,
  nalu_hypre_F90_Int *R,
  nalu_hypre_F90_Int *p,
  nalu_hypre_F90_Int *q,
  nalu_hypre_F90_Int *r,
  nalu_hypre_F90_RealArray *value,
  nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr   )

{
   *matrix = (nalu_hypre_F90_Obj)
             ( GenerateLaplacian(
                  nalu_hypre_F90_PassComm (comm),
                  nalu_hypre_F90_PassInt (nx),
                  nalu_hypre_F90_PassInt (ny),
                  nalu_hypre_F90_PassInt (nz),
                  nalu_hypre_F90_PassInt (P),
                  nalu_hypre_F90_PassInt (Q),
                  nalu_hypre_F90_PassInt (R),
                  nalu_hypre_F90_PassInt (p),
                  nalu_hypre_F90_PassInt (q),
                  nalu_hypre_F90_PassInt (r),
                  nalu_hypre_F90_PassRealArray (value) ) );

   *ierr = 0;
}
#ifdef __cplusplus
}
#endif
