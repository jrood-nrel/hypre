/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * nalu_hypre_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetobject, NALU_HYPRE_IJMATRIXSETOBJECT)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *object,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_IJMatrixSetObject(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                (void *)         *object  ) );
}

#ifdef __cplusplus
}
#endif
