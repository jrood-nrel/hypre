/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_SMGCreateRestrictOp( nalu_hypre_StructMatrix *A,
                           nalu_hypre_StructGrid   *cgrid,
                           NALU_HYPRE_Int           cdir  )
{
   nalu_hypre_StructMatrix *R = NULL;

   return R;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetupRestrictOp( nalu_hypre_StructMatrix *A,
                          nalu_hypre_StructMatrix *R,
                          nalu_hypre_StructVector *temp_vec,
                          NALU_HYPRE_Int           cdir,
                          nalu_hypre_Index         cindex,
                          nalu_hypre_Index         cstride  )
{
   return nalu_hypre_error_flag;
}
