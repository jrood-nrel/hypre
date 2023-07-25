/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structvectorsetrandomvalu, NALU_HYPRE_STRUCTVECTORSETRANDOMVALU)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_StructVectorSetRandomValues(
                (nalu_hypre_StructVector *) vector,
                nalu_hypre_F90_PassInt (seed) ));
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsetrandomvalues, NALU_HYPRE_STRUCTSETRANDOMVALUES)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_StructSetRandomValues(
                (nalu_hypre_StructVector *) vector,
                nalu_hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSetupInterpreter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsetupinterpreter, NALU_HYPRE_STRUCTSETUPINTERPRETER)
(nalu_hypre_F90_Obj *i,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSetupMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structsetupmatvec, NALU_HYPRE_STRUCTSETUPMATVEC)
(nalu_hypre_F90_Obj *mv,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_StructSetupMatvec(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_MatvecFunctions, mv)));
}

#ifdef __cplusplus
}
#endif
