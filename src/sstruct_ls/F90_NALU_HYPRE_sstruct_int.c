/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructInt Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fortran.h"
#include "NALU_HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructPVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructpvectorsetrandomva, NALU_HYPRE_SSTRUCTPVECTORSETRANDOMVA)
(nalu_hypre_F90_Obj *pvector,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_SStructPVectorSetRandomValues(
                (nalu_hypre_SStructPVector *) pvector,
                nalu_hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructvectorsetrandomval, NALU_HYPRE_SSTRUCTVECTORSETRANDOMVAL)
(nalu_hypre_F90_Obj *vector,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_SStructVectorSetRandomValues(
                (nalu_hypre_SStructVector *) vector,
                nalu_hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructsetrandomvalues, NALU_HYPRE_SSTRUCTSETRANDOMVALUES)
(nalu_hypre_F90_Obj *v,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_SStructSetRandomValues(
                (void *) v, nalu_hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructVectorSetupInterpreter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructsetupinterpreter, NALU_HYPRE_SSTRUCTSETUPINTERPRETER)
(nalu_hypre_F90_Obj *i,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 *  NALU_HYPRE_SStructSetupMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructsetupmatvec, NALU_HYPRE_SSTRUCTSETUPMATVEC)
(nalu_hypre_F90_Obj *mv,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SStructSetupMatvec(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_MatvecFunctions, mv)));
}

#ifdef __cplusplus
}
#endif
