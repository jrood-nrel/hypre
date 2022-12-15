/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRint Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int nalu_hypre_ParVectorSize( void *x );
NALU_HYPRE_Int aux_maskCount( NALU_HYPRE_Int n, nalu_hypre_F90_Int *mask );
void aux_indexFromMask( NALU_HYPRE_Int n, nalu_hypre_F90_Int *mask, nalu_hypre_F90_Int *index );

/*--------------------------------------------------------------------------
 * nalu_hypre_ParSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parsetrandomvalues, NALU_HYPRE_PARSETRANDOMVALUES)
(nalu_hypre_F90_Obj *v,
 nalu_hypre_F90_Int *seed,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorSetRandomValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, v),
                nalu_hypre_F90_PassInt (seed)));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParPrintVector
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parprintvector, NALU_HYPRE_PARPRINTVECTOR)
(nalu_hypre_F90_Obj *v,
 char *file,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorPrint(
                (nalu_hypre_ParVector *) v,
                (char *)            file));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParReadVector
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parreadvector, NALU_HYPRE_PARREADVECTOR)
(nalu_hypre_F90_Comm *comm,
 char *file,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = 0;

   nalu_hypre_ParReadVector(
      nalu_hypre_F90_PassComm (comm),
      (char *) file );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorsize, NALU_HYPRE_PARVECTORSIZE)
(nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorSize(
                (void *) x) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMultiVectorPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmultivectorprint, NALU_HYPRE_PARCSRMULTIVECTORPRINT)
(nalu_hypre_F90_Obj *x,
 char *file,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMultiVectorPrint(
                (void *)       x,
                (char *) file));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMultiVectorRead
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmultivectorread, NALU_HYPRE_PARCSRMULTIVECTORREAD)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *ii,
 char *file,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = 0;

   NALU_HYPRE_ParCSRMultiVectorRead(
      nalu_hypre_F90_PassComm (comm),
      (void *)       ii,
      (char *) file );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_tempparcsrsetupinterprete, NALU_HYPRE_TEMPPARCSRSETUPINTERPRETE)
(nalu_hypre_F90_Obj *i,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_TempParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrsetupinterpreter, NALU_HYPRE_PARCSRSETUPINTERPRETER)
(nalu_hypre_F90_Obj *i,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSetupMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrsetupmatvec, NALU_HYPRE_PARCSRSETUPMATVEC)
(nalu_hypre_F90_Obj *mv,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRSetupMatvec(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_MatvecFunctions, mv)));
}
#ifdef __cplusplus
}
#endif
