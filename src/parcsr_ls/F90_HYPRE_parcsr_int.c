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

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int hypre_ParVectorSize( void *x );
NALU_HYPRE_Int aux_maskCount( NALU_HYPRE_Int n, hypre_F90_Int *mask );
void aux_indexFromMask( NALU_HYPRE_Int n, hypre_F90_Int *mask, hypre_F90_Int *index );

/*--------------------------------------------------------------------------
 * hypre_ParSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parsetrandomvalues, NALU_HYPRE_PARSETRANDOMVALUES)
(hypre_F90_Obj *v,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParVectorSetRandomValues(
                hypre_F90_PassObj (NALU_HYPRE_ParVector, v),
                hypre_F90_PassInt (seed)));
}

/*--------------------------------------------------------------------------
 * hypre_ParPrintVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parprintvector, NALU_HYPRE_PARPRINTVECTOR)
(hypre_F90_Obj *v,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorPrint(
                (hypre_ParVector *) v,
                (char *)            file));
}

/*--------------------------------------------------------------------------
 * hypre_ParReadVector
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parreadvector, NALU_HYPRE_PARREADVECTOR)
(hypre_F90_Comm *comm,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = 0;

   hypre_ParReadVector(
      hypre_F90_PassComm (comm),
      (char *) file );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsize, NALU_HYPRE_PARVECTORSIZE)
(hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorSize(
                (void *) x) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMultiVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorprint, NALU_HYPRE_PARCSRMULTIVECTORPRINT)
(hypre_F90_Obj *x,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMultiVectorPrint(
                (void *)       x,
                (char *) file));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMultiVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmultivectorread, NALU_HYPRE_PARCSRMULTIVECTORREAD)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *ii,
 char *file,
 hypre_F90_Int *ierr)
{
   *ierr = 0;

   NALU_HYPRE_ParCSRMultiVectorRead(
      hypre_F90_PassComm (comm),
      (void *)       ii,
      (char *) file );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_tempparcsrsetupinterprete, NALU_HYPRE_TEMPPARCSRSETUPINTERPRETE)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_TempParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_TempParCSRSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupinterpreter, NALU_HYPRE_PARCSRSETUPINTERPRETER)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrsetupmatvec, NALU_HYPRE_PARCSRSETUPMATVEC)
(hypre_F90_Obj *mv,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRSetupMatvec(
                hypre_F90_PassObjRef (NALU_HYPRE_MatvecFunctions, mv)));
}
#ifdef __cplusplus
}
#endif
