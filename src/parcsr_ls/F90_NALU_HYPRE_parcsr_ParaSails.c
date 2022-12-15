/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParaSails Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailscreate, NALU_HYPRE_PARASAILSCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsdestroy, NALU_HYPRE_PARASAILSDESTROY)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetup, NALU_HYPRE_PARASAILSSETUP)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssolve, NALU_HYPRE_PARASAILSSOLVE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *b,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetParams
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetparams, NALU_HYPRE_PARASAILSSETPARAMS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *thresh,
  nalu_hypre_F90_Int *nlevels,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetParams(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (thresh),
                nalu_hypre_F90_PassInt (nlevels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetThresh,  NALU_HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetthresh, NALU_HYPRE_PARASAILSSETTHRESH)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *thresh,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetThresh(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (thresh) ) );
}


void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetthresh, NALU_HYPRE_PARASAILSGETTHRESH)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *thresh,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetThresh(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (thresh) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetNlevels,  NALU_HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetnlevels, NALU_HYPRE_PARASAILSSETNLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nlevels,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetNlevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (nlevels)) );
}


void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetnlevels, NALU_HYPRE_PARASAILSGETNLEVELS)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *nlevels,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetNlevels(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (nlevels)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetFilter, NALU_HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetfilter, NALU_HYPRE_PARASAILSSETFILTER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *filter,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetFilter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (filter)  ) );
}


void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetfilter, NALU_HYPRE_PARASAILSGETFILTER)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *filter,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetFilter(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (filter)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetSym, NALU_HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetsym, NALU_HYPRE_PARASAILSSETSYM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *sym,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetSym(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (sym)     ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetsym, NALU_HYPRE_PARASAILSGETSYM)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *sym,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetSym(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (sym)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLoadbal, NALU_HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetloadbal, NALU_HYPRE_PARASAILSSETLOADBAL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *loadbal,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetLoadbal(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassReal (loadbal) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetloadbal, NALU_HYPRE_PARASAILSGETLOADBAL)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Real *loadbal,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetLoadbal(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassRealRef (loadbal) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetReuse, NALU_HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetreuse, NALU_HYPRE_PARASAILSSETREUSE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *reuse,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetReuse(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (reuse) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetreuse, NALU_HYPRE_PARASAILSGETREUSE)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *reuse,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetReuse(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (reuse) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLogging, NALU_HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailssetlogging, NALU_HYPRE_PARASAILSSETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (logging) ) );
}

void
nalu_hypre_F90_IFACE(nalu_hypre_parasailsgetlogging, NALU_HYPRE_PARASAILSGETLOGGING)
( nalu_hypre_F90_Obj *solver,
  nalu_hypre_F90_Int *logging,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetLogging(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassIntRef (logging) ) );
}

#ifdef __cplusplus
}
#endif
