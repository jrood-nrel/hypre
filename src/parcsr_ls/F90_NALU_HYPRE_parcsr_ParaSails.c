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

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailscreate, NALU_HYPRE_PARASAILSCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailsdestroy, NALU_HYPRE_PARASAILSDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetup, NALU_HYPRE_PARASAILSSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssolve, NALU_HYPRE_PARASAILSSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetParams
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetparams, NALU_HYPRE_PARASAILSSETPARAMS)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetParams(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (thresh),
                hypre_F90_PassInt (nlevels) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetThresh,  NALU_HYPRE_ParaSailsGetThresh
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetthresh, NALU_HYPRE_PARASAILSSETTHRESH)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetThresh(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (thresh) ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetthresh, NALU_HYPRE_PARASAILSGETTHRESH)
( hypre_F90_Obj *solver,
  hypre_F90_Real *thresh,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetThresh(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (thresh) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetNlevels,  NALU_HYPRE_ParaSailsGetNlevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetnlevels, NALU_HYPRE_PARASAILSSETNLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetNlevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (nlevels)) );
}


void
hypre_F90_IFACE(hypre_parasailsgetnlevels, NALU_HYPRE_PARASAILSGETNLEVELS)
( hypre_F90_Obj *solver,
  hypre_F90_Int *nlevels,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetNlevels(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (nlevels)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetFilter, NALU_HYPRE_ParaSailsGetFilter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetfilter, NALU_HYPRE_PARASAILSSETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetFilter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (filter)  ) );
}


void
hypre_F90_IFACE(hypre_parasailsgetfilter, NALU_HYPRE_PARASAILSGETFILTER)
( hypre_F90_Obj *solver,
  hypre_F90_Real *filter,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetFilter(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (filter)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetSym, NALU_HYPRE_ParaSailsGetSym
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetsym, NALU_HYPRE_PARASAILSSETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetSym(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (sym)     ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetsym, NALU_HYPRE_PARASAILSGETSYM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *sym,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetSym(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (sym)     ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLoadbal, NALU_HYPRE_ParaSailsGetLoadbal
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetloadbal, NALU_HYPRE_PARASAILSSETLOADBAL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *loadbal,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetLoadbal(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassReal (loadbal) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetloadbal, NALU_HYPRE_PARASAILSGETLOADBAL)
( hypre_F90_Obj *solver,
  hypre_F90_Real *loadbal,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetLoadbal(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassRealRef (loadbal) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetReuse, NALU_HYPRE_ParaSailsGetReuse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetreuse, NALU_HYPRE_PARASAILSSETREUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *reuse,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetReuse(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (reuse) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetreuse, NALU_HYPRE_PARASAILSGETREUSE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *reuse,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetReuse(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (reuse) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParaSailsSetLogging, NALU_HYPRE_ParaSailsGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parasailssetlogging, NALU_HYPRE_PARASAILSSETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsSetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (logging) ) );
}

void
hypre_F90_IFACE(hypre_parasailsgetlogging, NALU_HYPRE_PARASAILSGETLOGGING)
( hypre_F90_Obj *solver,
  hypre_F90_Int *logging,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParaSailsGetLogging(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassIntRef (logging) ) );
}

#ifdef __cplusplus
}
#endif
