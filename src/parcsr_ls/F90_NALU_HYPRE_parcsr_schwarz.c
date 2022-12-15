/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Schwarz Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzcreate, NALU_HYPRE_SCHWARZCREATE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzCreate(
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzdestroy, NALU_HYPRE_SCHWARZDESTROY)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetup, NALU_HYPRE_SCHWARZSETUP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetup(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsolve, NALU_HYPRE_SCHWARZSOLVE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *A,
 nalu_hypre_F90_Obj *b,
 nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSolve(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/
void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetvariant, NALU_HYPRE_SCHWARZSETVARIANT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *variant,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetVariant(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (variant) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetoverlap, NALU_HYPRE_SCHWARZSETOVERLAP)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *overlap,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetOverlap(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (overlap)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetdomaintype, NALU_HYPRE_SCHWARZSETDOMAINTYPE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *domain_type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetDomainType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassInt (domain_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetdomainstructure, NALU_HYPRE_SCHWARZSETDOMAINSTRUCTURE)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Obj *domain_structure,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetDomainStructure(
                nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                nalu_hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, domain_structure)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetnumfunctions, NALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Int *num_functions,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetNumFunctions(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassInt (num_functions) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetrelaxweight, NALU_HYPRE_SCHWARZSETRELAXWEIGHT)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_Real *relax_weight,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetRelaxWeight(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassReal (relax_weight)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_schwarzsetdoffunc, NALU_HYPRE_SCHWARZSETDOFFUNC)
(nalu_hypre_F90_Obj *solver,
 nalu_hypre_F90_IntArray *dof_func,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetDofFunc(
               nalu_hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               nalu_hypre_F90_PassIntArray (dof_func)  ));
}
#ifdef __cplusplus
}
#endif
