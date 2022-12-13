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

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzcreate, NALU_HYPRE_SCHWARZCREATE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzCreate(
                hypre_F90_PassObjRef (NALU_HYPRE_Solver, solver)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzdestroy, NALU_HYPRE_SCHWARZDESTROY)
(hypre_F90_Obj *solver,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzDestroy(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetup, NALU_HYPRE_SCHWARZSETUP)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetup(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsolve, NALU_HYPRE_SCHWARZSOLVE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *A,
 hypre_F90_Obj *b,
 hypre_F90_Obj *x,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSolve(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, b),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetVariant
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_schwarzsetvariant, NALU_HYPRE_SCHWARZSETVARIANT)
(hypre_F90_Obj *solver,
 hypre_F90_Int *variant,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetVariant(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (variant) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetOverlap
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetoverlap, NALU_HYPRE_SCHWARZSETOVERLAP)
(hypre_F90_Obj *solver,
 hypre_F90_Int *overlap,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetOverlap(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (overlap)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomaintype, NALU_HYPRE_SCHWARZSETDOMAINTYPE)
(hypre_F90_Obj *solver,
 hypre_F90_Int *domain_type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetDomainType(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassInt (domain_type) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDomainStructure
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdomainstructure, NALU_HYPRE_SCHWARZSETDOMAINSTRUCTURE)
(hypre_F90_Obj *solver,
 hypre_F90_Obj *domain_structure,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_SchwarzSetDomainStructure(
                hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
                hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, domain_structure)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetNumFunctions
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetnumfunctions, NALU_HYPRE_SCHWARZSETNUMFUNCTIONS)
(hypre_F90_Obj *solver,
 hypre_F90_Int *num_functions,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetNumFunctions(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassInt (num_functions) ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetRelaxWeight
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetrelaxweight, NALU_HYPRE_SCHWARZSETRELAXWEIGHT)
(hypre_F90_Obj *solver,
 hypre_F90_Real *relax_weight,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetRelaxWeight(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassReal (relax_weight)));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SchwarzSetDofFunc
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_schwarzsetdoffunc, NALU_HYPRE_SCHWARZSETDOFFUNC)
(hypre_F90_Obj *solver,
 hypre_F90_IntArray *dof_func,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SchwarzSetDofFunc(
               hypre_F90_PassObj (NALU_HYPRE_Solver, solver),
               hypre_F90_PassIntArray (dof_func)  ));
}
#ifdef __cplusplus
}
#endif
