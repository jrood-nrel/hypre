/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphcreate, NALU_HYPRE_SSTRUCTGRAPHCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Obj *graph_ptr,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphCreate(
               nalu_hypre_F90_PassComm (comm),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructGraph, graph_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphdestroy, NALU_HYPRE_SSTRUCTGRAPHDESTROY)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetDomainGrid
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphsetdomaingrid, NALU_HYPRE_SSTRUCTGRAPHSETDOMAINGRID)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Obj *domain_grid,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetDomainGrid(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGrid, domain_grid) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetStencil
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphsetstencil, NALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Obj *stencil,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetStencil(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetFEM
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphsetfem, NALU_HYPRE_SSTRUCTGRAPHSETFEM)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetFEM(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassInt (part) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetFEMSparsity
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphsetfemsparsity, NALU_HYPRE_SSTRUCTGRAPHSETFEMSPARSITY)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_Int *nsparse,
 nalu_hypre_F90_IntArray *sparsity,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetFEMSparsity(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassInt (nsparse),
               nalu_hypre_F90_PassIntArray (sparsity) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphAddEntries-
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphaddentries, NALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *part,
 nalu_hypre_F90_IntArray *index,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *to_part,
 nalu_hypre_F90_IntArray *to_index,
 nalu_hypre_F90_Int *to_var,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphAddEntries(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassInt (part),
               nalu_hypre_F90_PassIntArray (index),
               nalu_hypre_F90_PassInt (var),
               nalu_hypre_F90_PassInt (to_part),
               nalu_hypre_F90_PassIntArray (to_index),
               nalu_hypre_F90_PassInt (to_var) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphassemble, NALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphAssemble(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructgraphsetobjecttype, NALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYPE)
(nalu_hypre_F90_Obj *graph,
 nalu_hypre_F90_Int *type,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetObjectType(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               nalu_hypre_F90_PassInt (type) ) );
}

#ifdef __cplusplus
}
#endif
