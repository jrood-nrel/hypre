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

#include "_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphcreate, NALU_HYPRE_SSTRUCTGRAPHCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *grid,
 hypre_F90_Obj *graph_ptr,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphCreate(
               hypre_F90_PassComm (comm),
               hypre_F90_PassObj (NALU_HYPRE_SStructGrid, grid),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructGraph, graph_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphdestroy, NALU_HYPRE_SSTRUCTGRAPHDESTROY)
(hypre_F90_Obj *graph,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetDomainGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetdomaingrid, NALU_HYPRE_SSTRUCTGRAPHSETDOMAINGRID)
(hypre_F90_Obj *graph,
 hypre_F90_Obj *domain_grid,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetDomainGrid(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassObj (NALU_HYPRE_SStructGrid, domain_grid) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetStencil
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetstencil, NALU_HYPRE_SSTRUCTGRAPHSETSTENCIL)
(hypre_F90_Obj *graph,
 hypre_F90_Int *part,
 hypre_F90_Int *var,
 hypre_F90_Obj *stencil,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetStencil(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassInt (part),
               hypre_F90_PassInt (var),
               hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetFEM
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetfem, NALU_HYPRE_SSTRUCTGRAPHSETFEM)
(hypre_F90_Obj *graph,
 hypre_F90_Int *part,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetFEM(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassInt (part) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetFEMSparsity
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetfemsparsity, NALU_HYPRE_SSTRUCTGRAPHSETFEMSPARSITY)
(hypre_F90_Obj *graph,
 hypre_F90_Int *part,
 hypre_F90_Int *nsparse,
 hypre_F90_IntArray *sparsity,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetFEMSparsity(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassInt (part),
               hypre_F90_PassInt (nsparse),
               hypre_F90_PassIntArray (sparsity) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphAddEntries-
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphaddentries, NALU_HYPRE_SSTRUCTGRAPHADDENTRIES)
(hypre_F90_Obj *graph,
 hypre_F90_Int *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int *var,
 hypre_F90_Int *to_part,
 hypre_F90_IntArray *to_index,
 hypre_F90_Int *to_var,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphAddEntries(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassInt (part),
               hypre_F90_PassIntArray (index),
               hypre_F90_PassInt (var),
               hypre_F90_PassInt (to_part),
               hypre_F90_PassIntArray (to_index),
               hypre_F90_PassInt (to_var) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphassemble, NALU_HYPRE_SSTRUCTGRAPHASSEMBLE)
(hypre_F90_Obj *graph,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphAssemble(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructGraphSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgraphsetobjecttype, NALU_HYPRE_SSTRUCTGRAPHSETOBJECTTYPE)
(hypre_F90_Obj *graph,
 hypre_F90_Int *type,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructGraphSetObjectType(
               hypre_F90_PassObj (NALU_HYPRE_SStructGraph, graph),
               hypre_F90_PassInt (type) ) );
}

#ifdef __cplusplus
}
#endif
