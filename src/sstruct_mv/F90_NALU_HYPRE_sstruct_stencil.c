/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructStencil interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilcreate, NALU_HYPRE_SSTRUCTSTENCILCREATE)
(hypre_F90_Int *ndim,
 hypre_F90_Int *size,
 hypre_F90_Obj *stencil_ptr,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructStencilCreate(
               hypre_F90_PassInt (ndim),
               hypre_F90_PassInt (size),
               hypre_F90_PassObjRef (NALU_HYPRE_SStructStencil, stencil_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencildestroy, NALU_HYPRE_SSTRUCTSTENCILDESTROY)
(hypre_F90_Obj *stencil,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructStencilDestroy(
               hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructstencilsetentry, NALU_HYPRE_SSTRUCTSTENCILSETENTRY)
(hypre_F90_Obj *stencil,
 hypre_F90_Int *entry,
 hypre_F90_IntArray *offset,
 hypre_F90_Int *var,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           (NALU_HYPRE_SStructStencilSetEntry(
               hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil),
               hypre_F90_PassInt (entry),
               hypre_F90_PassIntArray (offset),
               hypre_F90_PassInt (var) ) );
}

#ifdef __cplusplus
}
#endif
