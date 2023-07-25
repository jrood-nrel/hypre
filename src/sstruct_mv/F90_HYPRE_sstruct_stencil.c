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

#include "_nalu_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructstencilcreate, NALU_HYPRE_SSTRUCTSTENCILCREATE)
(nalu_hypre_F90_Int *ndim,
 nalu_hypre_F90_Int *size,
 nalu_hypre_F90_Obj *stencil_ptr,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructStencilCreate(
               nalu_hypre_F90_PassInt (ndim),
               nalu_hypre_F90_PassInt (size),
               nalu_hypre_F90_PassObjRef (NALU_HYPRE_SStructStencil, stencil_ptr) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructstencildestroy, NALU_HYPRE_SSTRUCTSTENCILDESTROY)
(nalu_hypre_F90_Obj *stencil,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructStencilDestroy(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructStencilSetEntry
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_sstructstencilsetentry, NALU_HYPRE_SSTRUCTSTENCILSETENTRY)
(nalu_hypre_F90_Obj *stencil,
 nalu_hypre_F90_Int *entry,
 nalu_hypre_F90_IntArray *offset,
 nalu_hypre_F90_Int *var,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int)
           (NALU_HYPRE_SStructStencilSetEntry(
               nalu_hypre_F90_PassObj (NALU_HYPRE_SStructStencil, stencil),
               nalu_hypre_F90_PassInt (entry),
               nalu_hypre_F90_PassIntArray (offset),
               nalu_hypre_F90_PassInt (var) ) );
}

#ifdef __cplusplus
}
#endif
