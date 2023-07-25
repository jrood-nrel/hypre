/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructStencil interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structstencilcreate, NALU_HYPRE_STRUCTSTENCILCREATE)
( nalu_hypre_F90_Int *dim,
  nalu_hypre_F90_Int *size,
  nalu_hypre_F90_Obj *stencil,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructStencilCreate(
              nalu_hypre_F90_PassInt (dim),
              nalu_hypre_F90_PassInt (size),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructStencil, stencil) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilSetElement
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structstencilsetelement, NALU_HYPRE_STRUCTSTENCILSETELEMENT)
( nalu_hypre_F90_Obj *stencil,
  nalu_hypre_F90_Int *element_index,
  nalu_hypre_F90_IntArray *offset,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructStencilSetElement(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructStencil, stencil),
              nalu_hypre_F90_PassInt (element_index),
              nalu_hypre_F90_PassIntArray (offset)       );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructStencilDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structstencildestroy, NALU_HYPRE_STRUCTSTENCILDESTROY)
( nalu_hypre_F90_Obj *stencil,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructStencilDestroy(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructStencil, stencil) );
}

#ifdef __cplusplus
}
#endif
