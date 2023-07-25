/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixcreate, NALU_HYPRE_STRUCTMATRIXCREATE)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *grid,
 nalu_hypre_F90_Obj *stencil,
 nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr)
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixCreate(
              nalu_hypre_F90_PassComm (comm),
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructStencil, stencil),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructMatrix, matrix)   );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixdestroy, NALU_HYPRE_STRUCTMATRIXDESTROY)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixDestroy(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixinitialize, NALU_HYPRE_STRUCTMATRIXINITIALIZE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixInitialize(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetvalues, NALU_HYPRE_STRUCTMATRIXSETVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *grid_index,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (grid_index),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetboxvalues, NALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetBoxValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixgetboxvalues, NALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixGetBoxValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetconstantva, NALU_HYPRE_STRUCTMATRIXSETCONSTANTVA)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixaddtovalues, NALU_HYPRE_STRUCTMATRIXADDTOVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *grid_index,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixAddToValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (grid_index),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixaddtoboxvalues, NALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *ilower,
  nalu_hypre_F90_IntArray *iupper,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixAddToBoxValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (ilower),
              nalu_hypre_F90_PassIntArray (iupper),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixaddtoconstant, NALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *num_stencil_indices,
  nalu_hypre_F90_IntArray *stencil_indices,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr              )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantValues(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassInt (num_stencil_indices),
              nalu_hypre_F90_PassIntArray (stencil_indices),
              nalu_hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixassemble, NALU_HYPRE_STRUCTMATRIXASSEMBLE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixAssemble(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetnumghost, NALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *num_ghost,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetNumGhost(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassIntArray (num_ghost) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixgetgrid, NALU_HYPRE_STRUCTMATRIXGETGRID)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *grid,
  nalu_hypre_F90_Int *ierr   )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixGetGrid(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_StructGrid, grid) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetsymmetric, NALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *symmetric,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetSymmetric(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassInt (symmetric) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetConstantEntries
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixsetconstanten, NALU_HYPRE_STRUCTMATRIXSETCONSTANTEN)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *nentries,
  nalu_hypre_F90_IntArray *entries,
  nalu_hypre_F90_Int *ierr                )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantEntries(
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassInt (nentries),
              nalu_hypre_F90_PassIntArray (entries)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixprint, NALU_HYPRE_STRUCTMATRIXPRINT)
(
   nalu_hypre_F90_Obj *matrix,
   nalu_hypre_F90_Int *all,
   nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixPrint(
              "NALU_HYPRE_StructMatrix.out",
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              nalu_hypre_F90_PassInt (all));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_structmatrixmatvec, NALU_HYPRE_STRUCTMATRIXMATVEC)
( nalu_hypre_F90_Complex *alpha,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Complex *beta,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_StructMatrixMatvec(
              nalu_hypre_F90_PassComplex (alpha),
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, x),
              nalu_hypre_F90_PassComplex (beta),
              nalu_hypre_F90_PassObj (NALU_HYPRE_StructVector, y)  );
}

#ifdef __cplusplus
}
#endif
