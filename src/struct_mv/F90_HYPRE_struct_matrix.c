/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixcreate, NALU_HYPRE_STRUCTMATRIXCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *grid,
 hypre_F90_Obj *stencil,
 hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixCreate(
              hypre_F90_PassComm (comm),
              hypre_F90_PassObj (NALU_HYPRE_StructGrid, grid),
              hypre_F90_PassObj (NALU_HYPRE_StructStencil, stencil),
              hypre_F90_PassObjRef (NALU_HYPRE_StructMatrix, matrix)   );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixdestroy, NALU_HYPRE_STRUCTMATRIXDESTROY)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixDestroy(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixinitialize, NALU_HYPRE_STRUCTMATRIXINITIALIZE)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixInitialize(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetvalues, NALU_HYPRE_STRUCTMATRIXSETVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (grid_index),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetboxvalues, NALU_HYPRE_STRUCTMATRIXSETBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetBoxValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetboxvalues, NALU_HYPRE_STRUCTMATRIXGETBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixGetBoxValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetconstantva, NALU_HYPRE_STRUCTMATRIXSETCONSTANTVA)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtovalues, NALU_HYPRE_STRUCTMATRIXADDTOVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixAddToValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (grid_index),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtoboxvalues, NALU_HYPRE_STRUCTMATRIXADDTOBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixAddToBoxValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtoconstant, NALU_HYPRE_STRUCTMATRIXADDTOCONSTANT)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantValues(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixassemble, NALU_HYPRE_STRUCTMATRIXASSEMBLE)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixAssemble(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetnumghost, NALU_HYPRE_STRUCTMATRIXSETNUMGHOST)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *num_ghost,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetNumGhost(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (num_ghost) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetgrid, NALU_HYPRE_STRUCTMATRIXGETGRID)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *grid,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixGetGrid(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassObjRef (NALU_HYPRE_StructGrid, grid) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetsymmetric, NALU_HYPRE_STRUCTMATRIXSETSYMMETRIC)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *symmetric,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetSymmetric(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (symmetric) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixSetConstantEntries
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetconstanten, NALU_HYPRE_STRUCTMATRIXSETCONSTANTEN)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *nentries,
  hypre_F90_IntArray *entries,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixSetConstantEntries(
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (nentries),
              hypre_F90_PassIntArray (entries)           );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixprint, NALU_HYPRE_STRUCTMATRIXPRINT)
(
   hypre_F90_Obj *matrix,
   hypre_F90_Int *all,
   hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixPrint(
              "NALU_HYPRE_StructMatrix.out",
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (all));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixmatvec, NALU_HYPRE_STRUCTMATRIXMATVEC)
( hypre_F90_Complex *alpha,
  hypre_F90_Obj *A,
  hypre_F90_Obj *x,
  hypre_F90_Complex *beta,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int) NALU_HYPRE_StructMatrixMatvec(
              hypre_F90_PassComplex (alpha),
              hypre_F90_PassObj (NALU_HYPRE_StructMatrix, A),
              hypre_F90_PassObj (NALU_HYPRE_StructVector, x),
              hypre_F90_PassComplex (beta),
              hypre_F90_PassObj (NALU_HYPRE_StructVector, y)  );
}

#ifdef __cplusplus
}
#endif
