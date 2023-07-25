/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * par_vector Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_setparvectordataowner, NALU_HYPRE_SETPARVECTORDATAOWNER)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *owns_data,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorSetDataOwner(
                (nalu_hypre_ParVector *) *vector,
                nalu_hypre_F90_PassInt (owns_data) ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SetParVectorConstantValue
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_setparvectorconstantvalue, NALU_HYPRE_SETPARVECTORCONSTANTVALUE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Complex *value,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorSetConstantValues(
                (nalu_hypre_ParVector *) *vector,
                nalu_hypre_F90_PassComplex (value)   ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_setparvectorrandomvalues, NALU_HYPRE_SETPARVECTORRANDOMVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *seed,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorSetRandomValues(
                (nalu_hypre_ParVector *) *vector,
                nalu_hypre_F90_PassInt (seed)    ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_copyparvector, NALU_HYPRE_COPYPARVECTOR)
( nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorCopy(
                (nalu_hypre_ParVector *) *x,
                (nalu_hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_scaleparvector, NALU_HYPRE_SCALEPARVECTOR)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Complex *scale,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorScale(
                nalu_hypre_F90_PassComplex (scale),
                (nalu_hypre_ParVector *) *vector ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_paraxpy, NALU_HYPRE_PARAXPY)
( nalu_hypre_F90_Complex *a,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int)
           ( nalu_hypre_ParVectorAxpy(
                nalu_hypre_F90_PassComplex (a),
                (nalu_hypre_ParVector *) *x,
                (nalu_hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parinnerprod, NALU_HYPRE_PARINNERPROD)
( nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Complex *inner_prod,
  nalu_hypre_F90_Int *ierr           )
{
   *inner_prod = (nalu_hypre_F90_Complex)
                 ( nalu_hypre_ParVectorInnerProd(
                      (nalu_hypre_ParVector *) *x,
                      (nalu_hypre_ParVector *) *y  ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_VectorToParVector
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_vectortoparvector, NALU_HYPRE_VECTORTOPARVECTOR)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_BigIntArray *vec_starts,
  nalu_hypre_F90_Obj *par_vector,
  nalu_hypre_F90_Int *ierr        )
{
   *par_vector = (nalu_hypre_F90_Obj)
                 ( nalu_hypre_VectorToParVector(
                      nalu_hypre_F90_PassComm (comm),
                      (nalu_hypre_Vector *) *vector,
                      nalu_hypre_F90_PassBigIntArray (vec_starts) ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorToVectorAll
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectortovectorall, NALU_HYPRE_PARVECTORTOVECTORALL)
( nalu_hypre_F90_Obj *par_vector,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr        )
{
   *vector = (nalu_hypre_F90_Obj)(
                nalu_hypre_ParVectorToVectorAll
                ( (nalu_hypre_ParVector *) *par_vector ) );

   *ierr = 0;
}

#ifdef __cplusplus
}
#endif
