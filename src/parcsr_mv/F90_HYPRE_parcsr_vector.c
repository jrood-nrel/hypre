/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorcreate, NALU_HYPRE_PARVECTORCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_BigInt *global_size,
  nalu_hypre_F90_BigIntArray *partitioning,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParVectorCreate(
              nalu_hypre_F90_PassComm (comm),
              nalu_hypre_F90_PassBigInt (global_size),
              nalu_hypre_F90_PassBigIntArray (partitioning),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParVector, vector) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parmultivectorcreate, NALU_HYPRE_PARMULTIVECTORCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_BigInt *global_size,
  nalu_hypre_F90_BigIntArray *partitioning,
  nalu_hypre_F90_Int *number_vectors,
  nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr          )
{
   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParMultiVectorCreate(
              nalu_hypre_F90_PassComm (comm),
              nalu_hypre_F90_PassBigInt (global_size),
              nalu_hypre_F90_PassBigIntArray (partitioning),
              nalu_hypre_F90_PassInt (number_vectors),
              nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParVector, vector) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectordestroy, NALU_HYPRE_PARVECTORDESTROY)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorinitialize, NALU_HYPRE_PARVECTORINITIALIZE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorInitialize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorread, NALU_HYPRE_PARVECTORREAD)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Obj *vector,
  char     *file_name,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorRead(
                nalu_hypre_F90_PassComm (comm),
                (char *)    file_name,
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParVector, vector) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorprint, NALU_HYPRE_PARVECTORPRINT)
( nalu_hypre_F90_Obj *vector,
  char     *fort_file_name,
  nalu_hypre_F90_Int *fort_file_name_size,
  nalu_hypre_F90_Int *ierr       )
{
   NALU_HYPRE_Int i;
   char *c_file_name;

   c_file_name = nalu_hypre_CTAlloc(char,  *fort_file_name_size, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < *fort_file_name_size; i++)
   {
      c_file_name[i] = fort_file_name[i];
   }

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorPrint(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, vector),
                (char *)           c_file_name ) );

   nalu_hypre_TFree(c_file_name, NALU_HYPRE_MEMORY_HOST);

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorsetconstantvalue, NALU_HYPRE_PARVECTORSETCONSTANTVALUE)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Complex *value,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorSetConstantValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, vector),
                nalu_hypre_F90_PassComplex (value)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorsetrandomvalues, NALU_HYPRE_PARVECTORSETRANDOMVALUES)
( nalu_hypre_F90_Obj *vector,
  nalu_hypre_F90_Int *seed,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorSetRandomValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, vector),
                nalu_hypre_F90_PassInt (seed)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorcopy, NALU_HYPRE_PARVECTORCOPY)
( nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorCopy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorcloneshallow, NALU_HYPRE_PARVECTORCLONESHALLOW)
( nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *xclone,
  nalu_hypre_F90_Int *ierr    )
{
   *xclone = (nalu_hypre_F90_Obj)
             ( NALU_HYPRE_ParVectorCloneShallow(
                  nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorscale, NALU_HYPRE_PARVECTORSCALE)
( nalu_hypre_F90_Complex *value,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorScale(
                nalu_hypre_F90_PassComplex (value),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectoraxpy, NALU_HYPRE_PARVECTORAXPY)
( nalu_hypre_F90_Complex *value,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorAxpy(
                nalu_hypre_F90_PassComplex (value),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parvectorinnerprod, NALU_HYPRE_PARVECTORINNERPROD)
(nalu_hypre_F90_Obj *x,
 nalu_hypre_F90_Obj *y,
 nalu_hypre_F90_Complex *prod,
 nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParVectorInnerProd(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y),
                nalu_hypre_F90_PassRealRef (prod) ) );
}

#ifdef __cplusplus
}
#endif
