/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixcreate, NALU_HYPRE_IJMATRIXCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_BigInt *ilower,
  nalu_hypre_F90_BigInt *iupper,
  nalu_hypre_F90_BigInt *jlower,
  nalu_hypre_F90_BigInt *jupper,
  nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassBigInt (ilower),
                nalu_hypre_F90_PassBigInt (iupper),
                nalu_hypre_F90_PassBigInt (jlower),
                nalu_hypre_F90_PassBigInt (jupper),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixdestroy, NALU_HYPRE_IJMATRIXDESTROY)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixinitialize, NALU_HYPRE_IJMATRIXINITIALIZE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixInitialize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetvalues, NALU_HYPRE_IJMATRIXSETVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *nrows,
  nalu_hypre_F90_IntArray *ncols,
  nalu_hypre_F90_BigIntArray *rows,
  nalu_hypre_F90_BigIntArray *cols,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (nrows),
                nalu_hypre_F90_PassIntArray (ncols),
                nalu_hypre_F90_PassBigIntArray (rows),
                nalu_hypre_F90_PassBigIntArray (cols),
                nalu_hypre_F90_PassComplexArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetconstantvalues, NALU_HYPRE_IJMATRIXSETCONSTANTVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Complex *value,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetConstantValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassComplex (value)  ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixaddtovalues, NALU_HYPRE_IJMATRIXADDTOVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *nrows,
  nalu_hypre_F90_IntArray *ncols,
  nalu_hypre_F90_BigIntArray *rows,
  nalu_hypre_F90_BigIntArray *cols,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixAddToValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (nrows),
                nalu_hypre_F90_PassIntArray (ncols),
                nalu_hypre_F90_PassBigIntArray (rows),
                nalu_hypre_F90_PassBigIntArray (cols),
                nalu_hypre_F90_PassComplexArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixassemble, NALU_HYPRE_IJMATRIXASSEMBLE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixAssemble(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixGetRowCounts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixgetrowcounts, NALU_HYPRE_IJMATRIXGETROWCOUNTS)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *nrows,
  nalu_hypre_F90_BigIntArray *rows,
  nalu_hypre_F90_IntArray *ncols,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixGetRowCounts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (nrows),
                nalu_hypre_F90_PassBigIntArray (rows),
                nalu_hypre_F90_PassIntArray (ncols) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixgetvalues, NALU_HYPRE_IJMATRIXGETVALUES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *nrows,
  nalu_hypre_F90_IntArray *ncols,
  nalu_hypre_F90_BigIntArray *rows,
  nalu_hypre_F90_BigIntArray *cols,
  nalu_hypre_F90_ComplexArray *values,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixGetValues(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (nrows),
                nalu_hypre_F90_PassIntArray (ncols),
                nalu_hypre_F90_PassBigIntArray (rows),
                nalu_hypre_F90_PassBigIntArray (cols),
                nalu_hypre_F90_PassComplexArray (values)  ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetobjecttype, NALU_HYPRE_IJMATRIXSETOBJECTTYPE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *type,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetObjectType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (type)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixgetobjecttype, NALU_HYPRE_IJMATRIXGETOBJECTTYPE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *type,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixGetObjectType(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassIntRef (type)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixgetlocalrange, NALU_HYPRE_IJMATRIXGETLOCALRANGE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *ilower,
  nalu_hypre_F90_BigInt *iupper,
  nalu_hypre_F90_BigInt *jlower,
  nalu_hypre_F90_BigInt *jupper,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixGetLocalRange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassBigIntRef (ilower),
                nalu_hypre_F90_PassBigIntRef (iupper),
                nalu_hypre_F90_PassBigIntRef (jlower),
                nalu_hypre_F90_PassBigIntRef (jupper) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixgetobject, NALU_HYPRE_IJMATRIXGETOBJECT)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *object,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixGetObject(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetrowsizes, NALU_HYPRE_IJMATRIXSETROWSIZES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *sizes,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetRowSizes(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassIntArray (sizes)   ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetdiagoffdsizes, NALU_HYPRE_IJMATRIXSETDIAGOFFDSIZES)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_IntArray *diag_sizes,
  nalu_hypre_F90_IntArray *offd_sizes,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetDiagOffdSizes(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassIntArray (diag_sizes),
                nalu_hypre_F90_PassIntArray (offd_sizes) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixsetmaxoffprocelmt, NALU_HYPRE_IJMATRIXSETMAXOFFPROCELMT)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *max_off_proc_elmts,
  nalu_hypre_F90_Int *ierr        )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixSetMaxOffProcElmts(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                nalu_hypre_F90_PassInt (max_off_proc_elmts) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixread, NALU_HYPRE_IJMATRIXREAD)
( char     *filename,
  nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Int *object_type,
  nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixRead(
                (char *)            filename,
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassInt (object_type),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_IJMatrix, matrix)    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_ijmatrixprint, NALU_HYPRE_IJMATRIXPRINT)
( nalu_hypre_F90_Obj *matrix,
  char     *filename,
  nalu_hypre_F90_Int *ierr      )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_IJMatrixPrint(
                nalu_hypre_F90_PassObj (NALU_HYPRE_IJMatrix, matrix),
                (char *)          filename ) );
}

#ifdef __cplusplus
}
#endif
