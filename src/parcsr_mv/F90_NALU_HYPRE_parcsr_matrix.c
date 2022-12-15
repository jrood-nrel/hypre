/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRMatrix Fortran interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixcreate, NALU_HYPRE_PARCSRMATRIXCREATE)
( nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_BigInt *global_num_rows,
  nalu_hypre_F90_BigInt *global_num_cols,
  nalu_hypre_F90_BigIntArray *row_starts,
  nalu_hypre_F90_BigIntArray *col_starts,
  nalu_hypre_F90_Int *num_cols_offd,
  nalu_hypre_F90_Int *num_nonzeros_diag,
  nalu_hypre_F90_Int *num_nonzeros_offd,
  nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr               )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixCreate(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassBigInt (global_num_rows),
                nalu_hypre_F90_PassBigInt (global_num_cols),
                nalu_hypre_F90_PassBigIntArray (row_starts),
                nalu_hypre_F90_PassBigIntArray (col_starts),
                nalu_hypre_F90_PassInt (num_cols_offd),
                nalu_hypre_F90_PassInt (num_nonzeros_diag),
                nalu_hypre_F90_PassInt (num_nonzeros_offd),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixdestroy, NALU_HYPRE_PARCSRMATRIXDESTROY)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixDestroy(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixinitialize, NALU_HYPRE_PARCSRMATRIXINITIALIZE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixInitialize(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixread, NALU_HYPRE_PARCSRMATRIXREAD)
( nalu_hypre_F90_Comm *comm,
  char     *file_name,
  nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Int *ierr       )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixRead(
                nalu_hypre_F90_PassComm (comm),
                (char *)    file_name,
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixprint, NALU_HYPRE_PARCSRMATRIXPRINT)
( nalu_hypre_F90_Obj *matrix,
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
           ( NALU_HYPRE_ParCSRMatrixPrint(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                (char *)              c_file_name ) );

   nalu_hypre_TFree(c_file_name, NALU_HYPRE_MEMORY_HOST);

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetcomm, NALU_HYPRE_PARCSRMATRIXGETCOMM)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Comm *comm,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetComm(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                (MPI_Comm *)          comm    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetdims, NALU_HYPRE_PARCSRMATRIXGETDIMS)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *M,
  nalu_hypre_F90_BigInt *N,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetDims(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                nalu_hypre_F90_PassBigIntRef (M),
                nalu_hypre_F90_PassBigIntRef (N)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetrowpartiti, NALU_HYPRE_PARCSRMATRIXGETROWPARTITI)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *row_partitioning_ptr,
  nalu_hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *row_partitioning;

   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetRowPartitioning(
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
              (NALU_HYPRE_BigInt **)    &row_partitioning  );

   *row_partitioning_ptr = (nalu_hypre_F90_Obj) row_partitioning;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetcolpartiti, NALU_HYPRE_PARCSRMATRIXGETCOLPARTITI)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_Obj *col_partitioning_ptr,
  nalu_hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_partitioning;

   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetColPartitioning(
              nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
              (NALU_HYPRE_BigInt **)    &col_partitioning  );

   *col_partitioning_ptr = (nalu_hypre_F90_Obj) col_partitioning;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetlocalrange, NALU_HYPRE_PARCSRMATRIXGETLOCALRANGE)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *row_start,
  nalu_hypre_F90_BigInt *row_end,
  nalu_hypre_F90_BigInt *col_start,
  nalu_hypre_F90_BigInt *col_end,
  nalu_hypre_F90_Int *ierr    )
{
   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetLocalRange(
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                nalu_hypre_F90_PassBigIntRef (row_start),
                nalu_hypre_F90_PassBigIntRef (row_end),
                nalu_hypre_F90_PassBigIntRef (col_start),
                nalu_hypre_F90_PassBigIntRef (col_end)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixgetrow, NALU_HYPRE_PARCSRMATRIXGETROW)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *row,
  nalu_hypre_F90_Int *size,
  nalu_hypre_F90_Obj *col_ind_ptr,
  nalu_hypre_F90_Obj *values_ptr,
  nalu_hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_ind;
   NALU_HYPRE_Complex    *values;

   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetRow(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_ParCSRMatrix, matrix),
              nalu_hypre_F90_PassBigInt      (row),
              nalu_hypre_F90_PassIntRef (size),
              (NALU_HYPRE_BigInt **)         &col_ind,
              (NALU_HYPRE_Complex **)            &values );

   *col_ind_ptr = (nalu_hypre_F90_Obj) col_ind;
   *values_ptr  = (nalu_hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixrestorerow, NALU_HYPRE_PARCSRMATRIXRESTOREROW)
( nalu_hypre_F90_Obj *matrix,
  nalu_hypre_F90_BigInt *row,
  nalu_hypre_F90_Int *size,
  nalu_hypre_F90_Obj *col_ind_ptr,
  nalu_hypre_F90_Obj *values_ptr,
  nalu_hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_ind;
   NALU_HYPRE_Complex    *values;

   *ierr = (nalu_hypre_F90_Int) NALU_HYPRE_ParCSRMatrixRestoreRow(
              nalu_hypre_F90_PassObj      (NALU_HYPRE_ParCSRMatrix, matrix),
              nalu_hypre_F90_PassBigInt      (row),
              nalu_hypre_F90_PassIntRef (size),
              (NALU_HYPRE_BigInt **)         &col_ind,
              (NALU_HYPRE_Complex **)            &values );

   *col_ind_ptr = (nalu_hypre_F90_Obj) col_ind;
   *values_ptr  = (nalu_hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_csrmatrixtoparcsrmatrix, NALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *A_CSR,
 nalu_hypre_F90_BigIntArray *row_partitioning,
 nalu_hypre_F90_BigIntArray *col_partitioning,
 nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr   )
{

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_CSRMatrixToParCSRMatrix(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, A_CSR),
                nalu_hypre_F90_PassBigIntArray (row_partitioning),
                nalu_hypre_F90_PassBigIntArray (col_partitioning),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_csrmatrixtoparcsrmatrix_withnewpartitioning,
                NALU_HYPRE_CSRMATRIXTOPARCSRMATRIX_WITHNEWPARTITIONING)
(nalu_hypre_F90_Comm *comm,
 nalu_hypre_F90_Obj *A_CSR,
 nalu_hypre_F90_Obj *matrix,
 nalu_hypre_F90_Int *ierr   )
{

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
                nalu_hypre_F90_PassComm (comm),
                nalu_hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, A_CSR),
                nalu_hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixmatvec, NALU_HYPRE_PARCSRMATRIXMATVEC)
( nalu_hypre_F90_Complex *alpha,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Complex *beta,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr   )
{

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixMatvec(
                nalu_hypre_F90_PassComplex (alpha),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassComplex (beta),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

void
nalu_hypre_F90_IFACE(nalu_hypre_parcsrmatrixmatvect, NALU_HYPRE_PARCSRMATRIXMATVECT)
( nalu_hypre_F90_Complex *alpha,
  nalu_hypre_F90_Obj *A,
  nalu_hypre_F90_Obj *x,
  nalu_hypre_F90_Complex *beta,
  nalu_hypre_F90_Obj *y,
  nalu_hypre_F90_Int *ierr    )
{

   *ierr = (nalu_hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixMatvecT(
                nalu_hypre_F90_PassComplex (alpha),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                nalu_hypre_F90_PassComplex (beta),
                nalu_hypre_F90_PassObj (NALU_HYPRE_ParVector, y)      ) );
}

#ifdef __cplusplus
}
#endif
