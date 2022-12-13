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

#include "_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixcreate, NALU_HYPRE_PARCSRMATRIXCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_BigInt *global_num_rows,
  hypre_F90_BigInt *global_num_cols,
  hypre_F90_BigIntArray *row_starts,
  hypre_F90_BigIntArray *col_starts,
  hypre_F90_Int *num_cols_offd,
  hypre_F90_Int *num_nonzeros_diag,
  hypre_F90_Int *num_nonzeros_offd,
  hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr               )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassBigInt (global_num_rows),
                hypre_F90_PassBigInt (global_num_cols),
                hypre_F90_PassBigIntArray (row_starts),
                hypre_F90_PassBigIntArray (col_starts),
                hypre_F90_PassInt (num_cols_offd),
                hypre_F90_PassInt (num_nonzeros_diag),
                hypre_F90_PassInt (num_nonzeros_offd),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix)  ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixdestroy, NALU_HYPRE_PARCSRMATRIXDESTROY)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixDestroy(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixinitialize, NALU_HYPRE_PARCSRMATRIXINITIALIZE)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixInitialize(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixread, NALU_HYPRE_PARCSRMATRIXREAD)
( hypre_F90_Comm *comm,
  char     *file_name,
  hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixRead(
                hypre_F90_PassComm (comm),
                (char *)    file_name,
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixprint, NALU_HYPRE_PARCSRMATRIXPRINT)
( hypre_F90_Obj *matrix,
  char     *fort_file_name,
  hypre_F90_Int *fort_file_name_size,
  hypre_F90_Int *ierr       )
{
   NALU_HYPRE_Int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char,  *fort_file_name_size, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < *fort_file_name_size; i++)
   {
      c_file_name[i] = fort_file_name[i];
   }

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixPrint(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                (char *)              c_file_name ) );

   hypre_TFree(c_file_name, NALU_HYPRE_MEMORY_HOST);

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetcomm, NALU_HYPRE_PARCSRMATRIXGETCOMM)
( hypre_F90_Obj *matrix,
  hypre_F90_Comm *comm,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetComm(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                (MPI_Comm *)          comm    ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetdims, NALU_HYPRE_PARCSRMATRIXGETDIMS)
( hypre_F90_Obj *matrix,
  hypre_F90_BigInt *M,
  hypre_F90_BigInt *N,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetDims(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                hypre_F90_PassBigIntRef (M),
                hypre_F90_PassBigIntRef (N)       ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetrowpartiti, NALU_HYPRE_PARCSRMATRIXGETROWPARTITI)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *row_partitioning_ptr,
  hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *row_partitioning;

   *ierr = (hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetRowPartitioning(
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
              (NALU_HYPRE_BigInt **)    &row_partitioning  );

   *row_partitioning_ptr = (hypre_F90_Obj) row_partitioning;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetcolpartiti, NALU_HYPRE_PARCSRMATRIXGETCOLPARTITI)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *col_partitioning_ptr,
  hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_partitioning;

   *ierr = (hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetColPartitioning(
              hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
              (NALU_HYPRE_BigInt **)    &col_partitioning  );

   *col_partitioning_ptr = (hypre_F90_Obj) col_partitioning;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetlocalrange, NALU_HYPRE_PARCSRMATRIXGETLOCALRANGE)
( hypre_F90_Obj *matrix,
  hypre_F90_BigInt *row_start,
  hypre_F90_BigInt *row_end,
  hypre_F90_BigInt *col_start,
  hypre_F90_BigInt *col_end,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixGetLocalRange(
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, matrix),
                hypre_F90_PassBigIntRef (row_start),
                hypre_F90_PassBigIntRef (row_end),
                hypre_F90_PassBigIntRef (col_start),
                hypre_F90_PassBigIntRef (col_end)) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixgetrow, NALU_HYPRE_PARCSRMATRIXGETROW)
( hypre_F90_Obj *matrix,
  hypre_F90_BigInt *row,
  hypre_F90_Int *size,
  hypre_F90_Obj *col_ind_ptr,
  hypre_F90_Obj *values_ptr,
  hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_ind;
   NALU_HYPRE_Complex    *values;

   *ierr = (hypre_F90_Int) NALU_HYPRE_ParCSRMatrixGetRow(
              hypre_F90_PassObj      (NALU_HYPRE_ParCSRMatrix, matrix),
              hypre_F90_PassBigInt      (row),
              hypre_F90_PassIntRef (size),
              (NALU_HYPRE_BigInt **)         &col_ind,
              (NALU_HYPRE_Complex **)            &values );

   *col_ind_ptr = (hypre_F90_Obj) col_ind;
   *values_ptr  = (hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrestorerow, NALU_HYPRE_PARCSRMATRIXRESTOREROW)
( hypre_F90_Obj *matrix,
  hypre_F90_BigInt *row,
  hypre_F90_Int *size,
  hypre_F90_Obj *col_ind_ptr,
  hypre_F90_Obj *values_ptr,
  hypre_F90_Int *ierr )
{
   NALU_HYPRE_Int *col_ind;
   NALU_HYPRE_Complex    *values;

   *ierr = (hypre_F90_Int) NALU_HYPRE_ParCSRMatrixRestoreRow(
              hypre_F90_PassObj      (NALU_HYPRE_ParCSRMatrix, matrix),
              hypre_F90_PassBigInt      (row),
              hypre_F90_PassIntRef (size),
              (NALU_HYPRE_BigInt **)         &col_ind,
              (NALU_HYPRE_Complex **)            &values );

   *col_ind_ptr = (hypre_F90_Obj) col_ind;
   *values_ptr  = (hypre_F90_Obj) values;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix, NALU_HYPRE_CSRMATRIXTOPARCSRMATRIX)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *A_CSR,
 hypre_F90_BigIntArray *row_partitioning,
 hypre_F90_BigIntArray *col_partitioning,
 hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_CSRMatrixToParCSRMatrix(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, A_CSR),
                hypre_F90_PassBigIntArray (row_partitioning),
                hypre_F90_PassBigIntArray (col_partitioning),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_csrmatrixtoparcsrmatrix_withnewpartitioning,
                NALU_HYPRE_CSRMATRIXTOPARCSRMATRIX_WITHNEWPARTITIONING)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *A_CSR,
 hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObj (NALU_HYPRE_CSRMatrix, A_CSR),
                hypre_F90_PassObjRef (NALU_HYPRE_ParCSRMatrix, matrix) ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvec, NALU_HYPRE_PARCSRMATRIXMATVEC)
( hypre_F90_Complex *alpha,
  hypre_F90_Obj *A,
  hypre_F90_Obj *x,
  hypre_F90_Complex *beta,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr   )
{

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixMatvec(
                hypre_F90_PassComplex (alpha),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                hypre_F90_PassComplex (beta),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, y)      ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixmatvect, NALU_HYPRE_PARCSRMATRIXMATVECT)
( hypre_F90_Complex *alpha,
  hypre_F90_Obj *A,
  hypre_F90_Obj *x,
  hypre_F90_Complex *beta,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr    )
{

   *ierr = (hypre_F90_Int)
           ( NALU_HYPRE_ParCSRMatrixMatvecT(
                hypre_F90_PassComplex (alpha),
                hypre_F90_PassObj (NALU_HYPRE_ParCSRMatrix, A),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, x),
                hypre_F90_PassComplex (beta),
                hypre_F90_PassObj (NALU_HYPRE_ParVector, y)      ) );
}

#ifdef __cplusplus
}
#endif
