/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_Euclid.h"
/* #include "getRow_dh.h" */
/* #include "Mat_dh.h" */
/* #include "Euclid_dh.h" */
/* #include "Mem_dh.h" */

/*-------------------------------------------------------------------
 *  NALU_HYPRE
 *-------------------------------------------------------------------*/
#if defined(NALU_HYPRE_GET_ROW)

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (NALU_HYPRE_GET_ROW)"
void EuclidGetRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  NALU_HYPRE_Int ierr;
  NALU_HYPRE_ParCSRMatrix mat = (NALU_HYPRE_ParCSRMatrix) A;
  ierr = NALU_HYPRE_ParCSRMatrixGetRow(mat, row, len, ind, val); 
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "NALU_HYPRE_ParCSRMatrixRestoreRow(row= %i) returned %i", row+1, ierr);
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (NALU_HYPRE_GET_ROW)"
void EuclidRestoreRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  NALU_HYPRE_Int ierr;
  NALU_HYPRE_ParCSRMatrix mat = (NALU_HYPRE_ParCSRMatrix) A;
  ierr = NALU_HYPRE_ParCSRMatrixRestoreRow(mat, row, len, ind, val); 
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "NALU_HYPRE_ParCSRMatrixRestoreRow(row= %i) returned %i", row+1, ierr);
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidGetDimensions (NALU_HYPRE)"
void EuclidGetDimensions(void *A, NALU_HYPRE_Int *beg_row, NALU_HYPRE_Int *rowsLocal, NALU_HYPRE_Int *rowsGlobal)
{
  START_FUNC_DH
  NALU_HYPRE_Int ierr, m, n;
  NALU_HYPRE_Int row_start, row_end, col_start, col_end;
  NALU_HYPRE_ParCSRMatrix mat = (NALU_HYPRE_ParCSRMatrix) A;

  ierr = NALU_HYPRE_ParCSRMatrixGetDims(mat, &m, &n);
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "NALU_HYPRE_ParCSRMatrixGetDims() returned %i", ierr);
    SET_V_ERROR(msgBuf_dh);
  }

  ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange(mat, &row_start, &row_end, 
                                       &col_start, &col_end);
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "NALU_HYPRE_ParCSRMatrixGetLocalRange() returned %i", ierr);
    SET_V_ERROR(msgBuf_dh);
  }

/* nalu_hypre_fprintf(stderr, "\n##### [%i] EuclidGetDimensions: m= %i  n= %i  beg_row= %i row_end= %i  col_start= %i  col_end= %i\n",
                             myid_dh, m,n,row_start,row_end,col_start,col_end);
*/

  *beg_row = row_start;
  *rowsLocal = (row_end - row_start + 1);
  *rowsGlobal = n;
  END_FUNC_DH
}

/*
#undef __FUNC__
#define __FUNC__ "EuclidReadLocalNz (NALU_HYPRE)"
NALU_HYPRE_Int EuclidReadLocalNz(void *A)
{
  START_FUNC_DH
  if (ignoreMe) SET_V_ERROR("not implemented");
  return(0);
  END_FUNC_DH
}
*/


/*-------------------------------------------------------------------
 *  PETSc
 *-------------------------------------------------------------------*/
#elif defined(PETSC_GET_ROW)

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (PETSC_GET_ROW)"
void EuclidGetRow(void *Ain, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  Mat A = Ain;
  NALU_HYPRE_Int ierr;

  ierr = MatGetRow(A, row, len, ind, val);
  if (ierr) { 
    nalu_hypre_sprintf(msgBuf_dh, "PETSc's MatGetRow bombed for row= %i", row);
    SET_V_ERROR(msgBuf_dh);
  }

  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (PETSC_GET_ROW)"
void EuclidRestoreRow(void *Ain, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  Mat A = (Mat)Ain;
  NALU_HYPRE_Int ierr;

  ierr = MatRestoreRow(A, row, len, ind, val);
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "PETSc's MatRestoreRow bombed for row= %i", row);
    SET_V_ERROR(msgBuf_dh);
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidGetDimensions (PETSC)"
void EuclidGetDimensions(void *Ain, NALU_HYPRE_Int *beg_row, NALU_HYPRE_Int *rowsLocal, NALU_HYPRE_Int *rowsGlobal)
{
  START_FUNC_DH
  Mat A = (Mat)Ain;
  NALU_HYPRE_Int first, ierr, last;
  NALU_HYPRE_Int rows, cols;

  ierr = MatGetOwnershipRange(A, &first, &last);
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "PETSc's MatGetOwnershipRange failed");
    SET_V_ERROR(msgBuf_dh);
  }
  ierr = MatGetSize(A, &rows, &cols); 
  if (ierr) {
    nalu_hypre_sprintf(msgBuf_dh, "PETSc'MatGetSize failed");
    SET_V_ERROR(msgBuf_dh);
  }
  if (rows != cols) {
    nalu_hypre_sprintf(msgBuf_dh, "matrix is not square; global dimensions: rows = %i, cols = %i", rows, cols);
    SET_V_ERROR(msgBuf_dh);
  }

  *beg_row = first;
  *rowsLocal = last - first;
  *rowsGlobal = rows;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidReadLocalNz (PETSC)"
NALU_HYPRE_Int EuclidReadLocalNz(void *Ain)
{
  START_FUNC_DH
  Mat A = (Mat)Ain;
  NALU_HYPRE_Int m, n, ierr;

  ierr = MatGetLocalSize(Ain, &m, &n); 
  if (ierr) SET_ERROR(-1, "PETSc::MatGetLocalSize failed!\n");
  END_FUNC_VAL(m)
}



/*-------------------------------------------------------------------
 *  Euclid  
 *-------------------------------------------------------------------*/
#elif defined(EUCLID_GET_ROW)


#undef __FUNC__
#define __FUNC__ "EuclidGetRow (EUCLID_GET_ROW)"
void EuclidGetRow(void *A, NALU_HYPRE_Int globalRow, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  Mat_dh B = (Mat_dh)A;  
  NALU_HYPRE_Int row = globalRow - B->beg_row;
  if (row > B->m) {
    nalu_hypre_sprintf(msgBuf_dh, "requested globalRow= %i, which is local row= %i, but only have %i rows!",
                                globalRow, row, B->m);
    SET_V_ERROR(msgBuf_dh);
  }
  *len = B->rp[row+1] - B->rp[row];
  if (ind != NULL) *ind = B->cval + B->rp[row]; 
  if (val != NULL) *val = B->aval + B->rp[row]; 
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (EUCLID_GET_ROW)"
void EuclidRestoreRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidGetDimensions (EUCLID)"
void EuclidGetDimensions(void *A, NALU_HYPRE_Int *beg_row, NALU_HYPRE_Int *rowsLocal, NALU_HYPRE_Int *rowsGlobal)
{
  START_FUNC_DH
  Mat_dh B = (Mat_dh)A;  
  *beg_row = B->beg_row;
  *rowsLocal = B->m;
  *rowsGlobal = B->n;
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidReadLocalNz (EUCLID)"
NALU_HYPRE_Int EuclidReadLocalNz(void *A)
{
  START_FUNC_DH
  Mat_dh B = (Mat_dh)A;  
  NALU_HYPRE_Int nz = B->rp[B->m];
  END_FUNC_VAL(nz)
}

/*-------------------------------------------------------------------
 *  Default
 *-------------------------------------------------------------------*/
#else

#undef __FUNC__
#define __FUNC__ "EuclidGetRow (ERROR)"
void EuclidGetRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidRestoreRow (ERROR)"
void EuclidRestoreRow(void *A, NALU_HYPRE_Int row, NALU_HYPRE_Int *len, NALU_HYPRE_Int **ind, NALU_HYPRE_Real **val) 
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidGetDimensions (ERROR)"
void EuclidGetDimensions(void *A, NALU_HYPRE_Int *beg_row, NALU_HYPRE_Int *rowsLocal, NALU_HYPRE_Int *rowsGlobal)
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "EuclidReadLocalNz (ERROR)"
NALU_HYPRE_Int EuclidReadLocalNz(void *A)
{
  START_FUNC_DH
  SET_ERROR(EUCLID_ERROR, "Oops; missing XXX_GET_ROW definition!");
  END_FUNC_DH
}



#endif

/*-------------------------------------------------------------------
 *  end of GET_ROW definitions
 *-------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "PrintMatUsingGetRow"
void PrintMatUsingGetRow(void* A, NALU_HYPRE_Int beg_row, NALU_HYPRE_Int m,
                          NALU_HYPRE_Int *n2o_row, NALU_HYPRE_Int *n2o_col, char *filename)
{
  START_FUNC_DH
  FILE *fp;
  NALU_HYPRE_Int *o2n_col = NULL, pe, i, j, *cval, len;
  NALU_HYPRE_Int newCol, newRow;
  NALU_HYPRE_Real *aval;

  /* form inverse column permutation */
  if (n2o_col != NULL) {
    o2n_col = (NALU_HYPRE_Int*)MALLOC_DH(m*sizeof(NALU_HYPRE_Int)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;
  }

  for (pe=0; pe<np_dh; ++pe) {

    nalu_hypre_MPI_Barrier(comm_dh);

    if (myid_dh == pe) {
      if (pe == 0) {
        fp=fopen(filename, "w");
      } else {
        fp=fopen(filename, "a");
      }
      if (fp == NULL) {
        nalu_hypre_sprintf(msgBuf_dh, "can't open %s for writing\n", filename);
        SET_V_ERROR(msgBuf_dh);
      }

      for (i=0; i<m; ++i) {

        if (n2o_row == NULL) {
          EuclidGetRow(A, i+beg_row, &len, &cval, &aval); CHECK_V_ERROR;
          for (j=0; j<len; ++j) {
            nalu_hypre_fprintf(fp, "%i %i %g\n", i+1, cval[j], aval[j]);
          }
          EuclidRestoreRow(A, i, &len, &cval, &aval); CHECK_V_ERROR;
        } else {
          newRow = n2o_row[i] + beg_row;
          EuclidGetRow(A, newRow, &len, &cval, &aval); CHECK_V_ERROR;
          for (j=0; j<len; ++j) {
            newCol = o2n_col[cval[j]-beg_row] + beg_row; 
            nalu_hypre_fprintf(fp, "%i %i %g\n", i+1, newCol, aval[j]);
          }
          EuclidRestoreRow(A, i, &len, &cval, &aval); CHECK_V_ERROR;
        }
      }
      fclose(fp);
    }
  }

  if (n2o_col != NULL) {
    FREE_DH(o2n_col); CHECK_V_ERROR;
  }
  END_FUNC_DH
}

/*------------------------------------------------------------------------
 *  functions for setting matrices
 *------------------------------------------------------------------------*/

#ifdef NALU_HYPRE_MODE
#undef __FUNC__
#define __FUNC__ "Euclid_dhInputHypreMat"
void Euclid_dhInputHypreMat(Euclid_dh ctx, NALU_HYPRE_ParCSRMatrix A)
{
  START_FUNC_DH
  NALU_HYPRE_Int M, N;
  NALU_HYPRE_Int beg_row, end_row, junk;

  /* get dimension and ownership information */
  NALU_HYPRE_ParCSRMatrixGetDims(A, &M , &N);
  if (M != N) {
    nalu_hypre_sprintf(msgBuf_dh, "Global matrix is not square: M= %i, N= %i", M, N);
    SET_V_ERROR(msgBuf_dh);
  }
  NALU_HYPRE_ParCSRMatrixGetLocalRange(A, &beg_row, &end_row, &junk, &junk);

  ctx->m = end_row - beg_row + 1;
  ctx->n = M;
  ctx->A = (void*)A;

  END_FUNC_DH
}
#endif
