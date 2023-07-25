/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Common.h"
#include "Mem.h"

#ifndef _MATRIX_H
#define _MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif
	
typedef struct
{
    MPI_Comm comm;

    NALU_HYPRE_Int      beg_row;
    NALU_HYPRE_Int      end_row;
    NALU_HYPRE_Int     *beg_rows;
    NALU_HYPRE_Int     *end_rows;

    Mem     *mem;

    NALU_HYPRE_Int     *lens;
    NALU_HYPRE_Int    **inds;
    NALU_HYPRE_Real **vals;

    NALU_HYPRE_Int     num_recv;
    NALU_HYPRE_Int     num_send;

    NALU_HYPRE_Int     sendlen;
    NALU_HYPRE_Int     recvlen;

    NALU_HYPRE_Int    *sendind;
    NALU_HYPRE_Real *sendbuf;
    NALU_HYPRE_Real *recvbuf;

    nalu_hypre_MPI_Request *recv_req;
    nalu_hypre_MPI_Request *send_req;
    nalu_hypre_MPI_Request *recv_req2;
    nalu_hypre_MPI_Request *send_req2;
    nalu_hypre_MPI_Status  *statuses;

    struct numbering *numb;
}
Matrix;

Matrix *MatrixCreate(MPI_Comm comm, NALU_HYPRE_Int beg_row, NALU_HYPRE_Int end_row);
Matrix *MatrixCreateLocal(NALU_HYPRE_Int beg_row, NALU_HYPRE_Int end_row);
void MatrixDestroy(Matrix *mat);
void MatrixSetRow(Matrix *mat, NALU_HYPRE_Int row, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind, NALU_HYPRE_Real *val);
void MatrixGetRow(Matrix *mat, NALU_HYPRE_Int row, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp, NALU_HYPRE_Real **valp);
NALU_HYPRE_Int  MatrixRowPe(Matrix *mat, NALU_HYPRE_Int row);
void MatrixPrint(Matrix *mat, char *filename);
void MatrixRead(Matrix *mat, char *filename);
void RhsRead(NALU_HYPRE_Real *rhs, Matrix *mat, char *filename);
NALU_HYPRE_Int  MatrixNnz(Matrix *mat);

void MatrixComplete(Matrix *mat);
void MatrixMatvec(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);
void MatrixMatvecSerial(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);
void MatrixMatvecTrans(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y);

#ifdef __cplusplus
}
#endif

#endif /* _MATRIX_H */
