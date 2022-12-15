/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * DiagScale - Diagonal scaling.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "math.h"
#include "Common.h"
#include "Matrix.h"
#include "RowPatt.h"
#include "DiagScale.h"
#include "OrderStat.h"
#include "Mem.h"

NALU_HYPRE_Int FindNumReplies(MPI_Comm comm, NALU_HYPRE_Int *replies_list);

#define DIAG_VALS_TAG      225
#define DIAG_INDS_TAG      226

/*--------------------------------------------------------------------------
 * ExchangeDiagEntries - Given a list of indices of diagonal entries required
 * by this processor, "reqind" of length "reqlen", return a list of 
 * corresponding diagonal entries, "diags".  Used internally only by
 * DiagScaleCreate.
 *
 * comm   - MPI communicator (input)
 * mat    - matrix used to map row and column numbers to processors (input)
 * reqlen - length of request list (input)
 * reqind - list of indices (input)
 * diags  - corresponding list of diagonal entries (output)
 * num_requests - number of requests (output)
 * requests - request handles, used to check that all responses are back 
 *            (output)
 * replies_list - array that indicates who we sent message to (output)
 *--------------------------------------------------------------------------*/

static void ExchangeDiagEntries(MPI_Comm comm, Matrix *mat, NALU_HYPRE_Int reqlen, 
  NALU_HYPRE_Int *reqind, NALU_HYPRE_Real *diags, NALU_HYPRE_Int *num_requests, nalu_hypre_MPI_Request *requests,
  NALU_HYPRE_Int *replies_list)
{
    nalu_hypre_MPI_Request request;
    NALU_HYPRE_Int i, j, this_pe;

    nalu_hypre_shell_sort(reqlen, reqind);

    *num_requests = 0;

    for (i=0; i<reqlen; i=j) /* j is set below */
    {
        /* The processor that owns the row with index reqind[i] */
        this_pe = MatrixRowPe(mat, reqind[i]);

        /* Figure out other rows we need from this_pe */
        for (j=i+1; j<reqlen; j++)
        {
            /* if row is on different pe */
            if (reqind[j] < mat->beg_rows[this_pe] ||
                reqind[j] > mat->end_rows[this_pe])
                   break;
        }

        /* Post receive for diagonal values */
        nalu_hypre_MPI_Irecv(&diags[i], j-i, nalu_hypre_MPI_REAL, this_pe, DIAG_VALS_TAG, 
	    comm, &requests[*num_requests]);

        /* Request rows in reqind[i..j-1] */
        nalu_hypre_MPI_Isend(&reqind[i], j-i, NALU_HYPRE_MPI_INT, this_pe, DIAG_INDS_TAG,
            comm, &request);
        nalu_hypre_MPI_Request_free(&request);
        (*num_requests)++;

	if (replies_list != NULL)
	    replies_list[this_pe] = 1;
    }
}

/*--------------------------------------------------------------------------
 * ExchangeDiagEntriesServer - Receive requests for diagonal entries and
 * send replies.  Used internally only by DiagScaleCreate.
 * 
 * comm   - MPI communicator (input)
 * mat    - matrix used to map row and column numbers to processors (input)
 * local_diags - local diagonal entries (input)
 * num_requests - number of requests to be received (input)
 *--------------------------------------------------------------------------*/

static void ExchangeDiagEntriesServer(MPI_Comm comm, Matrix *mat, 
  NALU_HYPRE_Real *local_diags, NALU_HYPRE_Int num_requests, Mem *mem, nalu_hypre_MPI_Request *requests)
{
    nalu_hypre_MPI_Status status;
    NALU_HYPRE_Int *recvbuf;
    NALU_HYPRE_Real *sendbuf;
    NALU_HYPRE_Int i, j, source, count;

    /* recvbuf contains requested indices */
    /* sendbuf contains corresponding diagonal entries */

    for (i=0; i<num_requests; i++)
    {
        nalu_hypre_MPI_Probe(nalu_hypre_MPI_ANY_SOURCE, DIAG_INDS_TAG, comm, &status);
        source = status.nalu_hypre_MPI_SOURCE;
	nalu_hypre_MPI_Get_count(&status, NALU_HYPRE_MPI_INT, &count);

        recvbuf = (NALU_HYPRE_Int *) MemAlloc(mem, count*sizeof(NALU_HYPRE_Int));
        sendbuf = (NALU_HYPRE_Real *) MemAlloc(mem, count*sizeof(NALU_HYPRE_Real));

        /*nalu_hypre_MPI_Recv(recvbuf, count, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_ANY_SOURCE, */
        nalu_hypre_MPI_Recv(recvbuf, count, NALU_HYPRE_MPI_INT, source, 
	    DIAG_INDS_TAG, comm, &status);
        source = status.nalu_hypre_MPI_SOURCE;

	/* Construct reply message of diagonal entries in sendbuf */
        for (j=0; j<count; j++)
	    sendbuf[j] = local_diags[recvbuf[j] - mat->beg_row];

	/* Use ready-mode send, since receives already posted */
	nalu_hypre_MPI_Irsend(sendbuf, count, nalu_hypre_MPI_REAL, source, 
	    DIAG_VALS_TAG, comm, &requests[i]);
    }
}

/*--------------------------------------------------------------------------
 * DiagScaleCreate - Return (a pointer to) a diagonal scaling object.
 * Scale using the diagonal of A.  Use the list of external indices
 * from the numbering object "numb".
 *--------------------------------------------------------------------------*/

DiagScale *DiagScaleCreate(Matrix *A, Numbering *numb)
{
    nalu_hypre_MPI_Request *requests;
    nalu_hypre_MPI_Status  *statuses;
    NALU_HYPRE_Int npes, row, j, num_requests, num_replies, *replies_list;
    NALU_HYPRE_Int len, *ind;
    NALU_HYPRE_Real *val, *temp;

    Mem *mem;
    nalu_hypre_MPI_Request *requests2;

    DiagScale *p = nalu_hypre_TAlloc(DiagScale, 1, NALU_HYPRE_MEMORY_HOST);

    /* Storage for local diagonal entries */
    p->local_diags = (NALU_HYPRE_Real *) 
        nalu_hypre_TAlloc(NALU_HYPRE_Real, (A->end_row - A->beg_row + 1) , NALU_HYPRE_MEMORY_HOST);

    /* Extract the local diagonal entries */
    for (row=0; row<=A->end_row - A->beg_row; row++)
    {
	MatrixGetRow(A, row, &len, &ind, &val);

        p->local_diags[row] = 1.0; /* in case no diag entry */

        for (j=0; j<len; j++)
        {
            if (ind[j] == row)
            {
                if (val[j] != 0.0)
                    p->local_diags[row] = 1.0 / sqrt(ABS(val[j]));
                break;
            }
        }
    }

    /* Get the list of diagonal indices that we need.
       This is simply the external indices */
    /* ExchangeDiagEntries will sort the list - so give it a copy */
    len = numb->num_ind - numb->num_loc;
    ind = NULL;
    p->ext_diags = NULL;
    if (len)
    {
        ind = nalu_hypre_TAlloc(NALU_HYPRE_Int, len , NALU_HYPRE_MEMORY_HOST);
        nalu_hypre_TMemcpy(ind,  &numb->local_to_global[numb->num_loc], NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

        /* buffer for receiving diagonal values from other processors */
        p->ext_diags = nalu_hypre_TAlloc(NALU_HYPRE_Real, len , NALU_HYPRE_MEMORY_HOST);
    }

    nalu_hypre_MPI_Comm_size(A->comm, &npes);
    requests = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
    statuses = nalu_hypre_TAlloc(nalu_hypre_MPI_Status, npes , NALU_HYPRE_MEMORY_HOST);
    replies_list = nalu_hypre_CTAlloc(NALU_HYPRE_Int, npes, NALU_HYPRE_MEMORY_HOST);

    ExchangeDiagEntries(A->comm, A, len, ind, p->ext_diags, &num_requests, 
        requests, replies_list);

    num_replies = FindNumReplies(A->comm, replies_list);
    nalu_hypre_TFree(replies_list,NALU_HYPRE_MEMORY_HOST);

    mem = MemCreate();
    requests2 = NULL;
    if (num_replies)
        requests2 = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, num_replies , NALU_HYPRE_MEMORY_HOST);

    ExchangeDiagEntriesServer(A->comm, A, p->local_diags, num_replies,
	mem, requests2);

    /* Wait for all replies */
    nalu_hypre_MPI_Waitall(num_requests, requests, statuses);
    nalu_hypre_TFree(requests,NALU_HYPRE_MEMORY_HOST);

    p->offset = A->end_row - A->beg_row + 1;

    /* ind contains global indices corresponding to order that entries
       are stored in ext_diags.  Reorder ext_diags in original ordering */
    NumberingGlobalToLocal(numb, len, ind, ind);
    temp = NULL;
    if (len)
        temp = nalu_hypre_TAlloc(NALU_HYPRE_Real, len , NALU_HYPRE_MEMORY_HOST);
    for (j=0; j<len; j++)
	temp[ind[j]-p->offset] = p->ext_diags[j];

    nalu_hypre_TFree(ind,NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(p->ext_diags,NALU_HYPRE_MEMORY_HOST);
    p->ext_diags = temp;

    /* Wait for all sends */
    nalu_hypre_MPI_Waitall(num_replies, requests2, statuses);
    nalu_hypre_TFree(requests2,NALU_HYPRE_MEMORY_HOST);
    MemDestroy(mem);

    nalu_hypre_TFree(statuses,NALU_HYPRE_MEMORY_HOST);
    return p;
}

/*--------------------------------------------------------------------------
 * DiagScaleDestroy - Destroy a diagonal scale object.
 *--------------------------------------------------------------------------*/

void DiagScaleDestroy(DiagScale *p)
{
    nalu_hypre_TFree(p->local_diags,NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(p->ext_diags,NALU_HYPRE_MEMORY_HOST);

    nalu_hypre_TFree(p,NALU_HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * DiagScaleGet -  Returns scale factor given a row number in local indexing.
 * The factor is the reciprocal of the square root of the diagonal entry.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real DiagScaleGet(DiagScale *p, NALU_HYPRE_Int index)
{
    if (index < p->offset)
    {
        return p->local_diags[index];
    }
    else
    {
        return p->ext_diags[index - p->offset];
    }
}
