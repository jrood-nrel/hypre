/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix - Matrix stored and accessible by rows.  Indices and values for
 * the matrix nonzeros are copied into the matrix a row at a time, in any
 * order using the MatrixGetRow function.  The MatrixPutRow function returns
 * a pointer to the indices and values of a row.  The matrix has a set of
 * row and column indices such that these indices begin at "beg" and end
 * at "end", where 0 <= "beg" <= "end".  In other words, the matrix indices
 * have any nonnegative base value, and the base values of the row and column
 * indices must agree.
 *
 *****************************************************************************/

#include <stdlib.h>
//#include <memory.h>
#include "Common.h"
#include "Matrix.h"
#include "Numbering.h"

#define MAX_NZ_PER_ROW 1000

/*--------------------------------------------------------------------------
 * MatrixCreate - Return (a pointer to) a matrix object.
 *--------------------------------------------------------------------------*/

Matrix *MatrixCreate(MPI_Comm comm, NALU_HYPRE_Int beg_row, NALU_HYPRE_Int end_row)
{
   NALU_HYPRE_Int num_rows, mype, npes;

   Matrix *mat = nalu_hypre_TAlloc(Matrix, 1, NALU_HYPRE_MEMORY_HOST);

   mat->comm = comm;

   mat->beg_row = beg_row;
   mat->end_row = end_row;

   mat->mem = (Mem *) MemCreate();

   num_rows = mat->end_row - mat->beg_row + 1;

   mat->lens = (NALU_HYPRE_Int *)     MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Int));
   mat->inds = (NALU_HYPRE_Int **)    MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Int *));
   mat->vals = (NALU_HYPRE_Real **) MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Real *));

   /* Send beg_row and end_row to all processors */
   /* This is needed in order to map row numbers to processors */

   nalu_hypre_MPI_Comm_rank(comm, &mype);
   nalu_hypre_MPI_Comm_size(comm, &npes);

   mat->beg_rows = (NALU_HYPRE_Int *) MemAlloc(mat->mem, npes * sizeof(NALU_HYPRE_Int));
   mat->end_rows = (NALU_HYPRE_Int *) MemAlloc(mat->mem, npes * sizeof(NALU_HYPRE_Int));

   nalu_hypre_MPI_Allgather(&beg_row, 1, NALU_HYPRE_MPI_INT, mat->beg_rows, 1, NALU_HYPRE_MPI_INT, comm);
   nalu_hypre_MPI_Allgather(&end_row, 1, NALU_HYPRE_MPI_INT, mat->end_rows, 1, NALU_HYPRE_MPI_INT, comm);

   mat->num_recv = 0;
   mat->num_send = 0;

   mat->recv_req  = NULL;
   mat->send_req  = NULL;
   mat->recv_req2 = NULL;
   mat->send_req2 = NULL;
   mat->statuses  = NULL;

   mat->sendind = NULL;
   mat->sendbuf = NULL;
   mat->recvbuf = NULL;

   mat->numb = NULL;

   return mat;
}

/*--------------------------------------------------------------------------
 * MatrixCreateLocal - Return (a pointer to) a matrix object.
 * The matrix created by this call is a local matrix, not a global matrix.
 *--------------------------------------------------------------------------*/

Matrix *MatrixCreateLocal(NALU_HYPRE_Int beg_row, NALU_HYPRE_Int end_row)
{
   NALU_HYPRE_Int num_rows;

   Matrix *mat = nalu_hypre_TAlloc(Matrix, 1, NALU_HYPRE_MEMORY_HOST);

   mat->comm = nalu_hypre_MPI_COMM_NULL;

   mat->beg_row = beg_row;
   mat->end_row = end_row;

   mat->mem = (Mem *) MemCreate();

   num_rows = mat->end_row - mat->beg_row + 1;

   mat->lens = (NALU_HYPRE_Int *)     MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Int));
   mat->inds = (NALU_HYPRE_Int **)    MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Int *));
   mat->vals = (NALU_HYPRE_Real **) MemAlloc(mat->mem, num_rows * sizeof(NALU_HYPRE_Real *));

   /* Send beg_row and end_row to all processors */
   /* This is needed in order to map row numbers to processors */

   mat->beg_rows = NULL;
   mat->end_rows = NULL;

   mat->num_recv = 0;
   mat->num_send = 0;

   mat->recv_req  = NULL;
   mat->send_req  = NULL;
   mat->recv_req2 = NULL;
   mat->send_req2 = NULL;
   mat->statuses  = NULL;

   mat->sendind = NULL;
   mat->sendbuf = NULL;
   mat->recvbuf = NULL;

   mat->numb = NULL;

   return mat;
}

/*--------------------------------------------------------------------------
 * MatrixDestroy - Destroy a matrix object "mat".
 *--------------------------------------------------------------------------*/

void MatrixDestroy(Matrix *mat)
{
   NALU_HYPRE_Int i;

   for (i=0; i<mat->num_recv; i++)
      nalu_hypre_MPI_Request_free(&mat->recv_req[i]);

   for (i=0; i<mat->num_send; i++)
      nalu_hypre_MPI_Request_free(&mat->send_req[i]);

   for (i=0; i<mat->num_send; i++)
      nalu_hypre_MPI_Request_free(&mat->recv_req2[i]);

   for (i=0; i<mat->num_recv; i++)
      nalu_hypre_MPI_Request_free(&mat->send_req2[i]);

   nalu_hypre_TFree(mat->recv_req,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->send_req,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->recv_req2,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->send_req2,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->statuses,NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(mat->sendind,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->sendbuf,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mat->recvbuf,NALU_HYPRE_MEMORY_HOST);

   MemDestroy(mat->mem);

   if (mat->numb)
      NumberingDestroy(mat->numb);

   nalu_hypre_TFree(mat,NALU_HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * MatrixSetRow - Set a row in a matrix.  Only local rows can be set.
 * Once a row has been set, it should not be set again, or else the
 * memory used by the existing row will not be recovered until
 * the matrix is destroyed.  "row" is in global coordinate numbering.
 *--------------------------------------------------------------------------*/

void MatrixSetRow(Matrix *mat, NALU_HYPRE_Int row, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind, NALU_HYPRE_Real *val)
{
   row -= mat->beg_row;

   mat->lens[row] = len;
   mat->inds[row] = (NALU_HYPRE_Int *) MemAlloc(mat->mem, len*sizeof(NALU_HYPRE_Int));
   mat->vals[row] = (NALU_HYPRE_Real *) MemAlloc(mat->mem, len*sizeof(NALU_HYPRE_Real));

   if (ind != NULL)
   {
      //nalu_hypre_TMemcpy(mat->inds[row], ind, NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      memcpy(mat->inds[row], ind, sizeof(NALU_HYPRE_Int) * len);
   }

   if (val != NULL)
   {
      //nalu_hypre_TMemcpy(mat->vals[row], val, NALU_HYPRE_Real, len, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      memcpy(mat->vals[row], val, sizeof(NALU_HYPRE_Real) * len);
   }
}

/*--------------------------------------------------------------------------
 * MatrixGetRow - Get a *local* row in a matrix.
 *--------------------------------------------------------------------------*/

void MatrixGetRow(Matrix *mat, NALU_HYPRE_Int row, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp, NALU_HYPRE_Real **valp)
{
   *lenp = mat->lens[row];
   *indp = mat->inds[row];
   *valp = mat->vals[row];
}

/*--------------------------------------------------------------------------
 * MatrixRowPe - Map "row" to a processor number.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int MatrixRowPe(Matrix *mat, NALU_HYPRE_Int row)
{
   NALU_HYPRE_Int npes, pe;

   NALU_HYPRE_Int *beg = mat->beg_rows;
   NALU_HYPRE_Int *end = mat->end_rows;

   nalu_hypre_MPI_Comm_size(mat->comm, &npes);

   for (pe=0; pe<npes; pe++)
   {
      if (row >= beg[pe] && row <= end[pe])
         return pe;
   }

   nalu_hypre_printf("MatrixRowPe: could not map row %d.\n", row);
   PARASAILS_EXIT;

   return -1; /* for picky compilers */
}

/*--------------------------------------------------------------------------
 * MatrixNnz - Return total number of nonzeros in preconditioner.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int MatrixNnz(Matrix *mat)
{
   NALU_HYPRE_Int num_local, i, total, alltotal;

   num_local = mat->end_row - mat->beg_row + 1;

   total = 0;
   for (i=0; i<num_local; i++)
      total += mat->lens[i];

   nalu_hypre_MPI_Allreduce(&total, &alltotal, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, mat->comm);

   return alltotal;
}

/*--------------------------------------------------------------------------
 * MatrixPrint - Print a matrix to a file "filename".  Each processor
 * appends to the file in order, but the file is overwritten if it exists.
 *--------------------------------------------------------------------------*/

void MatrixPrint(Matrix *mat, char *filename)
{
   NALU_HYPRE_Int mype, npes, pe;
   NALU_HYPRE_Int row, i, len, *ind;
   NALU_HYPRE_Real *val;

   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);
   nalu_hypre_MPI_Comm_size(mat->comm, &npes);

   for (pe=0; pe<npes; pe++)
   {
      nalu_hypre_MPI_Barrier(mat->comm);

      if (mype == pe)
      {
         FILE *file = fopen(filename, (pe==0 ? "w" : "a"));
         nalu_hypre_assert(file != NULL);

         for (row=0; row<=mat->end_row - mat->beg_row; row++)
         {
            MatrixGetRow(mat, row, &len, &ind, &val);

            for (i=0; i<len; i++)
               nalu_hypre_fprintf(file, "%d %d %.14e\n",
                     row + mat->beg_row,
                     mat->numb->local_to_global[ind[i]], val[i]);
         }

         fclose(file);
      }
   }
}

/*--------------------------------------------------------------------------
 * MatrixReadMaster - MatrixRead routine for processor 0.  Internal use.
 *--------------------------------------------------------------------------*/

static void MatrixReadMaster(Matrix *mat, char *filename)
{
   MPI_Comm comm = mat->comm;
   NALU_HYPRE_Int mype, npes;
   FILE *file;
   NALU_HYPRE_Int ret;
   NALU_HYPRE_Int num_rows, curr_proc;
   NALU_HYPRE_Int row, col;
   NALU_HYPRE_Real value;
   nalu_hypre_longint offset;
   nalu_hypre_longint outbuf;

   NALU_HYPRE_Int curr_row;
   NALU_HYPRE_Int len;
   NALU_HYPRE_Int ind[MAX_NZ_PER_ROW];
   NALU_HYPRE_Real val[MAX_NZ_PER_ROW];

   char line[100];
   NALU_HYPRE_Int oldrow;

   nalu_hypre_MPI_Request request;
   nalu_hypre_MPI_Status  status;

   nalu_hypre_MPI_Comm_size(mat->comm, &npes);
   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);

   file = fopen(filename, "r");
   nalu_hypre_assert(file != NULL);

   if (fgets(line, 100, file) == NULL)
   {
      nalu_hypre_fprintf(stderr, "Error reading file.\n");
      PARASAILS_EXIT;
   }

#ifdef EMSOLVE
   ret = nalu_hypre_sscanf(line, "%*d %d %*d %*d", &num_rows);
   for (row=0; row<num_rows; row++)
      nalu_hypre_fscanf(file, "%*d");
#else
   ret = nalu_hypre_sscanf(line, "%d %*d %*d", &num_rows);
#endif

   offset = ftell(file);
   nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);

   request = nalu_hypre_MPI_REQUEST_NULL;
   curr_proc = 1; /* proc for which we are looking for the beginning */
   while (curr_proc < npes)
   {
      if (row == mat->beg_rows[curr_proc])
      {
         nalu_hypre_MPI_Wait(&request, &status);
         outbuf = offset;
         nalu_hypre_MPI_Isend(&outbuf, 1, nalu_hypre_MPI_LONG, curr_proc, 0, comm, &request);
         curr_proc++;
      }
      offset = ftell(file);
      oldrow = row;
      nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);
      if (oldrow > row)
      {
         nalu_hypre_fprintf(stderr, "Matrix file is not sorted by rows.\n");
         PARASAILS_EXIT;
      }
   }

   /* Now read our own part */
   rewind(file);
   if (fgets(line, 100, file) == NULL)
   {
      nalu_hypre_fprintf(stderr, "Error reading file.\n");
      PARASAILS_EXIT;
   }

#ifdef EMSOLVE
   ret = nalu_hypre_sscanf(line, "%*d %d %*d %*d", &num_rows);
   for (row=0; row<num_rows; row++)
      nalu_hypre_fscanf(file, "%*d");
#else
   ret = nalu_hypre_sscanf(line, "%d %*d %*d", &num_rows);
#endif

   ret = nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);
   curr_row = row;
   len = 0;

   while (ret != EOF && row <= mat->end_row)
   {
      if (row != curr_row)
      {
         /* store this row */
         MatrixSetRow(mat, curr_row, len, ind, val);

         curr_row = row;

         /* reset row pointer */
         len = 0;
      }

      if (len >= MAX_NZ_PER_ROW)
      {
         nalu_hypre_fprintf(stderr, "The matrix has exceeded %d\n", MAX_NZ_PER_ROW);
         nalu_hypre_fprintf(stderr, "nonzeros per row.  Internal buffers must be\n");
         nalu_hypre_fprintf(stderr, "increased to continue.\n");
         PARASAILS_EXIT;
      }

      ind[len] = col;
      val[len] = value;
      len++;

      ret = nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);
   }

   /* Store the final row */
   if (ret == EOF || row > mat->end_row)
      MatrixSetRow(mat, mat->end_row, len, ind, val);

   fclose(file);

   nalu_hypre_MPI_Wait(&request, &status);
}

/*--------------------------------------------------------------------------
 * MatrixReadSlave - MatrixRead routine for other processors.  Internal use.
 *--------------------------------------------------------------------------*/

static void MatrixReadSlave(Matrix *mat, char *filename)
{
   MPI_Comm comm = mat->comm;
   nalu_hypre_MPI_Status status;
   NALU_HYPRE_Int mype;
   FILE *file;
   NALU_HYPRE_Int ret;
   NALU_HYPRE_Int row, col;
   NALU_HYPRE_Real value;
   nalu_hypre_longint offset;

   NALU_HYPRE_Int curr_row;
   NALU_HYPRE_Int len;
   NALU_HYPRE_Int ind[MAX_NZ_PER_ROW];
   NALU_HYPRE_Real val[MAX_NZ_PER_ROW];

   NALU_HYPRE_Real time0, time1;

   file = fopen(filename, "r");
   nalu_hypre_assert(file != NULL);

   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);

   nalu_hypre_MPI_Recv(&offset, 1, nalu_hypre_MPI_LONG, 0, 0, comm, &status);
   time0 = nalu_hypre_MPI_Wtime();

   ret = fseek(file, offset, SEEK_SET);
   nalu_hypre_assert(ret == 0);

   ret = nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);
   curr_row = row;
   len = 0;

   while (ret != EOF && row <= mat->end_row)
   {
      if (row != curr_row)
      {
         /* store this row */
         MatrixSetRow(mat, curr_row, len, ind, val);

         curr_row = row;

         /* reset row pointer */
         len = 0;
      }

      if (len >= MAX_NZ_PER_ROW)
      {
         nalu_hypre_fprintf(stderr, "The matrix has exceeded %d\n", MAX_NZ_PER_ROW);
         nalu_hypre_fprintf(stderr, "nonzeros per row.  Internal buffers must be\n");
         nalu_hypre_fprintf(stderr, "increased to continue.\n");
         PARASAILS_EXIT;
      }

      ind[len] = col;
      val[len] = value;
      len++;

      ret = nalu_hypre_fscanf(file, "%d %d %lf", &row, &col, &value);
   }

   /* Store the final row */
   if (ret == EOF || row > mat->end_row)
      MatrixSetRow(mat, mat->end_row, len, ind, val);

   fclose(file);
   time1 = nalu_hypre_MPI_Wtime();
   nalu_hypre_printf("%d: Time for slave read: %f\n", mype, time1-time0);
}

/*--------------------------------------------------------------------------
 * MatrixRead - Read a matrix file "filename" from disk and store in the
 * matrix "mat" which has already been created using MatrixCreate.  The format
 * assumes no nonzero rows, the rows are in order, and there will be at least
 * one row per processor.
 *--------------------------------------------------------------------------*/

void MatrixRead(Matrix *mat, char *filename)
{
   NALU_HYPRE_Int mype;
   NALU_HYPRE_Real time0, time1;

   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);

   time0 = nalu_hypre_MPI_Wtime();
   if (mype == 0)
      MatrixReadMaster(mat, filename);
   else
      MatrixReadSlave(mat, filename);
   time1 = nalu_hypre_MPI_Wtime();
   nalu_hypre_printf("%d: Time for reading matrix: %f\n", mype, time1-time0);

   MatrixComplete(mat);
}

/*--------------------------------------------------------------------------
 * RhsRead - Read a right-hand side file "filename" from disk and store in the
 * location pointed to by "rhs".  "mat" is needed to provide the partitioning
 * information.  The expected format is: a header line (n, nrhs) followed
 * by n values.  Also allows isis format, indicated by 1 NALU_HYPRE_Int in first line.
 *--------------------------------------------------------------------------*/

void RhsRead(NALU_HYPRE_Real *rhs, Matrix *mat, char *filename)
{
   FILE *file;
   nalu_hypre_MPI_Status status;
   NALU_HYPRE_Int mype, npes;
   NALU_HYPRE_Int num_rows, num_local, pe, i, converted;
   NALU_HYPRE_Real *buffer = NULL;
   NALU_HYPRE_Int buflen = 0;
   char line[100];
   NALU_HYPRE_Int dummy;

   nalu_hypre_MPI_Comm_size(mat->comm, &npes);
   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);

   num_local = mat->end_row - mat->beg_row + 1;

   if (mype != 0)
   {
      nalu_hypre_MPI_Recv(rhs, num_local, nalu_hypre_MPI_REAL, 0, 0, mat->comm, &status);
      return;
   }

   file = fopen(filename, "r");
   nalu_hypre_assert(file != NULL);

   if (fgets(line, 100, file) == NULL)
   {
      nalu_hypre_fprintf(stderr, "Error reading file.\n");
      PARASAILS_EXIT;
   }
   converted = nalu_hypre_sscanf(line, "%d %d", &num_rows, &dummy);
   nalu_hypre_assert(num_rows == mat->end_rows[npes-1]);

   /* Read own rows first */
   for (i=0; i<num_local; i++)
      if (converted == 1) /* isis format */
         nalu_hypre_fscanf(file, "%*d %lf", &rhs[i]);
      else
         nalu_hypre_fscanf(file, "%lf", &rhs[i]);

   for (pe=1; pe<npes; pe++)
   {
      num_local = mat->end_rows[pe] - mat->beg_rows[pe]+ 1;

      if (buflen < num_local)
      {
         nalu_hypre_TFree(buffer,NALU_HYPRE_MEMORY_HOST);
         buflen = num_local;
         buffer = nalu_hypre_TAlloc(NALU_HYPRE_Real, buflen , NALU_HYPRE_MEMORY_HOST);
      }

      for (i=0; i<num_local; i++)
         if (converted == 1) /* isis format */
            nalu_hypre_fscanf(file, "%*d %lf", &buffer[i]);
         else
            nalu_hypre_fscanf(file, "%lf", &buffer[i]);

      nalu_hypre_MPI_Send(buffer, num_local, nalu_hypre_MPI_REAL, pe, 0, mat->comm);
   }

   nalu_hypre_TFree(buffer,NALU_HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * SetupReceives
 *--------------------------------------------------------------------------*/

static void SetupReceives(Matrix *mat, NALU_HYPRE_Int reqlen, NALU_HYPRE_Int *reqind, NALU_HYPRE_Int *outlist)
{
   NALU_HYPRE_Int i, j, this_pe, mype;
   nalu_hypre_MPI_Request request;
   MPI_Comm comm = mat->comm;
   NALU_HYPRE_Int num_local = mat->end_row - mat->beg_row + 1;

   nalu_hypre_MPI_Comm_rank(comm, &mype);

   mat->num_recv = 0;

   /* Allocate recvbuf */
   /* recvbuf has numlocal entires saved for local part of x, used in matvec */
   mat->recvlen = reqlen; /* used for the transpose multiply */
   mat->recvbuf = nalu_hypre_TAlloc(NALU_HYPRE_Real, (reqlen+num_local) , NALU_HYPRE_MEMORY_HOST);

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

      /* Request rows in reqind[i..j-1] */
      nalu_hypre_MPI_Isend(&reqind[i], j-i, NALU_HYPRE_MPI_INT, this_pe, 444, comm, &request);
      nalu_hypre_MPI_Request_free(&request);

      /* Count of number of number of indices needed from this_pe */
      outlist[this_pe] = j-i;

      nalu_hypre_MPI_Recv_init(&mat->recvbuf[i+num_local], j-i, nalu_hypre_MPI_REAL, this_pe, 555,
            comm, &mat->recv_req[mat->num_recv]);

      nalu_hypre_MPI_Send_init(&mat->recvbuf[i+num_local], j-i, nalu_hypre_MPI_REAL, this_pe, 666,
            comm, &mat->send_req2[mat->num_recv]);

      mat->num_recv++;
   }
}

/*--------------------------------------------------------------------------
 * SetupSends
 * This function will wait for all receives to complete.
 *--------------------------------------------------------------------------*/

static void SetupSends(Matrix *mat, NALU_HYPRE_Int *inlist)
{
   NALU_HYPRE_Int i, j, mype, npes;
   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status  *statuses;
   MPI_Comm comm = mat->comm;

   nalu_hypre_MPI_Comm_rank(comm, &mype);
   nalu_hypre_MPI_Comm_size(comm, &npes);

   requests = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
   statuses = nalu_hypre_TAlloc(nalu_hypre_MPI_Status, npes , NALU_HYPRE_MEMORY_HOST);

   /* Determine size of and allocate sendbuf and sendind */
   mat->sendlen = 0;
   for (i=0; i<npes; i++)
      mat->sendlen += inlist[i];
   mat->sendbuf = NULL;
   mat->sendind = NULL;
   if (mat->sendlen)
   {
      mat->sendbuf = nalu_hypre_TAlloc(NALU_HYPRE_Real, mat->sendlen , NALU_HYPRE_MEMORY_HOST);
      mat->sendind = nalu_hypre_TAlloc(NALU_HYPRE_Int, mat->sendlen , NALU_HYPRE_MEMORY_HOST);
   }

   j = 0;
   mat->num_send = 0;
   for (i=0; i<npes; i++)
   {
      if (inlist[i] != 0)
      {
         /* Post receive for the actual indices */
         nalu_hypre_MPI_Irecv(&mat->sendind[j], inlist[i], NALU_HYPRE_MPI_INT, i, 444, comm,
               &requests[mat->num_send]);

         /* Set up the send */
         nalu_hypre_MPI_Send_init(&mat->sendbuf[j], inlist[i], nalu_hypre_MPI_REAL, i, 555, comm,
               &mat->send_req[mat->num_send]);

         /* Set up the receive for the transpose  */
         nalu_hypre_MPI_Recv_init(&mat->sendbuf[j], inlist[i], nalu_hypre_MPI_REAL, i, 666, comm,
               &mat->recv_req2[mat->num_send]);

         mat->num_send++;
         j += inlist[i];
      }

   }

   nalu_hypre_MPI_Waitall(mat->num_send, requests, statuses);
   nalu_hypre_TFree(requests,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(statuses,NALU_HYPRE_MEMORY_HOST);

   /* convert global indices to local indices */
   /* these are all indices on this processor */
   for (i=0; i<mat->sendlen; i++)
      mat->sendind[i] -= mat->beg_row;
}

/*--------------------------------------------------------------------------
 * MatrixComplete
 *--------------------------------------------------------------------------*/

void MatrixComplete(Matrix *mat)
{
   NALU_HYPRE_Int mype, npes;
   NALU_HYPRE_Int *outlist, *inlist;
   NALU_HYPRE_Int row, len, *ind;
   NALU_HYPRE_Real *val;

   nalu_hypre_MPI_Comm_rank(mat->comm, &mype);
   nalu_hypre_MPI_Comm_size(mat->comm, &npes);

   mat->recv_req = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
   mat->send_req = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
   mat->recv_req2 = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
   mat->send_req2 = nalu_hypre_TAlloc(nalu_hypre_MPI_Request, npes , NALU_HYPRE_MEMORY_HOST);
   mat->statuses = nalu_hypre_TAlloc(nalu_hypre_MPI_Status, npes , NALU_HYPRE_MEMORY_HOST);

   outlist = nalu_hypre_CTAlloc(NALU_HYPRE_Int, npes, NALU_HYPRE_MEMORY_HOST);
   inlist  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, npes, NALU_HYPRE_MEMORY_HOST);

   /* Create Numbering object */
   mat->numb = NumberingCreate(mat, PARASAILS_NROWS);

   SetupReceives(mat, mat->numb->num_ind - mat->numb->num_loc,
         &mat->numb->local_to_global[mat->numb->num_loc], outlist);

   nalu_hypre_MPI_Alltoall(outlist, 1, NALU_HYPRE_MPI_INT, inlist, 1, NALU_HYPRE_MPI_INT, mat->comm);

   SetupSends(mat, inlist);

   nalu_hypre_TFree(outlist,NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(inlist,NALU_HYPRE_MEMORY_HOST);

   /* Convert to local indices */
   for (row=0; row<=mat->end_row - mat->beg_row; row++)
   {
      MatrixGetRow(mat, row, &len, &ind, &val);
      NumberingGlobalToLocal(mat->numb, len, ind, ind);
   }
}

/*--------------------------------------------------------------------------
 * MatrixMatvec
 * Can be done in place.
 *--------------------------------------------------------------------------*/

void MatrixMatvec(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
   NALU_HYPRE_Int row, i, len, *ind;
   NALU_HYPRE_Real *val, temp;
   NALU_HYPRE_Int num_local = mat->end_row - mat->beg_row + 1;

   /* Set up persistent communications */

   /* Assumes MatrixComplete has been called */

   /* Put components of x into the right outgoing buffers */
   for (i=0; i<mat->sendlen; i++)
      mat->sendbuf[i] = x[mat->sendind[i]];

   nalu_hypre_MPI_Startall(mat->num_recv, mat->recv_req);
   nalu_hypre_MPI_Startall(mat->num_send, mat->send_req);

   /* Copy local part of x into top part of recvbuf */
   for (i=0; i<num_local; i++)
      mat->recvbuf[i] = x[i];

   nalu_hypre_MPI_Waitall(mat->num_recv, mat->recv_req, mat->statuses);

   /* do the multiply */
#ifdef NALU_HYPRE_USING_OPENMP
#pragma omp parallel for private(row,len,ind,val,temp,i) schedule(static)
#endif
   for (row=0; row<=mat->end_row - mat->beg_row; row++)
   {
      MatrixGetRow(mat, row, &len, &ind, &val);

      temp = 0.0;
      for (i=0; i<len; i++)
      {
         temp = temp + val[i] * mat->recvbuf[ind[i]];
      }
      y[row] = temp;
   }

   nalu_hypre_MPI_Waitall(mat->num_send, mat->send_req, mat->statuses);
}

void MatrixMatvecSerial(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
   NALU_HYPRE_Int row, i, len, *ind;
   NALU_HYPRE_Real *val, temp;
   NALU_HYPRE_Int num_local = mat->end_row - mat->beg_row + 1;

   /* Set up persistent communications */

   /* Assumes MatrixComplete has been called */

   /* Put components of x into the right outgoing buffers */
   for (i=0; i<mat->sendlen; i++)
      mat->sendbuf[i] = x[mat->sendind[i]];

   nalu_hypre_MPI_Startall(mat->num_recv, mat->recv_req);
   nalu_hypre_MPI_Startall(mat->num_send, mat->send_req);

   /* Copy local part of x into top part of recvbuf */
   for (i=0; i<num_local; i++)
      mat->recvbuf[i] = x[i];

   nalu_hypre_MPI_Waitall(mat->num_recv, mat->recv_req, mat->statuses);

   /* do the multiply */
   for (row=0; row<=mat->end_row - mat->beg_row; row++)
   {
      MatrixGetRow(mat, row, &len, &ind, &val);

      temp = 0.0;
      for (i=0; i<len; i++)
      {
         temp = temp + val[i] * mat->recvbuf[ind[i]];
      }
      y[row] = temp;
   }

   nalu_hypre_MPI_Waitall(mat->num_send, mat->send_req, mat->statuses);
}

/*--------------------------------------------------------------------------
 * MatrixMatvecTrans
 * Can be done in place.
 *--------------------------------------------------------------------------*/

void MatrixMatvecTrans(Matrix *mat, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
   NALU_HYPRE_Int row, i, len, *ind;
   NALU_HYPRE_Real *val;
   NALU_HYPRE_Int num_local = mat->end_row - mat->beg_row + 1;

   /* Set up persistent communications */

   /* Assumes MatrixComplete has been called */

   /* Post receives for local parts of the solution y */
   nalu_hypre_MPI_Startall(mat->num_send, mat->recv_req2);

   /* initialize accumulator buffer to zero */
   for (i=0; i<mat->recvlen+num_local; i++)
      mat->recvbuf[i] = 0.0;

   /* do the multiply */
   for (row=0; row<=mat->end_row - mat->beg_row; row++)
   {
      MatrixGetRow(mat, row, &len, &ind, &val);

      for (i=0; i<len; i++)
      {
         mat->recvbuf[ind[i]] += val[i] * x[row];
      }
   }

   /* Now can send nonlocal parts of solution to other procs */
   nalu_hypre_MPI_Startall(mat->num_recv, mat->send_req2);

   /* copy local part of solution into y */
   for (i=0; i<num_local; i++)
      y[i] = mat->recvbuf[i];

   /* alternatively, loop over a wait any */
   nalu_hypre_MPI_Waitall(mat->num_send, mat->recv_req2, mat->statuses);

   /* add all the incoming partial sums to y */
   for (i=0; i<mat->sendlen; i++)
      y[mat->sendind[i]] += mat->sendbuf[i];

   nalu_hypre_MPI_Waitall(mat->num_recv, mat->send_req2, mat->statuses);
}
