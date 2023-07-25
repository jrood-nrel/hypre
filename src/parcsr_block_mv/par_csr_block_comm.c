/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_block_mv.h"
#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockCommHandleCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRCommHandle *
nalu_hypre_ParCSRBlockCommHandleCreate(NALU_HYPRE_Int job,
                                  NALU_HYPRE_Int bnnz,
                                  nalu_hypre_ParCSRCommPkg *comm_pkg,
                                  void *send_data,
                                  void *recv_data )
{
   NALU_HYPRE_Int      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm comm      = nalu_hypre_ParCSRCommPkgComm(comm_pkg);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   NALU_HYPRE_Int         num_requests;
   nalu_hypre_MPI_Request *requests;
   NALU_HYPRE_Int    i, j, my_id, num_procs, ip, vec_start, vec_len;
   NALU_HYPRE_Complex *d_send_data = (NALU_HYPRE_Complex *) send_data;
   NALU_HYPRE_Complex *d_recv_data = (NALU_HYPRE_Complex *) recv_data;

   /*---------------------------------------------------------------------------
    * job = 1 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a Matvec,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    * job = 2 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a MatvecT,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    *------------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   j = 0;

   switch (job)
   {
      case  1:
      {
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
               (nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start) * bnnz;
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
               (nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start) * bnnz;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  2:
      {

         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len =
               (nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start) * bnnz;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start * bnnz], vec_len,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len =
               (nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start) * bnnz;
            nalu_hypre_MPI_Isend(&d_send_data[vec_start * bnnz], vec_len,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = nalu_hypre_CTAlloc(nalu_hypre_ParCSRCommHandle,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRCommHandleCommPkg(comm_handle)     = comm_pkg;
   nalu_hypre_ParCSRCommHandleSendData(comm_handle)    = send_data;
   nalu_hypre_ParCSRCommHandleRecvData(comm_handle)    = recv_data;
   nalu_hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   nalu_hypre_ParCSRCommHandleRequests(comm_handle)    = requests;
   return ( comm_handle );
}

/*--------------------------------------------------------------------
  nalu_hypre_ParCSRBlockCommHandleDestroy
  *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockCommHandleDestroy(nalu_hypre_ParCSRCommHandle *comm_handle)
{
   nalu_hypre_MPI_Status          *status0;

   if ( comm_handle == NULL ) { return nalu_hypre_error_flag; }

   if (nalu_hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      status0 = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,
                              nalu_hypre_ParCSRCommHandleNumRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Waitall(nalu_hypre_ParCSRCommHandleNumRequests(comm_handle),
                        nalu_hypre_ParCSRCommHandleRequests(comm_handle), status0);
      nalu_hypre_TFree(status0, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_handle, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixCreateAssumedPartition -
 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixCreateAssumedPartition( nalu_hypre_ParCSRBlockMatrix *matrix)
{
   NALU_HYPRE_BigInt global_num_cols;
   NALU_HYPRE_Int myid;
   NALU_HYPRE_BigInt  col_start = 0, col_end = 0;

   MPI_Comm   comm;

   nalu_hypre_IJAssumedPart *apart;

   global_num_cols = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   comm = nalu_hypre_ParCSRBlockMatrixComm(matrix);

   /* find out my actualy range of rows and columns */
   col_start =  nalu_hypre_ParCSRBlockMatrixFirstColDiag(matrix);
   col_end =  nalu_hypre_ParCSRBlockMatrixLastColDiag(matrix);

   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = nalu_hypre_CTAlloc(nalu_hypre_IJAssumedPart,  1, NALU_HYPRE_MEMORY_HOST);

   /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   nalu_hypre_GetAssumedPartitionRowRange(comm, myid, 0, global_num_cols,
                                     &(apart->row_start), &(apart->row_end));

   /*allocate some space for the partition of the assumed partition */
   apart->length = 0;
   /*room for 10 owners of the assumed partition*/
   apart->storage_length = 10; /*need to be >=1 */
   apart->proc_list = nalu_hypre_TAlloc(NALU_HYPRE_Int,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_start_list =   nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);
   apart->row_end_list =   nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  apart->storage_length, NALU_HYPRE_MEMORY_HOST);

   /* now we want to reconcile our actual partition with the assumed partition */
   nalu_hypre_LocateAssumedPartition(comm, col_start, col_end,
                                0, global_num_cols, apart, myid);

   /* this partition will be saved in the matrix data structure until the matrix
    * is destroyed */
   nalu_hypre_ParCSRBlockMatrixAssumedPartition(matrix) = apart;

   return nalu_hypre_error_flag;

}

/*--------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDestroyAssumedPartition
 *--------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixDestroyAssumedPartition( nalu_hypre_ParCSRBlockMatrix *matrix )
{

   nalu_hypre_IJAssumedPart *apart;

   apart = nalu_hypre_ParCSRMatrixAssumedPartition(matrix);

   if (apart->storage_length > 0)
   {
      nalu_hypre_TFree(apart->proc_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->row_start_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->row_end_list, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(apart->sort_index, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(apart, NALU_HYPRE_MEMORY_HOST);

   return (0);
}
