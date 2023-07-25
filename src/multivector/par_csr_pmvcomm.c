/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "par_csr_pmvcomm.h"

#include "_nalu_hypre_parcsr_mv.h"

/*==========================================================================*/

nalu_hypre_ParCSRCommMultiHandle *
nalu_hypre_ParCSRCommMultiHandleCreate (NALU_HYPRE_Int                   job,
                                   nalu_hypre_ParCSRCommPkg *comm_pkg,
                                   void                *send_data,
                                   void                *recv_data,
                                   NALU_HYPRE_Int                 num_vecs )
{
   NALU_HYPRE_Int            num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int            num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm             comm      = nalu_hypre_ParCSRCommPkgComm(comm_pkg);

   nalu_hypre_ParCSRCommMultiHandle *comm_handle;
   NALU_HYPRE_Int                   num_requests;
   nalu_hypre_MPI_Request           *requests;

   NALU_HYPRE_Int                  i, j;
   NALU_HYPRE_Int                  my_id, num_procs;
   NALU_HYPRE_Int                  ip, vec_start, vec_len;

   /*--------------------------------------------------------------------
    * nalu_hypre_Initialize sets up a communication handle,
    * posts receives and initiates sends. It always requires num_sends,
    * num_recvs, recv_procs and send_procs to be set in comm_pkg.
    * There are different options for job:
    * job = 1 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a Matvec,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    * job = 2 : is used to initialize communication exchange for the parts
    *           of vector needed to perform a MatvecT,  it requires send_data
    *           and recv_data to be doubles, recv_vec_starts and
    *           send_map_starts need to be set in comm_pkg.
    *--------------------------------------------------------------------*/

   num_requests = num_sends + num_recvs;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   j = 0;
   switch (job)
   {
      case  1:
      {
         NALU_HYPRE_Complex *d_send_data = (NALU_HYPRE_Complex *) send_data;
         NALU_HYPRE_Complex *d_recv_data = (NALU_HYPRE_Complex *) recv_data;
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start * num_vecs], vec_len * num_vecs,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Isend(&d_send_data[vec_start * num_vecs], vec_len * num_vecs,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  2:
      {
         NALU_HYPRE_Complex *d_send_data = (NALU_HYPRE_Complex *) send_data;
         NALU_HYPRE_Complex *d_recv_data = (NALU_HYPRE_Complex *) recv_data;
         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start * num_vecs], vec_len * num_vecs,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&d_send_data[vec_start * num_vecs], vec_len * num_vecs,
                            NALU_HYPRE_MPI_COMPLEX, ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }

   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = nalu_hypre_CTAlloc(nalu_hypre_ParCSRCommMultiHandle,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRCommMultiHandleCommPkg(comm_handle)     = comm_pkg;
   nalu_hypre_ParCSRCommMultiHandleSendData(comm_handle)    = send_data;
   nalu_hypre_ParCSRCommMultiHandleRecvData(comm_handle)    = recv_data;
   nalu_hypre_ParCSRCommMultiHandleNumRequests(comm_handle) = num_requests;
   nalu_hypre_ParCSRCommMultiHandleRequests(comm_handle)    = requests;

   return (comm_handle);
}

NALU_HYPRE_Int
nalu_hypre_ParCSRCommMultiHandleDestroy(nalu_hypre_ParCSRCommMultiHandle *comm_handle)
{
   nalu_hypre_MPI_Status *status0;
   NALU_HYPRE_Int    ierr = 0;

   if (nalu_hypre_ParCSRCommMultiHandleNumRequests(comm_handle))
   {
      status0 = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,
                              nalu_hypre_ParCSRCommMultiHandleNumRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Waitall(nalu_hypre_ParCSRCommMultiHandleNumRequests(comm_handle),
                        nalu_hypre_ParCSRCommMultiHandleRequests(comm_handle), status0);
      nalu_hypre_TFree(status0, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(nalu_hypre_ParCSRCommMultiHandleRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_handle, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

