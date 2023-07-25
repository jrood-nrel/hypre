/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*==========================================================================*/

#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
static CommPkgJobType getJobTypeOf(NALU_HYPRE_Int job)
{
   CommPkgJobType job_type = NALU_HYPRE_COMM_PKG_JOB_COMPLEX;
   switch (job)
   {
      case  1:
         job_type = NALU_HYPRE_COMM_PKG_JOB_COMPLEX;
         break;
      case  2:
         job_type = NALU_HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE;
         break;
      case  11:
         job_type = NALU_HYPRE_COMM_PKG_JOB_INT;
         break;
      case  12:
         job_type = NALU_HYPRE_COMM_PKG_JOB_INT_TRANSPOSE;
         break;
      case  21:
         job_type = NALU_HYPRE_COMM_PKG_JOB_BIGINT;
         break;
      case  22:
         job_type = NALU_HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE;
         break;
   } // switch (job)

   return job_type;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRPersistentCommHandleCreate
 *
 * When send_data and recv_data are NULL, buffers are internally
 * allocated and CommHandle owns the buffer
 *------------------------------------------------------------------*/

nalu_hypre_ParCSRPersistentCommHandle*
nalu_hypre_ParCSRPersistentCommHandleCreate( NALU_HYPRE_Int job, nalu_hypre_ParCSRCommPkg *comm_pkg )
{
   NALU_HYPRE_Int i;
   size_t num_bytes_send, num_bytes_recv;

   nalu_hypre_ParCSRPersistentCommHandle *comm_handle = nalu_hypre_CTAlloc(nalu_hypre_ParCSRPersistentCommHandle, 1,
                                                                 NALU_HYPRE_MEMORY_HOST);

   CommPkgJobType job_type = getJobTypeOf(job);

   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm  comm      = nalu_hypre_ParCSRCommPkgComm(comm_pkg);

   NALU_HYPRE_Int num_requests = num_sends + num_recvs;
   nalu_hypre_MPI_Request *requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_requests, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRCommHandleNumRequests(comm_handle) = num_requests;
   nalu_hypre_ParCSRCommHandleRequests(comm_handle)    = requests;

   void *send_buff = NULL, *recv_buff = NULL;

   switch (job_type)
   {
      case NALU_HYPRE_COMM_PKG_JOB_COMPLEX:
         num_bytes_send = sizeof(NALU_HYPRE_Complex) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_recv = sizeof(NALU_HYPRE_Complex) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_Complex *)recv_buff + vec_start, vec_len, NALU_HYPRE_MPI_COMPLEX,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Send_init( (NALU_HYPRE_Complex *)send_buff + vec_start, vec_len, NALU_HYPRE_MPI_COMPLEX,
                                 ip, 0, comm, requests + num_recvs + i );
         }
         break;

      case NALU_HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE:
         num_bytes_recv = sizeof(NALU_HYPRE_Complex) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_send = sizeof(NALU_HYPRE_Complex) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_Complex *)recv_buff + vec_start, vec_len, NALU_HYPRE_MPI_COMPLEX,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Send_init( (NALU_HYPRE_Complex *)send_buff + vec_start, vec_len, NALU_HYPRE_MPI_COMPLEX,
                                 ip, 0, comm, requests + num_sends + i );
         }
         break;

      case NALU_HYPRE_COMM_PKG_JOB_INT:
         num_bytes_send = sizeof(NALU_HYPRE_Int) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_recv = sizeof(NALU_HYPRE_Int) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_Int *)recv_buff + vec_start, vec_len, NALU_HYPRE_MPI_INT,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Send_init( (NALU_HYPRE_Int *)send_buff + vec_start, vec_len, NALU_HYPRE_MPI_INT,
                                 ip, 0, comm, requests + num_recvs + i );
         }
         break;

      case NALU_HYPRE_COMM_PKG_JOB_INT_TRANSPOSE:
         num_bytes_recv = sizeof(NALU_HYPRE_Int) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_send = sizeof(NALU_HYPRE_Int) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_Int *)recv_buff + vec_start, vec_len, NALU_HYPRE_MPI_INT,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Send_init( (NALU_HYPRE_Int *)send_buff + vec_start, vec_len, NALU_HYPRE_MPI_INT,
                                 ip, 0, comm, requests + num_sends + i );
         }
         break;

      case NALU_HYPRE_COMM_PKG_JOB_BIGINT:
         num_bytes_send = sizeof(NALU_HYPRE_BigInt) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_recv = sizeof(NALU_HYPRE_BigInt) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_BigInt *)recv_buff + (NALU_HYPRE_BigInt)vec_start, vec_len,
                                 NALU_HYPRE_MPI_BIG_INT,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Send_init( (NALU_HYPRE_BigInt *)send_buff + (NALU_HYPRE_BigInt)vec_start, vec_len,
                                 NALU_HYPRE_MPI_BIG_INT,
                                 ip, 0, comm, requests + num_recvs + i);
         }
         break;

      case NALU_HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE:
         num_bytes_recv = sizeof(NALU_HYPRE_BigInt) * nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         num_bytes_send = sizeof(NALU_HYPRE_BigInt) * nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
         recv_buff = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                  NALU_HYPRE_MEMORY_HOST);
         send_buff = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs),
                                  NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Recv_init( (NALU_HYPRE_BigInt *)recv_buff + (NALU_HYPRE_BigInt)vec_start, vec_len,
                                 NALU_HYPRE_MPI_BIG_INT,
                                 ip, 0, comm, requests + i );
         }
         for (i = 0; i < num_recvs; ++i)
         {
            NALU_HYPRE_Int ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            NALU_HYPRE_Int vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            NALU_HYPRE_Int vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;

            nalu_hypre_MPI_Send_init( (NALU_HYPRE_BigInt *)send_buff + (NALU_HYPRE_BigInt)vec_start, vec_len,
                                 NALU_HYPRE_MPI_BIG_INT,
                                 ip, 0, comm, requests + num_sends + i);
         }
         break;
      default:
         nalu_hypre_assert(1 == 0);
         break;
   } // switch (job_type)

   nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle) = recv_buff;
   nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle) = send_buff;
   nalu_hypre_ParCSRCommHandleNumSendBytes(comm_handle)   = num_bytes_send;
   nalu_hypre_ParCSRCommHandleNumRecvBytes(comm_handle)   = num_bytes_recv;

   return ( comm_handle );
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkgGetPersistentCommHandle
 *------------------------------------------------------------------*/

nalu_hypre_ParCSRPersistentCommHandle*
nalu_hypre_ParCSRCommPkgGetPersistentCommHandle( NALU_HYPRE_Int job, nalu_hypre_ParCSRCommPkg *comm_pkg )
{
   CommPkgJobType type = getJobTypeOf(job);
   if (!comm_pkg->persistent_comm_handles[type])
   {
      /* data is owned by persistent comm handle */
      comm_pkg->persistent_comm_handles[type] =
         nalu_hypre_ParCSRPersistentCommHandleCreate(job, comm_pkg);
   }

   return comm_pkg->persistent_comm_handles[type];
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRPersistentCommHandleDestroy
 *------------------------------------------------------------------*/

void
nalu_hypre_ParCSRPersistentCommHandleDestroy( nalu_hypre_ParCSRPersistentCommHandle *comm_handle )
{
   if (comm_handle)
   {
      nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(comm_handle->requests, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(comm_handle, NALU_HYPRE_MEMORY_HOST);
   }
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRPersistentCommHandleStart
 *------------------------------------------------------------------*/

void
nalu_hypre_ParCSRPersistentCommHandleStart( nalu_hypre_ParCSRPersistentCommHandle *comm_handle,
                                       NALU_HYPRE_MemoryLocation              send_memory_location,
                                       void                             *send_data )
{
   nalu_hypre_ParCSRCommHandleSendData(comm_handle) = send_data;
   nalu_hypre_ParCSRCommHandleSendMemoryLocation(comm_handle) = send_memory_location;

   if (nalu_hypre_ParCSRCommHandleNumRequests(comm_handle) > 0)
   {
      nalu_hypre_TMemcpy( nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle),
                     send_data,
                     char,
                     nalu_hypre_ParCSRCommHandleNumSendBytes(comm_handle),
                     NALU_HYPRE_MEMORY_HOST,
                     send_memory_location );

      NALU_HYPRE_Int ret = nalu_hypre_MPI_Startall(nalu_hypre_ParCSRCommHandleNumRequests(comm_handle),
                                         nalu_hypre_ParCSRCommHandleRequests(comm_handle));
      if (nalu_hypre_MPI_SUCCESS != ret)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "MPI error\n");
         /*nalu_hypre_printf("MPI error %d in %s (%s, line %u)\n", ret, __FUNCTION__, __FILE__, __LINE__);*/
      }
   }
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRPersistentCommHandleWait
 *------------------------------------------------------------------*/

void
nalu_hypre_ParCSRPersistentCommHandleWait( nalu_hypre_ParCSRPersistentCommHandle *comm_handle,
                                      NALU_HYPRE_MemoryLocation              recv_memory_location,
                                      void                             *recv_data )
{
   nalu_hypre_ParCSRCommHandleRecvData(comm_handle) = recv_data;
   nalu_hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle) = recv_memory_location;

   if (nalu_hypre_ParCSRCommHandleNumRequests(comm_handle) > 0)
   {
      NALU_HYPRE_Int ret = nalu_hypre_MPI_Waitall(nalu_hypre_ParCSRCommHandleNumRequests(comm_handle),
                                        nalu_hypre_ParCSRCommHandleRequests(comm_handle),
                                        nalu_hypre_MPI_STATUSES_IGNORE);
      if (nalu_hypre_MPI_SUCCESS != ret)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "MPI error\n");
         /*nalu_hypre_printf("MPI error %d in %s (%s, line %u)\n", ret, __FUNCTION__, __FILE__, __LINE__);*/
      }

      nalu_hypre_TMemcpy(recv_data,
                    nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle),
                    char,
                    nalu_hypre_ParCSRCommHandleNumRecvBytes(comm_handle),
                    recv_memory_location,
                    NALU_HYPRE_MEMORY_HOST);
   }
}
#endif // NALU_HYPRE_USING_PERSISTENT_COMM

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommHandleCreate
 *------------------------------------------------------------------*/

nalu_hypre_ParCSRCommHandle*
nalu_hypre_ParCSRCommHandleCreate ( NALU_HYPRE_Int            job,
                               nalu_hypre_ParCSRCommPkg *comm_pkg,
                               void                *send_data,
                               void                *recv_data )
{
   return nalu_hypre_ParCSRCommHandleCreate_v2(job, comm_pkg, NALU_HYPRE_MEMORY_HOST, send_data,
                                          NALU_HYPRE_MEMORY_HOST, recv_data);
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommHandleCreate_v2
 *------------------------------------------------------------------*/

nalu_hypre_ParCSRCommHandle*
nalu_hypre_ParCSRCommHandleCreate_v2 ( NALU_HYPRE_Int            job,
                                  nalu_hypre_ParCSRCommPkg *comm_pkg,
                                  NALU_HYPRE_MemoryLocation send_memory_location,
                                  void                *send_data_in,
                                  NALU_HYPRE_MemoryLocation recv_memory_location,
                                  void                *recv_data_in )
{
   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ParCSRCommHandleCreate_v2");

   NALU_HYPRE_Int                  num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int                  num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   MPI_Comm                   comm      = nalu_hypre_ParCSRCommPkgComm(comm_pkg);
   NALU_HYPRE_Int                  num_send_bytes = 0;
   NALU_HYPRE_Int                  num_recv_bytes = 0;
   nalu_hypre_ParCSRCommHandle    *comm_handle;
   NALU_HYPRE_Int                  num_requests;
   nalu_hypre_MPI_Request         *requests;
   NALU_HYPRE_Int                  i, j;
   NALU_HYPRE_Int                  my_id, num_procs;
   NALU_HYPRE_Int                  ip, vec_start, vec_len;
   void                      *send_data;
   void                      *recv_data;

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
    * job = 11: similar to job = 1, but exchanges data of type NALU_HYPRE_Int (not NALU_HYPRE_Complex),
    *           requires send_data and recv_data to be ints
    *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * job = 12: similar to job = 2, but exchanges data of type NALU_HYPRE_Int (not NALU_HYPRE_Complex),
    *           requires send_data and recv_data to be ints
    *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * job = 21: similar to job = 1, but exchanges data of type NALU_HYPRE_BigInt (not NALU_HYPRE_Complex),
    *           requires send_data and recv_data to be ints
    *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * job = 22: similar to job = 2, but exchanges data of type NALU_HYPRE_BigInt (not NALU_HYPRE_Complex),
    *           requires send_data and recv_data to be ints
    *           recv_vec_starts and send_map_starts need to be set in comm_pkg.
    * default: ignores send_data and recv_data, requires send_mpi_types
    *           and recv_mpi_types to be set in comm_pkg.
    *           datatypes need to point to absolute
    *           addresses, e.g. generated using nalu_hypre_MPI_Address .
    *--------------------------------------------------------------------*/
#ifndef NALU_HYPRE_WITH_GPU_AWARE_MPI
   switch (job)
   {
      case 1:
         num_send_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_Complex);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_Complex);
         break;
      case 2:
         num_send_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_Complex);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_Complex);
         break;
      case 11:
         num_send_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_Int);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_Int);
         break;
      case 12:
         num_send_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_Int);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_Int);
         break;
      case 21:
         num_send_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_BigInt);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_BigInt);
         break;
      case 22:
         num_send_bytes = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs) * sizeof(NALU_HYPRE_BigInt);
         num_recv_bytes = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * sizeof(NALU_HYPRE_BigInt);
         break;
   }

   nalu_hypre_MemoryLocation act_send_memory_location = nalu_hypre_GetActualMemLocation(send_memory_location);

   if ( act_send_memory_location == nalu_hypre_MEMORY_DEVICE ||
        act_send_memory_location == nalu_hypre_MEMORY_UNIFIED )
   {
      //send_data = _nalu_hypre_TAlloc(char, num_send_bytes, nalu_hypre_MEMORY_HOST_PINNED);
      send_data = nalu_hypre_TAlloc(char, num_send_bytes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_GpuProfilingPushRange("MPI-D2H");
      nalu_hypre_TMemcpy(send_data, send_data_in, char, num_send_bytes, NALU_HYPRE_MEMORY_HOST,
                    NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_GpuProfilingPopRange();
   }
   else
   {
      send_data = send_data_in;
   }

   nalu_hypre_MemoryLocation act_recv_memory_location = nalu_hypre_GetActualMemLocation(recv_memory_location);

   if ( act_recv_memory_location == nalu_hypre_MEMORY_DEVICE ||
        act_recv_memory_location == nalu_hypre_MEMORY_UNIFIED )
   {
      //recv_data = nalu_hypre_TAlloc(char, num_recv_bytes, nalu_hypre_MEMORY_HOST_PINNED);
      recv_data = nalu_hypre_TAlloc(char, num_recv_bytes, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      recv_data = recv_data_in;
   }
#else /* #ifndef NALU_HYPRE_WITH_GPU_AWARE_MPI */
   send_data = send_data_in;
   recv_data = recv_data_in;
#endif

   num_requests = num_sends + num_recvs;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_requests, NALU_HYPRE_MEMORY_HOST);

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
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&d_send_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  2:
      {
         NALU_HYPRE_Complex *d_send_data = (NALU_HYPRE_Complex *) send_data;
         NALU_HYPRE_Complex *d_recv_data = (NALU_HYPRE_Complex *) recv_data;
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Irecv(&d_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&d_send_data[vec_start], vec_len, NALU_HYPRE_MPI_COMPLEX,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  11:
      {
         NALU_HYPRE_Int *i_send_data = (NALU_HYPRE_Int *) send_data;
         NALU_HYPRE_Int *i_recv_data = (NALU_HYPRE_Int *) recv_data;
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Irecv(&i_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_INT,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&i_send_data[vec_start], vec_len, NALU_HYPRE_MPI_INT,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  12:
      {
         NALU_HYPRE_Int *i_send_data = (NALU_HYPRE_Int *) send_data;
         NALU_HYPRE_Int *i_recv_data = (NALU_HYPRE_Int *) recv_data;
         for (i = 0; i < num_sends; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Irecv(&i_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_INT,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&i_send_data[vec_start], vec_len, NALU_HYPRE_MPI_INT,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  21:
      {
         NALU_HYPRE_BigInt *i_send_data = (NALU_HYPRE_BigInt *) send_data;
         NALU_HYPRE_BigInt *i_recv_data = (NALU_HYPRE_BigInt *) recv_data;
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Irecv(&i_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Isend(&i_send_data[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
      case  22:
      {
         NALU_HYPRE_BigInt *i_send_data = (NALU_HYPRE_BigInt *) send_data;
         NALU_HYPRE_BigInt *i_recv_data = (NALU_HYPRE_BigInt *) recv_data;
         for (i = 0; i < num_sends; i++)
         {
            vec_start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1) - vec_start;
            ip = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
            nalu_hypre_MPI_Irecv(&i_recv_data[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                            ip, 0, comm, &requests[j++]);
         }
         for (i = 0; i < num_recvs; i++)
         {
            ip = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
            vec_start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
            vec_len = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i + 1) - vec_start;
            nalu_hypre_MPI_Isend(&i_send_data[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                            ip, 0, comm, &requests[j++]);
         }
         break;
      }
   }
   /*--------------------------------------------------------------------
    * set up comm_handle and return
    *--------------------------------------------------------------------*/

   comm_handle = nalu_hypre_CTAlloc(nalu_hypre_ParCSRCommHandle,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRCommHandleCommPkg(comm_handle)            = comm_pkg;
   nalu_hypre_ParCSRCommHandleSendMemoryLocation(comm_handle) = send_memory_location;
   nalu_hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle) = recv_memory_location;
   nalu_hypre_ParCSRCommHandleNumSendBytes(comm_handle)       = num_send_bytes;
   nalu_hypre_ParCSRCommHandleNumRecvBytes(comm_handle)       = num_recv_bytes;
   nalu_hypre_ParCSRCommHandleSendData(comm_handle)           = send_data_in;
   nalu_hypre_ParCSRCommHandleRecvData(comm_handle)           = recv_data_in;
   nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle)     = send_data;
   nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle)     = recv_data;
   nalu_hypre_ParCSRCommHandleNumRequests(comm_handle)        = num_requests;
   nalu_hypre_ParCSRCommHandleRequests(comm_handle)           = requests;

   nalu_hypre_GpuProfilingPopRange();

   return ( comm_handle );
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommHandleDestroy
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRCommHandleDestroy( nalu_hypre_ParCSRCommHandle *comm_handle )
{
   if ( comm_handle == NULL )
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ParCSRCommHandleDestroy");

   if (nalu_hypre_ParCSRCommHandleNumRequests(comm_handle))
   {
      nalu_hypre_MPI_Status *status0;
      status0 = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,
                              nalu_hypre_ParCSRCommHandleNumRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_GpuProfilingPushRange("nalu_hypre_MPI_Waitall");
      nalu_hypre_MPI_Waitall(nalu_hypre_ParCSRCommHandleNumRequests(comm_handle),
                        nalu_hypre_ParCSRCommHandleRequests(comm_handle), status0);
      nalu_hypre_GpuProfilingPopRange();
      nalu_hypre_TFree(status0, NALU_HYPRE_MEMORY_HOST);
   }

#ifndef NALU_HYPRE_WITH_GPU_AWARE_MPI
   nalu_hypre_MemoryLocation act_send_memory_location = nalu_hypre_GetActualMemLocation(
                                                      nalu_hypre_ParCSRCommHandleSendMemoryLocation(comm_handle));
   if ( act_send_memory_location == nalu_hypre_MEMORY_DEVICE ||
        act_send_memory_location == nalu_hypre_MEMORY_UNIFIED )
   {
      //nalu_hypre_HostPinnedFree(nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle));
      nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle), NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_MemoryLocation act_recv_memory_location = nalu_hypre_GetActualMemLocation(
                                                      nalu_hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle));
   if ( act_recv_memory_location == nalu_hypre_MEMORY_DEVICE ||
        act_recv_memory_location == nalu_hypre_MEMORY_UNIFIED )
   {
      nalu_hypre_GpuProfilingPushRange("MPI-H2D");
      nalu_hypre_TMemcpy( nalu_hypre_ParCSRCommHandleRecvData(comm_handle),
                     nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle),
                     char,
                     nalu_hypre_ParCSRCommHandleNumRecvBytes(comm_handle),
                     NALU_HYPRE_MEMORY_DEVICE,
                     NALU_HYPRE_MEMORY_HOST );
      nalu_hypre_GpuProfilingPopRange();
      //nalu_hypre_HostPinnedFree(nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle));
      nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle), NALU_HYPRE_MEMORY_HOST);
   }
#endif

   nalu_hypre_TFree(nalu_hypre_ParCSRCommHandleRequests(comm_handle), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_handle, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkgCreate_core
 *
 * This function does all the communications and computations for
 * nalu_hypre_ParCSRCommPkgCreate(nalu_hypre_ParCSRMatrix *A) and
 * nalu_hypre_BooleanMatvecCommPkgCreate(nalu_hypre_ParCSRBooleanMatrix *A)
 *
 * To support both data types, it has hardly any data structures
 * other than NALU_HYPRE_Int*.
 *------------------------------------------------------------------*/

void
nalu_hypre_ParCSRCommPkgCreate_core(
   /* input args: */
   MPI_Comm   comm,
   NALU_HYPRE_BigInt *col_map_offd,
   NALU_HYPRE_BigInt  first_col_diag,
   NALU_HYPRE_BigInt *col_starts,
   NALU_HYPRE_Int  num_cols_diag,
   NALU_HYPRE_Int  num_cols_offd,
   /* pointers to output args: */
   NALU_HYPRE_Int  *p_num_recvs,
   NALU_HYPRE_Int **p_recv_procs,
   NALU_HYPRE_Int **p_recv_vec_starts,
   NALU_HYPRE_Int  *p_num_sends,
   NALU_HYPRE_Int **p_send_procs,
   NALU_HYPRE_Int **p_send_map_starts,
   NALU_HYPRE_Int **p_send_map_elmts
)
{
   NALU_HYPRE_Int    i, j;
   NALU_HYPRE_Int    num_procs, my_id, proc_num, num_elmts;
   NALU_HYPRE_Int    local_info;
   NALU_HYPRE_BigInt offd_col;
   NALU_HYPRE_BigInt *big_buf_data = NULL;
   NALU_HYPRE_Int    *proc_mark, *proc_add, *tmp, *recv_buf, *displs, *info;
   /* outputs: */
   NALU_HYPRE_Int  num_recvs, *recv_procs, *recv_vec_starts;
   NALU_HYPRE_Int  num_sends, *send_procs, *send_map_starts, *send_map_elmts;
   NALU_HYPRE_Int  ip, vec_start, vec_len, num_requests;

   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status *status;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   proc_mark = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   proc_add = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);

   /* ----------------------------------------------------------------------
    * determine which processors to receive from (set proc_mark) and num_recvs,
    * at the end of the loop proc_mark[i] contains the number of elements to be
    * received from Proc. i
    * ---------------------------------------------------------------------*/

   proc_num = 0;
   if (num_cols_offd)
   {
      offd_col = col_map_offd[0];
   }

   num_recvs = 0;
   for (i = 0; i < num_cols_offd; i++)
   {
      if (num_cols_diag)
      {
         proc_num = nalu_hypre_min(num_procs - 1, (NALU_HYPRE_Int)(offd_col / (NALU_HYPRE_BigInt)num_cols_diag));
      }

      while (col_starts[proc_num] > offd_col )
      {
         proc_num = proc_num - 1;
      }

      while (col_starts[proc_num + 1] - 1 < offd_col )
      {
         proc_num = proc_num + 1;
      }

      proc_mark[num_recvs] = proc_num;
      j = i;
      while (col_starts[proc_num + 1] > offd_col)
      {
         proc_add[num_recvs]++;
         if (j < num_cols_offd - 1)
         {
            j++;
            offd_col = col_map_offd[j];
         }
         else
         {
            j++;
            offd_col = col_starts[num_procs];
         }
      }
      num_recvs++;

      i = (j < num_cols_offd) ? (j - 1) : j;
   }

   local_info = 2 * num_recvs;

   nalu_hypre_MPI_Allgather(&local_info, 1, NALU_HYPRE_MPI_INT, info, 1, NALU_HYPRE_MPI_INT, comm);

   /* ----------------------------------------------------------------------
    * generate information to be sent: tmp contains for each recv_proc:
    * id of recv_procs, number of elements to be received for this processor,
    * indices of elements (in this order)
    * ---------------------------------------------------------------------*/

   displs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < num_procs + 1; i++)
   {
      displs[i] = displs[i - 1] + info[i - 1];
   }
   recv_buf = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  displs[num_procs], NALU_HYPRE_MEMORY_HOST);

   recv_procs = NULL;
   tmp = NULL;
   if (num_recvs)
   {
      recv_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs, NALU_HYPRE_MEMORY_HOST);
      tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_info, NALU_HYPRE_MEMORY_HOST);
   }
   recv_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_recvs; i++)
   {
      num_elmts = proc_add[i];
      recv_procs[i] = proc_mark[i];
      recv_vec_starts[i + 1] = recv_vec_starts[i] + num_elmts;
      tmp[j++] = proc_mark[i];
      tmp[j++] = num_elmts;
   }

   nalu_hypre_MPI_Allgatherv(tmp, local_info, NALU_HYPRE_MPI_INT, recv_buf, info,
                        displs, NALU_HYPRE_MPI_INT, comm);

   /* ----------------------------------------------------------------------
    * determine num_sends and number of elements to be sent
    * ---------------------------------------------------------------------*/

   num_sends = 0;
   num_elmts = 0;
   proc_add[0] = 0;
   for (i = 0; i < num_procs; i++)
   {
      j = displs[i];
      while ( j < displs[i + 1])
      {
         if (recv_buf[j++] == my_id)
         {
            proc_mark[num_sends] = i;
            num_sends++;
            proc_add[num_sends] = proc_add[num_sends - 1] + recv_buf[j];
            break;
         }
         j++;
      }
   }

   /* ----------------------------------------------------------------------
    * determine send_procs and actual elements to be send (in send_map_elmts)
    * and send_map_starts whose i-th entry points to the beginning of the
    * elements to be send to proc. i
    * ---------------------------------------------------------------------*/

   send_procs = NULL;
   send_map_elmts = NULL;

   if (num_sends)
   {
      send_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends, NALU_HYPRE_MEMORY_HOST);
      send_map_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  proc_add[num_sends], NALU_HYPRE_MEMORY_HOST);
      big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  proc_add[num_sends], NALU_HYPRE_MEMORY_HOST);
   }
   send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   num_requests = num_recvs + num_sends;
   if (num_requests)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_requests, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_sends; i++)
   {
      send_map_starts[i + 1] = proc_add[i + 1];
      send_procs[i] = proc_mark[i];
   }

   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i + 1] - vec_start;
      ip = send_procs[i];
      nalu_hypre_MPI_Irecv(&big_buf_data[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      ip, 0, comm, &requests[j++]);
   }
   for (i = 0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i + 1] - vec_start;
      ip = recv_procs[i];
      nalu_hypre_MPI_Isend(&col_map_offd[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      ip, 0, comm, &requests[j++]);
   }

   if (num_requests)
   {
      nalu_hypre_MPI_Waitall(num_requests, requests, status);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_sends)
   {
      for (i = 0; i < send_map_starts[num_sends]; i++)
      {
         send_map_elmts[i] = (NALU_HYPRE_Int)(big_buf_data[i] - first_col_diag);
      }
   }

   nalu_hypre_TFree(proc_add, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(proc_mark, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(info, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   /* finish up with the hand-coded call-by-reference... */
   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_procs;
   *p_send_map_starts = send_map_starts;
   *p_send_map_elmts = send_map_elmts;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkgCreate
 *
 * Creates the communication package with MPI collectives calls.
 *
 * Notes:
 *    1) This version does not use the assumed partition.
 *    2) comm_pkg must be allocated outside of this function
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRCommPkgCreate( MPI_Comm             comm,
                           NALU_HYPRE_BigInt        *col_map_offd,
                           NALU_HYPRE_BigInt         first_col_diag,
                           NALU_HYPRE_BigInt        *col_starts,
                           NALU_HYPRE_Int            num_cols_diag,
                           NALU_HYPRE_Int            num_cols_offd,
                           nalu_hypre_ParCSRCommPkg *comm_pkg )
{
   NALU_HYPRE_Int  num_sends;
   NALU_HYPRE_Int *send_procs;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;

   NALU_HYPRE_Int  num_recvs;
   NALU_HYPRE_Int *recv_procs;
   NALU_HYPRE_Int *recv_vec_starts;

   nalu_hypre_ParCSRCommPkgCreate_core(comm, col_map_offd, first_col_diag,
                                  col_starts, num_cols_diag, num_cols_offd,
                                  &num_recvs, &recv_procs, &recv_vec_starts,
                                  &num_sends, &send_procs, &send_map_starts,
                                  &send_map_elmts);

   /* Fill the communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkgCreateAndFill
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRCommPkgCreateAndFill( MPI_Comm              comm,
                                  NALU_HYPRE_Int             num_recvs,
                                  NALU_HYPRE_Int            *recv_procs,
                                  NALU_HYPRE_Int            *recv_vec_starts,
                                  NALU_HYPRE_Int             num_sends,
                                  NALU_HYPRE_Int            *send_procs,
                                  NALU_HYPRE_Int            *send_map_starts,
                                  NALU_HYPRE_Int            *send_map_elmts,
                                  nalu_hypre_ParCSRCommPkg **comm_pkg_ptr )
{
   nalu_hypre_ParCSRCommPkg  *comm_pkg;

   /* Allocate memory for comm_pkg if needed */
   if (*comm_pkg_ptr == NULL)
   {
      comm_pkg = nalu_hypre_TAlloc(nalu_hypre_ParCSRCommPkg, 1, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      comm_pkg = *comm_pkg_ptr;
   }

   /* Set default info */
   nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg)      = 1;
   nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) = NULL;
#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   nalu_hypre_ParCSRCommPkgTmpData(comm_pkg)            = NULL;
   nalu_hypre_ParCSRCommPkgBufData(comm_pkg)            = NULL;
   nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg)            = NULL;
#endif
#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   NALU_HYPRE_Int i;

   for (i = 0; i < NUM_OF_COMM_PKG_JOB_TYPE; i++)
   {
      comm_pkg->persistent_comm_handles[i] = NULL;
   }
#endif

   /* Set input info */
   nalu_hypre_ParCSRCommPkgComm(comm_pkg)          = comm;
   nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg)      = num_recvs;
   nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg)     = recv_procs;
   nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   nalu_hypre_ParCSRCommPkgNumSends(comm_pkg)      = num_sends;
   nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg)     = send_procs;
   nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  = send_map_elmts;

   /* Set output pointer */
   *comm_pkg_ptr = comm_pkg;

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkgUpdateVecStarts
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRCommPkgUpdateVecStarts( nalu_hypre_ParCSRCommPkg *comm_pkg,
                                    nalu_hypre_ParVector     *x )
{
   nalu_hypre_Vector *x_local         = nalu_hypre_ParVectorLocalVector(x);
   NALU_HYPRE_Int     num_vectors     = nalu_hypre_VectorNumVectors(x_local);
   NALU_HYPRE_Int     vecstride       = nalu_hypre_VectorVectorStride(x_local);
   NALU_HYPRE_Int     idxstride       = nalu_hypre_VectorIndexStride(x_local);

   NALU_HYPRE_Int     num_components  = nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg);
   NALU_HYPRE_Int     num_sends       = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int     num_recvs       = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   NALU_HYPRE_Int    *recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   NALU_HYPRE_Int    *send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   NALU_HYPRE_Int    *send_map_elmts  = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   NALU_HYPRE_Int    *send_map_elmts_new;

   NALU_HYPRE_Int     i, j;

   nalu_hypre_assert(num_components > 0);

   if (num_vectors != num_components)
   {
      /* Update number of components in the communication package */
      nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg) = num_vectors;

      /* Allocate send_maps_elmts */
      send_map_elmts_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                         send_map_starts[num_sends] * num_vectors,
                                         NALU_HYPRE_MEMORY_HOST);

      /* Update send_maps_elmts */
      if (num_vectors > num_components)
      {
         if (num_components == 1)
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_vectors; j++)
               {
                  send_map_elmts_new[i * num_vectors + j] = send_map_elmts[i] * idxstride +
                                                            j * vecstride;
               }
            }
         }
         else
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_vectors; j++)
               {
                  send_map_elmts_new[i * num_vectors + j] =
                     send_map_elmts[i * num_components] * idxstride + j * vecstride;
               }
            }
         }
      }
      else
      {
         /* num_vectors < num_components */
         if (num_vectors == 1)
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               send_map_elmts_new[i] = send_map_elmts[i * num_components];
            }
         }
         else
         {
            for (i = 0; i < send_map_starts[num_sends]; i++)
            {
               for (j = 0; j < num_vectors; j++)
               {
                  send_map_elmts_new[i * num_vectors + j] =
                     send_map_elmts[i * num_components + j];
               }
            }
         }
      }
      nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elmts_new;

      /* Free memory */
      nalu_hypre_TFree(send_map_elmts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg), NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg));
      nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg) = NULL;
#endif

      /* Update send_map_starts */
      for (i = 0; i < num_sends + 1; i++)
      {
         send_map_starts[i] *= num_vectors / num_components;
      }

      /* Update recv_vec_starts */
      for (i = 0; i < num_recvs + 1; i++)
      {
         recv_vec_starts[i] *= num_vectors / num_components;
      }
   }

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_MatvecCommPkgCreate
 *
 * Generates the communication package for A using assumed partition
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MatvecCommPkgCreate ( nalu_hypre_ParCSRMatrix *A )
{
   MPI_Comm             comm  = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_IJAssumedPart *apart = nalu_hypre_ParCSRMatrixAssumedPartition(A);
   nalu_hypre_ParCSRCommPkg *comm_pkg;

   NALU_HYPRE_BigInt         first_col_diag  = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt        *col_map_offd    = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int            num_cols_offd   = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   NALU_HYPRE_BigInt         global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(A);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Create the assumed partition and should own it */
   if (apart == NULL)
   {
      nalu_hypre_ParCSRMatrixCreateAssumedPartition(A);
      nalu_hypre_ParCSRMatrixOwnsAssumedPartition(A) = 1;
      apart = nalu_hypre_ParCSRMatrixAssumedPartition(A);
   }

   /*-----------------------------------------------------------
    * setup commpkg
    *----------------------------------------------------------*/

   comm_pkg = nalu_hypre_TAlloc(nalu_hypre_ParCSRCommPkg, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixCommPkg(A) = comm_pkg;
   nalu_hypre_ParCSRCommPkgCreateApart( comm, col_map_offd, first_col_diag,
                                   num_cols_offd, global_num_cols,
                                   apart,
                                   comm_pkg );

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_MatvecCommPkgDestroy
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MatvecCommPkgDestroy( nalu_hypre_ParCSRCommPkg *comm_pkg )
{
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   NALU_HYPRE_Int i;
   for (i = NALU_HYPRE_COMM_PKG_JOB_COMPLEX; i < NUM_OF_COMM_PKG_JOB_TYPE; ++i)
   {
      if (comm_pkg->persistent_comm_handles[i])
      {
         nalu_hypre_ParCSRPersistentCommHandleDestroy(comm_pkg->persistent_comm_handles[i]);
      }
   }
#endif

   if (nalu_hypre_ParCSRCommPkgNumSends(comm_pkg))
   {
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg), NALU_HYPRE_MEMORY_DEVICE);
   }
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg), NALU_HYPRE_MEMORY_HOST);
   /* if (nalu_hypre_ParCSRCommPkgSendMPITypes(comm_pkg))
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMPITypes(comm_pkg), NALU_HYPRE_MEMORY_HOST); */
   if (nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg))
   {
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg), NALU_HYPRE_MEMORY_HOST);
   /* if (nalu_hypre_ParCSRCommPkgRecvMPITypes(comm_pkg))
      nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvMPITypes(comm_pkg), NALU_HYPRE_MEMORY_HOST); */

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgTmpData(comm_pkg), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgBufData(comm_pkg), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg));
#endif

   nalu_hypre_TFree(comm_pkg, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_ParCSRFindExtendCommPkg
 *
 * AHB 11/06 : alternate to the extend function below - creates a
 * second comm pkg based on indices - this makes it easier to use the
 * global partition
 *
 * RL: renamed and moved it here
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRFindExtendCommPkg(MPI_Comm              comm,
                              NALU_HYPRE_BigInt          global_num,
                              NALU_HYPRE_BigInt          my_first,
                              NALU_HYPRE_Int             local_num,
                              NALU_HYPRE_BigInt         *starts,
                              nalu_hypre_IJAssumedPart  *apart,
                              NALU_HYPRE_Int             indices_len,
                              NALU_HYPRE_BigInt         *indices,
                              nalu_hypre_ParCSRCommPkg **extend_comm_pkg)
{
   nalu_hypre_ParCSRCommPkg *new_comm_pkg = nalu_hypre_TAlloc(nalu_hypre_ParCSRCommPkg, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_assert(apart != NULL);
   nalu_hypre_ParCSRCommPkgCreateApart(comm, indices, my_first, indices_len,
                                  global_num, apart, new_comm_pkg);

   *extend_comm_pkg = new_comm_pkg;

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_BuildCSRMatrixMPIDataType
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BuildCSRMatrixMPIDataType( NALU_HYPRE_Int num_nonzeros,
                                 NALU_HYPRE_Int num_rows,
                                 NALU_HYPRE_Complex *a_data,
                                 NALU_HYPRE_Int *a_i,
                                 NALU_HYPRE_Int *a_j,
                                 nalu_hypre_MPI_Datatype *csr_matrix_datatype )
{
   NALU_HYPRE_Int            block_lens[3];
   nalu_hypre_MPI_Aint       displ[3];
   nalu_hypre_MPI_Datatype   types[3];

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_rows + 1;
   block_lens[2] = num_nonzeros;

   types[0] = NALU_HYPRE_MPI_COMPLEX;
   types[1] = NALU_HYPRE_MPI_INT;
   types[2] = NALU_HYPRE_MPI_INT;

   nalu_hypre_MPI_Address(a_data, &displ[0]);
   nalu_hypre_MPI_Address(a_i, &displ[1]);
   nalu_hypre_MPI_Address(a_j, &displ[2]);
   nalu_hypre_MPI_Type_struct(3, block_lens, displ, types, csr_matrix_datatype);
   nalu_hypre_MPI_Type_commit(csr_matrix_datatype);

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------
 * nalu_hypre_BuildCSRMatrixMPIDataType
 *------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BuildCSRJDataType( NALU_HYPRE_Int num_nonzeros,
                         NALU_HYPRE_Complex *a_data,
                         NALU_HYPRE_Int *a_j,
                         nalu_hypre_MPI_Datatype *csr_jdata_datatype )
{
   NALU_HYPRE_Int          block_lens[2];
   nalu_hypre_MPI_Aint     displs[2];
   nalu_hypre_MPI_Datatype types[2];

   block_lens[0] = num_nonzeros;
   block_lens[1] = num_nonzeros;

   types[0] = NALU_HYPRE_MPI_COMPLEX;
   types[1] = NALU_HYPRE_MPI_INT;

   nalu_hypre_MPI_Address(a_data, &displs[0]);
   nalu_hypre_MPI_Address(a_j, &displs[1]);

   nalu_hypre_MPI_Type_struct(2, block_lens, displs, types, csr_jdata_datatype);
   nalu_hypre_MPI_Type_commit(csr_jdata_datatype);

   return nalu_hypre_error_flag;
}
