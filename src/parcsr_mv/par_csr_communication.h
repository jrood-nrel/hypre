/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
typedef enum CommPkgJobType
{
   NALU_HYPRE_COMM_PKG_JOB_COMPLEX = 0,
   NALU_HYPRE_COMM_PKG_JOB_COMPLEX_TRANSPOSE,
   NALU_HYPRE_COMM_PKG_JOB_INT,
   NALU_HYPRE_COMM_PKG_JOB_INT_TRANSPOSE,
   NALU_HYPRE_COMM_PKG_JOB_BIGINT,
   NALU_HYPRE_COMM_PKG_JOB_BIGINT_TRANSPOSE,
   NUM_OF_COMM_PKG_JOB_TYPE,
} CommPkgJobType;
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRCommHandle, nalu_hypre_ParCSRPersistentCommHandle
 *--------------------------------------------------------------------------*/
struct _nalu_hypre_ParCSRCommPkg;

typedef struct
{
   struct _nalu_hypre_ParCSRCommPkg *comm_pkg;
   NALU_HYPRE_MemoryLocation  send_memory_location;
   NALU_HYPRE_MemoryLocation  recv_memory_location;
   NALU_HYPRE_Int             num_send_bytes;
   NALU_HYPRE_Int             num_recv_bytes;
   void                 *send_data;
   void                 *recv_data;
   void                 *send_data_buffer;
   void                 *recv_data_buffer;
   NALU_HYPRE_Int             num_requests;
   nalu_hypre_MPI_Request    *requests;
} nalu_hypre_ParCSRCommHandle;

typedef nalu_hypre_ParCSRCommHandle nalu_hypre_ParCSRPersistentCommHandle;

typedef struct _nalu_hypre_ParCSRCommPkg
{
   MPI_Comm                          comm;
   NALU_HYPRE_Int                         num_components;
   NALU_HYPRE_Int                         num_sends;
   NALU_HYPRE_Int                        *send_procs;
   NALU_HYPRE_Int                        *send_map_starts;
   NALU_HYPRE_Int                        *send_map_elmts;
   NALU_HYPRE_Int                        *device_send_map_elmts;
   NALU_HYPRE_Int                         num_recvs;
   NALU_HYPRE_Int                        *recv_procs;
   NALU_HYPRE_Int                        *recv_vec_starts;
   /* remote communication information */
   nalu_hypre_MPI_Datatype               *send_mpi_types;
   nalu_hypre_MPI_Datatype               *recv_mpi_types;
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   nalu_hypre_ParCSRPersistentCommHandle *persistent_comm_handles[NUM_OF_COMM_PKG_JOB_TYPE];
#endif
#if defined(NALU_HYPRE_USING_GPU)
   /* temporary memory for matvec. cudaMalloc is expensive. alloc once and reuse */
   NALU_HYPRE_Complex                    *tmp_data;
   NALU_HYPRE_Complex                    *buf_data;
   nalu_hypre_CSRMatrix                  *matrix_E; /* for matvecT */
#endif
} nalu_hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRCommPkgComm(comm_pkg)                (comm_pkg -> comm)
#define nalu_hypre_ParCSRCommPkgNumComponents(comm_pkg)       (comm_pkg -> num_components)
#define nalu_hypre_ParCSRCommPkgNumSends(comm_pkg)            (comm_pkg -> num_sends)
#define nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg)           (comm_pkg -> send_procs)
#define nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i)         (comm_pkg -> send_procs[i])
#define nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg)       (comm_pkg -> send_map_starts)
#define nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)      (comm_pkg -> send_map_starts[i])
#define nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg)        (comm_pkg -> send_map_elmts)
#define nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg)  (comm_pkg -> device_send_map_elmts)
#define nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)       (comm_pkg -> send_map_elmts[i])
#define nalu_hypre_ParCSRCommPkgDeviceSendMapElmt(comm_pkg,i) (comm_pkg -> device_send_map_elmts[i])
#define nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg)            (comm_pkg -> num_recvs)
#define nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg)           (comm_pkg -> recv_procs)
#define nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i)         (comm_pkg -> recv_procs[i])
#define nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)       (comm_pkg -> recv_vec_starts)
#define nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)      (comm_pkg -> recv_vec_starts[i])
#define nalu_hypre_ParCSRCommPkgSendMPITypes(comm_pkg)        (comm_pkg -> send_mpi_types)
#define nalu_hypre_ParCSRCommPkgSendMPIType(comm_pkg,i)       (comm_pkg -> send_mpi_types[i])
#define nalu_hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)        (comm_pkg -> recv_mpi_types)
#define nalu_hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i)       (comm_pkg -> recv_mpi_types[i])

#if defined(NALU_HYPRE_USING_GPU)
#define nalu_hypre_ParCSRCommPkgTmpData(comm_pkg)             ((comm_pkg) -> tmp_data)
#define nalu_hypre_ParCSRCommPkgBufData(comm_pkg)             ((comm_pkg) -> buf_data)
#define nalu_hypre_ParCSRCommPkgMatrixE(comm_pkg)             ((comm_pkg) -> matrix_E)
#endif

static inline void
nalu_hypre_ParCSRCommPkgCopySendMapElmtsToDevice(nalu_hypre_ParCSRCommPkg *comm_pkg)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) == NULL)
   {
      nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) =
         nalu_hypre_TAlloc(NALU_HYPRE_Int,
                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_TMemcpy(nalu_hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                    nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg),
                    NALU_HYPRE_Int,
                    nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_HOST);
   }
#endif
}

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRCommHandleCommPkg(comm_handle)                (comm_handle -> comm_pkg)
#define nalu_hypre_ParCSRCommHandleSendMemoryLocation(comm_handle)     (comm_handle -> send_memory_location)
#define nalu_hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle)     (comm_handle -> recv_memory_location)
#define nalu_hypre_ParCSRCommHandleNumSendBytes(comm_handle)           (comm_handle -> num_send_bytes)
#define nalu_hypre_ParCSRCommHandleNumRecvBytes(comm_handle)           (comm_handle -> num_recv_bytes)
#define nalu_hypre_ParCSRCommHandleSendData(comm_handle)               (comm_handle -> send_data)
#define nalu_hypre_ParCSRCommHandleRecvData(comm_handle)               (comm_handle -> recv_data)
#define nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle)         (comm_handle -> send_data_buffer)
#define nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle)         (comm_handle -> recv_data_buffer)
#define nalu_hypre_ParCSRCommHandleNumRequests(comm_handle)            (comm_handle -> num_requests)
#define nalu_hypre_ParCSRCommHandleRequests(comm_handle)               (comm_handle -> requests)
#define nalu_hypre_ParCSRCommHandleRequest(comm_handle, i)             (comm_handle -> requests[i])

#endif /* NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER */
