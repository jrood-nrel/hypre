/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
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
 * hypre_ParCSRCommHandle, hypre_ParCSRPersistentCommHandle
 *--------------------------------------------------------------------------*/
struct _hypre_ParCSRCommPkg;

typedef struct
{
   struct _hypre_ParCSRCommPkg *comm_pkg;
   NALU_HYPRE_MemoryLocation  send_memory_location;
   NALU_HYPRE_MemoryLocation  recv_memory_location;
   NALU_HYPRE_Int             num_send_bytes;
   NALU_HYPRE_Int             num_recv_bytes;
   void                 *send_data;
   void                 *recv_data;
   void                 *send_data_buffer;
   void                 *recv_data_buffer;
   NALU_HYPRE_Int             num_requests;
   hypre_MPI_Request    *requests;
} hypre_ParCSRCommHandle;

typedef hypre_ParCSRCommHandle hypre_ParCSRPersistentCommHandle;

typedef struct _hypre_ParCSRCommPkg
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
   hypre_MPI_Datatype               *send_mpi_types;
   hypre_MPI_Datatype               *recv_mpi_types;
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   hypre_ParCSRPersistentCommHandle *persistent_comm_handles[NUM_OF_COMM_PKG_JOB_TYPE];
#endif
#if defined(NALU_HYPRE_USING_GPU)
   /* temporary memory for matvec. cudaMalloc is expensive. alloc once and reuse */
   NALU_HYPRE_Complex                    *tmp_data;
   NALU_HYPRE_Complex                    *buf_data;
   hypre_CSRMatrix                  *matrix_E; /* for matvecT */
#endif
} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommPkgComm(comm_pkg)                (comm_pkg -> comm)
#define hypre_ParCSRCommPkgNumComponents(comm_pkg)       (comm_pkg -> num_components)
#define hypre_ParCSRCommPkgNumSends(comm_pkg)            (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)           (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)         (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg)       (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)      (comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)        (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg)  (comm_pkg -> device_send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)       (comm_pkg -> send_map_elmts[i])
#define hypre_ParCSRCommPkgDeviceSendMapElmt(comm_pkg,i) (comm_pkg -> device_send_map_elmts[i])
#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)            (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)           (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)         (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg)       (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)      (comm_pkg -> recv_vec_starts[i])
#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)        (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i)       (comm_pkg -> send_mpi_types[i])
#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)        (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i)       (comm_pkg -> recv_mpi_types[i])

#if defined(NALU_HYPRE_USING_GPU)
#define hypre_ParCSRCommPkgTmpData(comm_pkg)             ((comm_pkg) -> tmp_data)
#define hypre_ParCSRCommPkgBufData(comm_pkg)             ((comm_pkg) -> buf_data)
#define hypre_ParCSRCommPkgMatrixE(comm_pkg)             ((comm_pkg) -> matrix_E)
#endif

static inline void
hypre_ParCSRCommPkgCopySendMapElmtsToDevice(hypre_ParCSRCommPkg *comm_pkg)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) == NULL)
   {
      hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) =
         hypre_TAlloc(NALU_HYPRE_Int,
                      hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                      NALU_HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                    hypre_ParCSRCommPkgSendMapElmts(comm_pkg),
                    NALU_HYPRE_Int,
                    hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    NALU_HYPRE_MEMORY_DEVICE,
                    NALU_HYPRE_MEMORY_HOST);
   }
#endif
}

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRCommHandleCommPkg(comm_handle)                (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendMemoryLocation(comm_handle)     (comm_handle -> send_memory_location)
#define hypre_ParCSRCommHandleRecvMemoryLocation(comm_handle)     (comm_handle -> recv_memory_location)
#define hypre_ParCSRCommHandleNumSendBytes(comm_handle)           (comm_handle -> num_send_bytes)
#define hypre_ParCSRCommHandleNumRecvBytes(comm_handle)           (comm_handle -> num_recv_bytes)
#define hypre_ParCSRCommHandleSendData(comm_handle)               (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)               (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleSendDataBuffer(comm_handle)         (comm_handle -> send_data_buffer)
#define hypre_ParCSRCommHandleRecvDataBuffer(comm_handle)         (comm_handle -> recv_data_buffer)
#define hypre_ParCSRCommHandleNumRequests(comm_handle)            (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)               (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)             (comm_handle -> requests[i])

#endif /* NALU_HYPRE_PAR_CSR_COMMUNICATION_HEADER */
