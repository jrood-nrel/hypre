/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef NALU_HYPRE_PAR_CSR_PMVCOMM_HEADER
#define NALU_HYPRE_PAR_CSR_PMVCOMM_HEADER

#include "_nalu_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRCommMultiHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_ParCSRCommPkg  *comm_pkg;
   void          *send_data;
   void          *recv_data;
   NALU_HYPRE_Int                  num_requests;
   nalu_hypre_MPI_Request          *requests;

} nalu_hypre_ParCSRCommMultiHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_ParCSRCommMultiHandle
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ParCSRCommMultiHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define nalu_hypre_ParCSRCommMultiHandleSendData(comm_handle)    (comm_handle -> send_data)
#define nalu_hypre_ParCSRCommMultiHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define nalu_hypre_ParCSRCommMultiHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define nalu_hypre_ParCSRCommMultiHandleRequests(comm_handle)    (comm_handle -> requests)
#define nalu_hypre_ParCSRCommMultiHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

nalu_hypre_ParCSRCommMultiHandle *
nalu_hypre_ParCSRCommMultiHandleCreate ( NALU_HYPRE_Int             job,
                                    nalu_hypre_ParCSRCommPkg   *comm_pkg,
                                    void                  *send_data,
                                    void                  *recv_data,
                                    NALU_HYPRE_Int                   nvecs       );


NALU_HYPRE_Int
nalu_hypre_ParCSRCommMultiHandleDestroy(nalu_hypre_ParCSRCommMultiHandle *comm_handle);

#ifdef __cplusplus
}
#endif

#endif /* NALU_HYPRE_PAR_CSR_MULTICOMMUNICATION_HEADER */
