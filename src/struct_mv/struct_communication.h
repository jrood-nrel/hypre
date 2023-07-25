/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_COMMUNICATION_HEADER
#define nalu_hypre_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_CommInfo:
 *
 * For "reverse" communication, send_transforms is not needed (may be NULL).
 * For "forward" communication, recv_transforms is not needed (may be NULL).
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_CommInfo_struct
{
   NALU_HYPRE_Int              ndim;
   nalu_hypre_BoxArrayArray   *send_boxes;
   nalu_hypre_Index            send_stride;
   NALU_HYPRE_Int            **send_processes;
   NALU_HYPRE_Int            **send_rboxnums;
   nalu_hypre_BoxArrayArray   *send_rboxes;  /* send_boxes, some with periodic shift */

   nalu_hypre_BoxArrayArray   *recv_boxes;
   nalu_hypre_Index            recv_stride;
   NALU_HYPRE_Int            **recv_processes;
   NALU_HYPRE_Int            **recv_rboxnums;
   nalu_hypre_BoxArrayArray   *recv_rboxes;  /* recv_boxes, some with periodic shift */

   NALU_HYPRE_Int              num_transforms;  /* may be 0    = identity transform */
   nalu_hypre_Index           *coords;          /* may be NULL = identity transform */
   nalu_hypre_Index           *dirs;            /* may be NULL = identity transform */
   NALU_HYPRE_Int            **send_transforms; /* may be NULL = identity transform */
   NALU_HYPRE_Int            **recv_transforms; /* may be NULL = identity transform */

   NALU_HYPRE_Int              boxes_match;  /* true (>0) if each send box has a
                                         * matching box on the recv processor */

} nalu_hypre_CommInfo;

/*--------------------------------------------------------------------------
 * nalu_hypre_CommEntryType:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_CommEntryType_struct
{
   NALU_HYPRE_Int  offset;                       /* offset for the data */
   NALU_HYPRE_Int  dim;                          /* dimension of the communication */
   NALU_HYPRE_Int  length_array[NALU_HYPRE_MAXDIM];   /* last dim has length num_values */
   NALU_HYPRE_Int  stride_array[NALU_HYPRE_MAXDIM + 1];
   NALU_HYPRE_Int *order;                        /* order of last dim values */

} nalu_hypre_CommEntryType;

/*--------------------------------------------------------------------------
 * nalu_hypre_CommType:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_CommType_struct
{
   NALU_HYPRE_Int             proc;
   NALU_HYPRE_Int             bufsize;     /* message buffer size (in doubles) */
   NALU_HYPRE_Int             num_entries;
   nalu_hypre_CommEntryType  *entries;

   /* this is only needed until first send buffer prefix is packed */
   NALU_HYPRE_Int            *rem_boxnums; /* entry remote box numbers */
   nalu_hypre_Box            *rem_boxes;   /* entry remote boxes */

} nalu_hypre_CommType;

/*--------------------------------------------------------------------------
 * nalu_hypre_CommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_CommPkg_struct
{
   MPI_Comm             comm;

   /* is this the first communication? */
   NALU_HYPRE_Int            first_comm;

   NALU_HYPRE_Int            ndim;
   NALU_HYPRE_Int            num_values;
   nalu_hypre_Index          send_stride;
   nalu_hypre_Index          recv_stride;

   /* total send buffer size (in doubles) */
   NALU_HYPRE_Int            send_bufsize;
   /* total recv buffer size (in doubles) */
   NALU_HYPRE_Int            recv_bufsize;
   /* total send buffer size (in doubles) at the first comm. */
   NALU_HYPRE_Int            send_bufsize_first_comm;
   /* total recv buffer size (in doubles) at the first comm. */
   NALU_HYPRE_Int            recv_bufsize_first_comm;

   NALU_HYPRE_Int            num_sends;
   NALU_HYPRE_Int            num_recvs;
   nalu_hypre_CommType      *send_types;
   nalu_hypre_CommType      *recv_types;

   nalu_hypre_CommType      *copy_from_type;
   nalu_hypre_CommType      *copy_to_type;

   /* these pointers are just to help free up memory for send/from types */
   nalu_hypre_CommEntryType *entries;
   NALU_HYPRE_Int           *rem_boxnums;
   nalu_hypre_Box           *rem_boxes;

   NALU_HYPRE_Int            num_orders;
   /* num_orders x num_values */
   NALU_HYPRE_Int          **orders;

   /* offsets into recv data (by box) */
   NALU_HYPRE_Int           *recv_data_offsets;
   /* recv data dimensions (by box) */
   nalu_hypre_BoxArray      *recv_data_space;

   nalu_hypre_Index          identity_coord;
   nalu_hypre_Index          identity_dir;
   NALU_HYPRE_Int           *identity_order;
} nalu_hypre_CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_CommHandle_struct
{
   nalu_hypre_CommPkg     *comm_pkg;
   NALU_HYPRE_Complex     *send_data;
   NALU_HYPRE_Complex     *recv_data;

   NALU_HYPRE_Int          num_requests;
   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status  *status;

   NALU_HYPRE_Complex    **send_buffers;
   NALU_HYPRE_Complex    **recv_buffers;

   NALU_HYPRE_Complex    **send_buffers_mpi;
   NALU_HYPRE_Complex    **recv_buffers_mpi;

   /* set = 0, add = 1 */
   NALU_HYPRE_Int          action;

} nalu_hypre_CommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_CommInto
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CommInfoNDim(info)           (info -> ndim)
#define nalu_hypre_CommInfoSendBoxes(info)      (info -> send_boxes)
#define nalu_hypre_CommInfoSendStride(info)     (info -> send_stride)
#define nalu_hypre_CommInfoSendProcesses(info)  (info -> send_processes)
#define nalu_hypre_CommInfoSendRBoxnums(info)   (info -> send_rboxnums)
#define nalu_hypre_CommInfoSendRBoxes(info)     (info -> send_rboxes)

#define nalu_hypre_CommInfoRecvBoxes(info)      (info -> recv_boxes)
#define nalu_hypre_CommInfoRecvStride(info)     (info -> recv_stride)
#define nalu_hypre_CommInfoRecvProcesses(info)  (info -> recv_processes)
#define nalu_hypre_CommInfoRecvRBoxnums(info)   (info -> recv_rboxnums)
#define nalu_hypre_CommInfoRecvRBoxes(info)     (info -> recv_rboxes)

#define nalu_hypre_CommInfoNumTransforms(info)  (info -> num_transforms)
#define nalu_hypre_CommInfoCoords(info)         (info -> coords)
#define nalu_hypre_CommInfoDirs(info)           (info -> dirs)
#define nalu_hypre_CommInfoSendTransforms(info) (info -> send_transforms)
#define nalu_hypre_CommInfoRecvTransforms(info) (info -> recv_transforms)

#define nalu_hypre_CommInfoBoxesMatch(info)     (info -> boxes_match)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_CommEntryType
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CommEntryTypeOffset(entry)       (entry -> offset)
#define nalu_hypre_CommEntryTypeDim(entry)          (entry -> dim)
#define nalu_hypre_CommEntryTypeLengthArray(entry)  (entry -> length_array)
#define nalu_hypre_CommEntryTypeStrideArray(entry)  (entry -> stride_array)
#define nalu_hypre_CommEntryTypeOrder(entry)        (entry -> order)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_CommType
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CommTypeProc(type)          (type -> proc)
#define nalu_hypre_CommTypeBufsize(type)       (type -> bufsize)
#define nalu_hypre_CommTypeNumEntries(type)    (type -> num_entries)
#define nalu_hypre_CommTypeEntries(type)       (type -> entries)
#define nalu_hypre_CommTypeEntry(type, i)    (&(type -> entries[i]))

#define nalu_hypre_CommTypeRemBoxnums(type)    (type -> rem_boxnums)
#define nalu_hypre_CommTypeRemBoxnum(type, i)  (type -> rem_boxnums[i])
#define nalu_hypre_CommTypeRemBoxes(type)      (type -> rem_boxes)
#define nalu_hypre_CommTypeRemBox(type, i)   (&(type -> rem_boxes[i]))

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_CommPkg
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CommPkgComm(comm_pkg)                       (comm_pkg -> comm)

#define nalu_hypre_CommPkgFirstComm(comm_pkg)                  (comm_pkg -> first_comm)

#define nalu_hypre_CommPkgNDim(comm_pkg)                       (comm_pkg -> ndim)
#define nalu_hypre_CommPkgNumValues(comm_pkg)                  (comm_pkg -> num_values)
#define nalu_hypre_CommPkgSendStride(comm_pkg)                 (comm_pkg -> send_stride)
#define nalu_hypre_CommPkgRecvStride(comm_pkg)                 (comm_pkg -> recv_stride)
#define nalu_hypre_CommPkgSendBufsize(comm_pkg)                (comm_pkg -> send_bufsize)
#define nalu_hypre_CommPkgRecvBufsize(comm_pkg)                (comm_pkg -> recv_bufsize)
#define nalu_hypre_CommPkgSendBufsizeFirstComm(comm_pkg)       (comm_pkg -> send_bufsize_first_comm)
#define nalu_hypre_CommPkgRecvBufsizeFirstComm(comm_pkg)       (comm_pkg -> recv_bufsize_first_comm)

#define nalu_hypre_CommPkgNumSends(comm_pkg)                   (comm_pkg -> num_sends)
#define nalu_hypre_CommPkgNumRecvs(comm_pkg)                   (comm_pkg -> num_recvs)
#define nalu_hypre_CommPkgSendTypes(comm_pkg)                  (comm_pkg -> send_types)
#define nalu_hypre_CommPkgSendType(comm_pkg, i)              (&(comm_pkg -> send_types[i]))
#define nalu_hypre_CommPkgRecvTypes(comm_pkg)                  (comm_pkg -> recv_types)
#define nalu_hypre_CommPkgRecvType(comm_pkg, i)              (&(comm_pkg -> recv_types[i]))

#define nalu_hypre_CommPkgCopyFromType(comm_pkg)               (comm_pkg -> copy_from_type)
#define nalu_hypre_CommPkgCopyToType(comm_pkg)                 (comm_pkg -> copy_to_type)

#define nalu_hypre_CommPkgEntries(comm_pkg)                    (comm_pkg -> entries)
#define nalu_hypre_CommPkgRemBoxnums(comm_pkg)                 (comm_pkg -> rem_boxnums)
#define nalu_hypre_CommPkgRemBoxes(comm_pkg)                   (comm_pkg -> rem_boxes)

#define nalu_hypre_CommPkgNumOrders(comm_pkg)                  (comm_pkg -> num_orders)
#define nalu_hypre_CommPkgOrders(comm_pkg)                     (comm_pkg -> orders)

#define nalu_hypre_CommPkgRecvDataOffsets(comm_pkg)            (comm_pkg -> recv_data_offsets)
#define nalu_hypre_CommPkgRecvDataSpace(comm_pkg)              (comm_pkg -> recv_data_space)

#define nalu_hypre_CommPkgIdentityCoord(comm_pkg)              (comm_pkg -> identity_coord)
#define nalu_hypre_CommPkgIdentityDir(comm_pkg)                (comm_pkg -> identity_dir)
#define nalu_hypre_CommPkgIdentityOrder(comm_pkg)              (comm_pkg -> identity_order)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_CommHandle
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CommHandleCommPkg(comm_handle)              (comm_handle -> comm_pkg)
#define nalu_hypre_CommHandleSendData(comm_handle)             (comm_handle -> send_data)
#define nalu_hypre_CommHandleRecvData(comm_handle)             (comm_handle -> recv_data)
#define nalu_hypre_CommHandleNumRequests(comm_handle)          (comm_handle -> num_requests)
#define nalu_hypre_CommHandleRequests(comm_handle)             (comm_handle -> requests)
#define nalu_hypre_CommHandleStatus(comm_handle)               (comm_handle -> status)
#define nalu_hypre_CommHandleSendBuffers(comm_handle)          (comm_handle -> send_buffers)
#define nalu_hypre_CommHandleRecvBuffers(comm_handle)          (comm_handle -> recv_buffers)
#define nalu_hypre_CommHandleAction(comm_handle)               (comm_handle -> action)
#define nalu_hypre_CommHandleSendBuffersMPI(comm_handle)       (comm_handle -> send_buffers_mpi)
#define nalu_hypre_CommHandleRecvBuffersMPI(comm_handle)       (comm_handle -> recv_buffers_mpi)

#endif
