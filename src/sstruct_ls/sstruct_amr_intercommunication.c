/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructAMRInterCommunication: Given the sendinfo, recvinfo, etc.,
 * a communication pkg is formed. This pkg may be used for amr inter_level
 * communication.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructAMRInterCommunication( nalu_hypre_SStructSendInfoData *sendinfo,
                                    nalu_hypre_SStructRecvInfoData *recvinfo,
                                    nalu_hypre_BoxArray            *send_data_space,
                                    nalu_hypre_BoxArray            *recv_data_space,
                                    NALU_HYPRE_Int                  num_values,
                                    MPI_Comm                   comm,
                                    nalu_hypre_CommPkg            **comm_pkg_ptr )
{
   nalu_hypre_CommInfo         *comm_info;
   nalu_hypre_CommPkg          *comm_pkg;

   nalu_hypre_BoxArrayArray    *sendboxes;
   NALU_HYPRE_Int             **sprocesses;
   nalu_hypre_BoxArrayArray    *send_rboxes;
   NALU_HYPRE_Int             **send_rboxnums;

   nalu_hypre_BoxArrayArray    *recvboxes;
   NALU_HYPRE_Int             **rprocesses;
   nalu_hypre_BoxArrayArray    *recv_rboxes;
   NALU_HYPRE_Int             **recv_rboxnums;

   nalu_hypre_BoxArray         *boxarray;

   NALU_HYPRE_Int               i, j;
   NALU_HYPRE_Int               ierr = 0;

   /*------------------------------------------------------------------------
    *  The communication info is copied from sendinfo & recvinfo.
    *------------------------------------------------------------------------*/
   sendboxes  = nalu_hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);
   send_rboxes = nalu_hypre_BoxArrayArrayDuplicate(sendinfo -> send_boxes);

   sprocesses   = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArrayArraySize(send_rboxes), NALU_HYPRE_MEMORY_HOST);
   send_rboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArrayArraySize(send_rboxes),
                                 NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxArrayI(i, sendboxes)
   {
      boxarray = nalu_hypre_BoxArrayArrayBoxArray(sendboxes, i);
      sprocesses[i]   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);
      send_rboxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(j, boxarray)
      {
         sprocesses[i][j]   = (sendinfo -> send_procs)[i][j];
         send_rboxnums[i][j] = (sendinfo -> send_remote_boxnums)[i][j];
      }
   }

   recvboxes   = nalu_hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   recv_rboxes = nalu_hypre_BoxArrayArrayDuplicate(recvinfo -> recv_boxes);
   rprocesses  = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArrayArraySize(recvboxes), NALU_HYPRE_MEMORY_HOST);

   /* dummy pointer for CommInfoCreate */
   recv_rboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArrayArraySize(recvboxes), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxArrayI(i, recvboxes)
   {
      boxarray = nalu_hypre_BoxArrayArrayBoxArray(recvboxes, i);
      rprocesses[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);
      recv_rboxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(j, boxarray)
      {
         rprocesses[i][j]   = (recvinfo -> recv_procs)[i][j];
      }
   }


   nalu_hypre_CommInfoCreate(sendboxes, recvboxes, sprocesses, rprocesses,
                        send_rboxnums, recv_rboxnums, send_rboxes,
                        recv_rboxes, 1, &comm_info);

   nalu_hypre_CommPkgCreate(comm_info,
                       send_data_space,
                       recv_data_space,
                       num_values, NULL, 0, comm,
                       &comm_pkg);
   nalu_hypre_CommInfoDestroy(comm_info);

   *comm_pkg_ptr = comm_pkg;

   return ierr;
}


