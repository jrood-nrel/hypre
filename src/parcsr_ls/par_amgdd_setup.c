/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.h"

/*****************************************************************************
 *
 * Routine for setting up the composite grids in AMG-DD

 *****************************************************************************/

/*****************************************************************************
 * nalu_hypre_BoomerAMGDDSetup
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSetup( void               *amgdd_vdata,
                        nalu_hypre_ParCSRMatrix *A,
                        nalu_hypre_ParVector    *b,
                        nalu_hypre_ParVector    *x )
{
   MPI_Comm               comm;
   nalu_hypre_ParAMGDDData     *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_ParAMGData       *amg_data   = nalu_hypre_ParAMGDDDataAMG(amgdd_data);

   NALU_HYPRE_Int               pad;
   NALU_HYPRE_Int               num_levels;
   NALU_HYPRE_Int               amgdd_start_level;
   NALU_HYPRE_Int               num_ghost_layers;

   nalu_hypre_ParCSRMatrix    **A_array;
   nalu_hypre_AMGDDCompGrid   **compGrid;
   nalu_hypre_AMGDDCommPkg     *compGridCommPkg;
   NALU_HYPRE_Int              *padding;
   NALU_HYPRE_Int              *nodes_added_on_level;
   NALU_HYPRE_Int             **A_tmp_info;

   nalu_hypre_MPI_Request      *requests;
   nalu_hypre_MPI_Status       *status;
   NALU_HYPRE_Int             **send_buffer_size;
   NALU_HYPRE_Int             **recv_buffer_size;
   NALU_HYPRE_Int            ***num_send_nodes;
   NALU_HYPRE_Int            ***num_recv_nodes;
   NALU_HYPRE_Int           ****send_flag;
   NALU_HYPRE_Int           ****recv_map;
   NALU_HYPRE_Int           ****recv_red_marker;
   NALU_HYPRE_Int             **send_buffer;
   NALU_HYPRE_Int             **recv_buffer;
   NALU_HYPRE_Int             **send_flag_buffer;
   NALU_HYPRE_Int             **recv_map_send_buffer;
   NALU_HYPRE_Int              *send_flag_buffer_size;
   NALU_HYPRE_Int              *recv_map_send_buffer_size;

   NALU_HYPRE_Int               num_procs;
   NALU_HYPRE_Int               num_send_procs;
   NALU_HYPRE_Int               num_recv_procs;
   NALU_HYPRE_Int               level, i, j;
   NALU_HYPRE_Int               num_requests;
   NALU_HYPRE_Int               request_counter;

   // If the underlying AMG data structure has not yet been set up, call BoomerAMGSetup()
   if (!nalu_hypre_ParAMGDataAArray(amg_data))
   {
      nalu_hypre_BoomerAMGSetup((void*) amg_data, A, b, x);
   }

   // Get number of processes
   comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   // get info from amg about how to setup amgdd
   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   pad               = nalu_hypre_ParAMGDDDataPadding(amgdd_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   num_ghost_layers  = nalu_hypre_ParAMGDDDataNumGhostLayers(amgdd_data);
   amgdd_start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);
   if (amgdd_start_level >= (num_levels - 1))
   {
      amgdd_start_level = num_levels - 2;
      nalu_hypre_ParAMGDDDataStartLevel(amgdd_data) = amgdd_start_level;
   }

   // Allocate pointer for the composite grids
   compGrid = nalu_hypre_CTAlloc(nalu_hypre_AMGDDCompGrid *, num_levels, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParAMGDDDataCompGrid(amgdd_data) = compGrid;

   // In the 1 processor case, just need to initialize the comp grids
   if (num_procs == 1)
   {
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         compGrid[level] = nalu_hypre_AMGDDCompGridCreate();
         nalu_hypre_AMGDDCompGridInitialize(amgdd_data, 0, level);
      }
      nalu_hypre_AMGDDCompGridFinalize(amgdd_data);
      nalu_hypre_AMGDDCompGridSetupRelax(amgdd_data);

      return nalu_hypre_error_flag;
   }

   // Get the padding on each level
   padding = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_levels, NALU_HYPRE_MEMORY_HOST);
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      padding[level] = pad;
   }

   // Initialize composite grid structures
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      compGrid[level] = nalu_hypre_AMGDDCompGridCreate();
      nalu_hypre_AMGDDCompGridInitialize(amgdd_data, padding[level], level);
   }

   // Create the compGridCommPkg and grab a few frequently used variables
   compGridCommPkg = nalu_hypre_AMGDDCommPkgCreate(num_levels);
   nalu_hypre_ParAMGDDDataCommPkg(amgdd_data) = compGridCommPkg;

   send_buffer_size = nalu_hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = nalu_hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg);
   send_flag = nalu_hypre_AMGDDCommPkgSendFlag(compGridCommPkg);
   num_send_nodes = nalu_hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg);
   num_recv_nodes = nalu_hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg);
   recv_map = nalu_hypre_AMGDDCommPkgRecvMap(compGridCommPkg);
   recv_red_marker = nalu_hypre_AMGDDCommPkgRecvRedMarker(compGridCommPkg);
   nodes_added_on_level = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_levels, NALU_HYPRE_MEMORY_HOST);

   // On each level, setup the compGridCommPkg so that it has communication info for distance (eta + numGhostLayers)
   for (level = amgdd_start_level; level < num_levels; level++)
   {
      nalu_hypre_BoomerAMGDD_SetupNearestProcessorNeighbors(A_array[level],
                                                       compGridCommPkg,
                                                       level,
                                                       padding,
                                                       num_ghost_layers);
   }

   // Find maximum number of requests and allocate memory
   num_requests = 0;
   for (level = num_levels - 1; level >= amgdd_start_level; level--)
   {
      comm = nalu_hypre_ParCSRMatrixComm(A_array[level]);
      num_send_procs = nalu_hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
      num_recv_procs = nalu_hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];
      num_requests   = nalu_hypre_max(num_requests, num_send_procs + num_recv_procs);
   }
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_requests, NALU_HYPRE_MEMORY_HOST);
   status   = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_requests, NALU_HYPRE_MEMORY_HOST);

   /////////////////////////////////////////////////////////////////

   // Loop over levels from coarsest to finest to build up the composite grids

   /////////////////////////////////////////////////////////////////

   for (level = num_levels - 1; level >= amgdd_start_level; level--)
   {
      comm = nalu_hypre_ParCSRMatrixComm(A_array[level]);
      num_send_procs = nalu_hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
      num_recv_procs = nalu_hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];
      num_requests   = num_send_procs + num_recv_procs;

      // Initialize request counter
      request_counter = 0;

      //////////// Communicate buffer sizes ////////////
      if (num_recv_procs)
      {
         recv_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_recv_procs, NALU_HYPRE_MEMORY_HOST);
         recv_buffer_size[level] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recv_procs, NALU_HYPRE_MEMORY_HOST);

         recv_map[level] = nalu_hypre_CTAlloc(NALU_HYPRE_Int**, num_recv_procs, NALU_HYPRE_MEMORY_HOST);
         recv_red_marker[level] = nalu_hypre_CTAlloc(NALU_HYPRE_Int**, num_recv_procs, NALU_HYPRE_MEMORY_HOST);
         num_recv_nodes[level] = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_recv_procs, NALU_HYPRE_MEMORY_HOST);

         recv_map_send_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_recv_procs, NALU_HYPRE_MEMORY_HOST);
         recv_map_send_buffer_size = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recv_procs, NALU_HYPRE_MEMORY_HOST);

         // Post the receives for the buffer size
         for (i = 0; i < num_recv_procs; i++)
         {
            nalu_hypre_MPI_Irecv(&(recv_buffer_size[level][i]), 1, NALU_HYPRE_MPI_INT,
                            nalu_hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
         }
      }

      if (num_send_procs)
      {
         send_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_send_procs, NALU_HYPRE_MEMORY_HOST);
         send_buffer_size[level] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_send_procs, NALU_HYPRE_MEMORY_HOST);
         send_flag_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_send_procs, NALU_HYPRE_MEMORY_HOST);
         send_flag_buffer_size = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_send_procs, NALU_HYPRE_MEMORY_HOST);

         // Pack send buffers
         for (i = 0; i < num_send_procs; i++)
         {
            send_buffer[i] = nalu_hypre_BoomerAMGDD_PackSendBuffer(amgdd_data, i, level, padding,
                                                              &(send_flag_buffer_size[i]));
         }

         // Send the buffer sizes
         for (i = 0; i < num_send_procs; i++)
         {
            nalu_hypre_MPI_Isend(&(send_buffer_size[level][i]), 1, NALU_HYPRE_MPI_INT,
                            nalu_hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 0, comm, &(requests[request_counter++]));
         }
      }

      // Wait for all buffer sizes to be received
      nalu_hypre_MPI_Waitall(num_requests, requests, status);
      request_counter = 0;

      //////////// Communicate buffers ////////////
      for (i = 0; i < num_recv_procs; i++)
      {
         recv_buffer[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_buffer_size[level][i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_MPI_Irecv(recv_buffer[i], recv_buffer_size[level][i], NALU_HYPRE_MPI_INT,
                         nalu_hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
      }

      for (i = 0; i < num_send_procs; i++)
      {
         nalu_hypre_MPI_Isend(send_buffer[i], send_buffer_size[level][i], NALU_HYPRE_MPI_INT,
                         nalu_hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 1, comm, &(requests[request_counter++]));
      }

      // Wait for buffers to be received
      nalu_hypre_MPI_Waitall(num_requests, requests, status);
      request_counter = 0;

      //////////// Unpack the received buffers ////////////
      A_tmp_info = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_recv_procs, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_recv_procs; i++)
      {
         recv_map[level][i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_levels, NALU_HYPRE_MEMORY_HOST);
         recv_red_marker[level][i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_levels, NALU_HYPRE_MEMORY_HOST);
         num_recv_nodes[level][i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_levels, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_BoomerAMGDD_UnpackRecvBuffer(amgdd_data, recv_buffer[i], A_tmp_info,
                                            &(recv_map_send_buffer_size[i]), nodes_added_on_level, level, i);

         recv_map_send_buffer[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_map_send_buffer_size[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoomerAMGDD_PackRecvMapSendBuffer(recv_map_send_buffer[i], recv_red_marker[level][i],
                                                 num_recv_nodes[level][i], &(recv_buffer_size[level][i]), level, num_levels);
      }

      //////////// Setup local indices for the composite grid ////////////
      nalu_hypre_AMGDDCompGridSetupLocalIndices(compGrid, nodes_added_on_level, recv_map, num_recv_procs,
                                           A_tmp_info, level, num_levels);
      for (j = level; j < num_levels; j++)
      {
         nodes_added_on_level[j] = 0;
      }

      //////////// Communicate redundancy info ////////////
      // post receives for send maps
      for (i = 0; i < num_send_procs; i++)
      {
         send_flag_buffer[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_flag_buffer_size[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_MPI_Irecv(send_flag_buffer[i], send_flag_buffer_size[i], NALU_HYPRE_MPI_INT,
                         nalu_hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
      }

      // send the recv_map_send_buffer's
      for (i = 0; i < num_recv_procs; i++)
      {
         nalu_hypre_MPI_Isend(recv_map_send_buffer[i], recv_map_send_buffer_size[i], NALU_HYPRE_MPI_INT,
                         nalu_hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 2, comm, &(requests[request_counter++]));
      }

      // wait for maps to be received
      nalu_hypre_MPI_Waitall(num_requests, requests, status);

      // unpack and setup the send flag arrays
      for (i = 0; i < num_send_procs; i++)
      {
         nalu_hypre_BoomerAMGDD_UnpackSendFlagBuffer(compGrid, send_flag_buffer[i], send_flag[level][i],
                                                num_send_nodes[level][i], &(send_buffer_size[level][i]), level, num_levels);
      }

      // clean up memory for this level
      for (i = 0; i < num_send_procs; i++)
      {
         nalu_hypre_TFree(send_buffer[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_flag_buffer[i], NALU_HYPRE_MEMORY_HOST);
      }
      for (i = 0; i < num_recv_procs; i++)
      {
         nalu_hypre_TFree(recv_buffer[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(recv_map_send_buffer[i], NALU_HYPRE_MEMORY_HOST);
      }

      if (num_send_procs)
      {
         nalu_hypre_TFree(send_buffer, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_flag_buffer, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_flag_buffer_size, NALU_HYPRE_MEMORY_HOST);
      }
      if (num_recv_procs)
      {
         nalu_hypre_TFree(recv_buffer, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(recv_map_send_buffer, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(recv_map_send_buffer_size, NALU_HYPRE_MEMORY_HOST);
      }
   }

   /////////////////////////////////////////////////////////////////

   // Done with loop over levels. Now just finalize things.

   /////////////////////////////////////////////////////////////////

   nalu_hypre_BoomerAMGDD_FixUpRecvMaps(compGrid, compGridCommPkg, amgdd_start_level, num_levels);

   // Communicate data for A and all info for P
   nalu_hypre_BoomerAMGDD_CommunicateRemainingMatrixInfo(amgdd_data);

   // Setup the local indices for P
   nalu_hypre_AMGDDCompGridSetupLocalIndicesP(amgdd_data);

   // Finalize the comp grid structures
   nalu_hypre_AMGDDCompGridFinalize(amgdd_data);

   // Setup extra info for specific relaxation methods
   nalu_hypre_AMGDDCompGridSetupRelax(amgdd_data);

   // Cleanup memory
   nalu_hypre_TFree(padding, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nodes_added_on_level, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
