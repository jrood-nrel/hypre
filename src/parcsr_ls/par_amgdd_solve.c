/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDDSolve( void               *amgdd_vdata,
                        nalu_hypre_ParCSRMatrix *A,
                        nalu_hypre_ParVector    *f,
                        nalu_hypre_ParVector    *u )
{
   nalu_hypre_ParAMGDDData   *amgdd_data = (nalu_hypre_ParAMGDDData*) amgdd_vdata;
   nalu_hypre_ParAMGData     *amg_data   = nalu_hypre_ParAMGDDDataAMG(amgdd_data);

   nalu_hypre_AMGDDCompGrid **compGrids;
   nalu_hypre_ParCSRMatrix  **A_array;
   nalu_hypre_ParCSRMatrix  **P_array;
   nalu_hypre_ParVector     **F_array;
   nalu_hypre_ParVector     **U_array;
   nalu_hypre_ParVector      *res;
   nalu_hypre_ParVector      *Vtemp;
   nalu_hypre_ParVector      *Ztemp;

   NALU_HYPRE_Int             myid;
   NALU_HYPRE_Int             min_iter;
   NALU_HYPRE_Int             max_iter;
   NALU_HYPRE_Int             converge_type;
   NALU_HYPRE_Int             i, level;
   NALU_HYPRE_Int             num_levels;
   NALU_HYPRE_Int             amgdd_start_level;
   NALU_HYPRE_Int             fac_num_cycles;
   NALU_HYPRE_Int             cycle_count;
   NALU_HYPRE_Int             amg_print_level;
   NALU_HYPRE_Int             amg_logging;
   NALU_HYPRE_Real            tol;
   NALU_HYPRE_Real            resid_nrm;
   NALU_HYPRE_Real            resid_nrm_init;
   NALU_HYPRE_Real            rhs_norm;
   NALU_HYPRE_Real            old_resid;
   NALU_HYPRE_Real            relative_resid;
   NALU_HYPRE_Real            conv_factor;
   NALU_HYPRE_Real            alpha = -1.0;
   NALU_HYPRE_Real            beta = 1.0;
   NALU_HYPRE_Real            ieee_check = 0.0;

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /* Set some data */
   amgdd_start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);
   fac_num_cycles    = nalu_hypre_ParAMGDDDataFACNumCycles(amgdd_data);
   compGrids         = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   amg_print_level   = nalu_hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging       = nalu_hypre_ParAMGDataLogging(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   converge_type     = nalu_hypre_ParAMGDataConvergeType(amg_data);
   min_iter          = nalu_hypre_ParAMGDataMinIter(amg_data);
   max_iter          = nalu_hypre_ParAMGDataMaxIter(amg_data);
   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   P_array           = nalu_hypre_ParAMGDataPArray(amg_data);
   F_array           = nalu_hypre_ParAMGDataFArray(amg_data);
   U_array           = nalu_hypre_ParAMGDataUArray(amg_data);
   Vtemp             = nalu_hypre_ParAMGDataVtemp(amg_data);
   Ztemp             = nalu_hypre_ParAMGDDDataZtemp(amg_data);
   tol               = nalu_hypre_ParAMGDataTol(amg_data);
   cycle_count       = 0;
   if (amg_logging > 1)
   {
      res = nalu_hypre_ParAMGDataResidual(amg_data);
   }

   // Setup extra temporary variable to hold the solution if necessary
   if (!Ztemp)
   {
      Ztemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_array[amgdd_start_level]),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A_array[amgdd_start_level]),
                                    nalu_hypre_ParCSRMatrixRowStarts(A_array[amgdd_start_level]));
      nalu_hypre_ParVectorInitialize(Ztemp);
      nalu_hypre_ParAMGDDDataZtemp(amg_data) = Ztemp;
   }

   /*-----------------------------------------------------------------------
    * Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (myid == 0 && amg_print_level > 1)
   {
      nalu_hypre_BoomerAMGWriteSolverParams(amg_data);
   }

   /*-----------------------------------------------------------------------
    * Set the fine grid operator, left-hand side, and right-hand side
    *-----------------------------------------------------------------------*/
   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;
   if (A != A_array[0])
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "WARNING: calling nalu_hypre_BoomerAMGDDSolve with different matrix than what was used for initial setup. "
                        "Non-owned parts of fine-grid matrix and fine-grid communication patterns may be incorrect.\n");
      nalu_hypre_AMGDDCompGridMatrixOwnedDiag(nalu_hypre_AMGDDCompGridA(compGrids[0])) = nalu_hypre_ParCSRMatrixDiag(A);
      nalu_hypre_AMGDDCompGridMatrixOwnedOffd(nalu_hypre_AMGDDCompGridA(compGrids[0])) = nalu_hypre_ParCSRMatrixOffd(A);
   }

   if (compGrids[0])
   {
      nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridU(compGrids[0])) = nalu_hypre_ParVectorLocalVector(u);
      nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridF(compGrids[0])) = nalu_hypre_ParVectorLocalVector(f);
   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/
   if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
   {
      if (amg_logging > 1)
      {
         nalu_hypre_ParVectorCopy(F_array[0], res);
         if (tol > 0.)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, res);
         }
         resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(res, res));
      }
      else
      {
         nalu_hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > 0.)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
         }
         resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resid_nrm != 0.)
      {
         ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (amg_print_level > 0)
         {
            nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            nalu_hypre_printf("ERROR -- nalu_hypre_BoomerAMGDDSolve: INFs and/or NaNs detected in input.\n");
            nalu_hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);

         return nalu_hypre_error_flag;
      }

      /* r0 */
      resid_nrm_init = resid_nrm;

      if (0 == converge_type)
      {
         rhs_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(f, f));
         if (rhs_norm)
         {
            relative_resid = resid_nrm_init / rhs_norm;
         }
         else
         {
            relative_resid = resid_nrm_init;
         }
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.;
   }

   if (myid == 0 && amg_print_level > 1)
   {
      nalu_hypre_printf("                                            relative\n");
      nalu_hypre_printf("               residual        factor       residual\n");
      nalu_hypre_printf("               --------        ------       --------\n");
      nalu_hypre_printf("    Initial    %e                 %e\n",
                   resid_nrm_init, relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main cycle loop
    *-----------------------------------------------------------------------*/
   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      // Do normal AMG V-cycle down-sweep to where we start AMG-DD
      if (amgdd_start_level > 0)
      {
         nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         nalu_hypre_ParAMGDataPartialCycleControl(amg_data) = 0;
         nalu_hypre_BoomerAMGCycle( (void*) amg_data, F_array, U_array);
      }
      else
      {
         // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
         nalu_hypre_ParVectorCopy(F_array[amgdd_start_level], Vtemp);
         nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[amgdd_start_level],
                                  U_array[amgdd_start_level], beta,
                                  F_array[amgdd_start_level]);
      }

      // AMG-DD cycle
      nalu_hypre_BoomerAMGDD_ResidualCommunication(amgdd_data);

      // Save the original solution (updated at the end of the AMG-DD cycle)
      nalu_hypre_ParVectorCopy(U_array[amgdd_start_level], Ztemp);

      // Zero solution on all levels
      for (level = amgdd_start_level; level < num_levels; level++)
      {
         nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridU(compGrids[level]), 0.0);

         if (nalu_hypre_AMGDDCompGridQ(compGrids[level]))
         {
            nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridQ(compGrids[level]), 0.0);
         }
      }

      for (level = amgdd_start_level; level < num_levels; level++)
      {
         nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridT(compGrids[level]), 0.0 );
         nalu_hypre_AMGDDCompGridVectorSetConstantValues(nalu_hypre_AMGDDCompGridS(compGrids[level]), 0.0 );
      }

      // Do FAC cycles
      if (fac_num_cycles > 0)
      {
         nalu_hypre_BoomerAMGDD_FAC((void*) amgdd_data, 1);
      }
      for (i = 1; i < fac_num_cycles; i++)
      {
         nalu_hypre_BoomerAMGDD_FAC((void*) amgdd_data, 0);
      }

      // Update fine grid solution
      nalu_hypre_ParVectorAxpy(1.0, Ztemp, U_array[amgdd_start_level]);

      // Do normal AMG V-cycle up-sweep back up to the fine grid
      if (amgdd_start_level > 0)
      {
         // Interpolate
         nalu_hypre_ParCSRMatrixMatvec(1.0, P_array[amgdd_start_level - 1],
                                  U_array[amgdd_start_level], 1.0,
                                  U_array[amgdd_start_level - 1]);
         // V-cycle back to finest grid
         nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = amgdd_start_level - 1;
         nalu_hypre_ParAMGDataPartialCycleControl(amg_data) = 1;

         nalu_hypre_BoomerAMGCycle((void*) amg_data, F_array, U_array);

         nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = - 1;
         nalu_hypre_ParAMGDataPartialCycleControl(amg_data) = -1;
      }
      else
      {
         // Copy RHS back into f
         nalu_hypre_ParVectorCopy(Vtemp, F_array[amgdd_start_level]);
      }

      /*---------------------------------------------------------------
       * Compute fine-grid residual and residual norm
       *----------------------------------------------------------------*/
      if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
      {
         old_resid = resid_nrm;

         if (amg_logging > 1)
         {
            nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta,
                                               F_array[0], res);
            resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(res, res));
         }
         else
         {
            nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta,
                                               F_array[0], Vtemp);
            resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         if (old_resid)
         {
            conv_factor = resid_nrm / old_resid;
         }
         else
         {
            conv_factor = resid_nrm;
         }

         if (0 == converge_type)
         {
            if (rhs_norm)
            {
               relative_resid = resid_nrm / rhs_norm;
            }
            else
            {
               relative_resid = resid_nrm;
            }
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         nalu_hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }

      if (myid == 0 && amg_print_level > 1)
      {
         nalu_hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }

      // Update cycle counter
      ++cycle_count;
      nalu_hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
   }

   if (cycle_count == max_iter && tol > 0.)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("\n\n==============================================");
         nalu_hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         nalu_hypre_printf("      within the allowed %d V-cycles\n", max_iter);
         nalu_hypre_printf("==============================================");
      }

      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   if (myid == 0 && amg_print_level > 1)
   {
      nalu_hypre_printf("\n");
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: Don't reallocate requests/sends at each level. Implement
 *       a nalu_hypre_AMGDDCommPkgHandle data structure (see nalu_hypre_ParCSRCommHandle)
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_ResidualCommunication( nalu_hypre_ParAMGDDData *amgdd_data )
{
   nalu_hypre_ParAMGData      *amg_data = nalu_hypre_ParAMGDDDataAMG(amgdd_data);

   // info from amg
   nalu_hypre_ParCSRMatrix   **A_array;
   nalu_hypre_ParCSRMatrix   **R_array;
   nalu_hypre_ParVector      **F_array;
   nalu_hypre_AMGDDCommPkg    *compGridCommPkg;
   nalu_hypre_AMGDDCompGrid  **compGrid;

   // temporary arrays used for communication during comp grid setup
   NALU_HYPRE_Complex        **send_buffers;
   NALU_HYPRE_Complex        **recv_buffers;

   // MPI stuff
   MPI_Comm               comm;
   nalu_hypre_MPI_Request     *requests;
   nalu_hypre_MPI_Status      *status;
   NALU_HYPRE_Int              request_counter = 0;
   NALU_HYPRE_Int              num_procs;
   NALU_HYPRE_Int              num_sends, num_recvs;
   NALU_HYPRE_Int              send_buffer_size, recv_buffer_size;

   NALU_HYPRE_Int              num_levels, amgdd_start_level;
   NALU_HYPRE_Int              level, i;

   // Get info from amg
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   amgdd_start_level = nalu_hypre_ParAMGDDDataStartLevel(amgdd_data);
   compGrid          = nalu_hypre_ParAMGDDDataCompGrid(amgdd_data);
   compGridCommPkg   = nalu_hypre_ParAMGDDDataCommPkg(amgdd_data);
   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   R_array           = nalu_hypre_ParAMGDataRArray(amg_data);
   F_array           = nalu_hypre_ParAMGDataFArray(amg_data);

   // Restrict residual down to all levels
   for (level = amgdd_start_level; level < num_levels - 1; level++)
   {
      if (nalu_hypre_ParAMGDataRestriction(amg_data))
      {
         nalu_hypre_ParCSRMatrixMatvec(1.0, R_array[level], F_array[level], 0.0, F_array[level + 1]);
      }
      else
      {
         nalu_hypre_ParCSRMatrixMatvecT(1.0, R_array[level], F_array[level], 0.0, F_array[level + 1]);
      }
   }

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = num_levels - 1; level >= amgdd_start_level; level--)
   {
      // Get some communication info
      comm = nalu_hypre_ParCSRMatrixComm(A_array[level]);
      nalu_hypre_MPI_Comm_size(comm, &num_procs);

      if (num_procs > 1)
      {
         num_sends = nalu_hypre_AMGDDCommPkgNumSendProcs(compGridCommPkg)[level];
         num_recvs = nalu_hypre_AMGDDCommPkgNumRecvProcs(compGridCommPkg)[level];

         if ( num_sends || num_recvs ) // If there are any owned nodes on this level
         {
            // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
            recv_buffers = nalu_hypre_CTAlloc(NALU_HYPRE_Complex *, num_recvs, NALU_HYPRE_MEMORY_HOST);
            send_buffers = nalu_hypre_CTAlloc(NALU_HYPRE_Complex *, num_sends, NALU_HYPRE_MEMORY_HOST);
            request_counter = 0;
            requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);
            status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);

            // allocate space for the receive buffers and post the receives
            for (i = 0; i < num_recvs; i++)
            {
               recv_buffer_size = nalu_hypre_AMGDDCommPkgRecvBufferSize(compGridCommPkg)[level][i];
               recv_buffers[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, recv_buffer_size, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_MPI_Irecv(recv_buffers[i], recv_buffer_size, NALU_HYPRE_MPI_COMPLEX,
                               nalu_hypre_AMGDDCommPkgRecvProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            for (i = 0; i < num_sends; i++)
            {
               send_buffer_size = nalu_hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[level][i];
               send_buffers[i] = nalu_hypre_BoomerAMGDD_PackResidualBuffer(compGrid, compGridCommPkg, level, i);
               nalu_hypre_MPI_Isend(send_buffers[i], send_buffer_size, NALU_HYPRE_MPI_COMPLEX,
                               nalu_hypre_AMGDDCommPkgSendProcs(compGridCommPkg)[level][i], 3, comm, &requests[request_counter++]);
            }

            // wait for buffers to be received
            nalu_hypre_MPI_Waitall( request_counter, requests, status );

            nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_sends; i++)
            {
               nalu_hypre_TFree(send_buffers[i], NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_TFree(send_buffers, NALU_HYPRE_MEMORY_HOST);

            // Unpack recv buffers
            for (i = 0; i < num_recvs; i++)
            {
               nalu_hypre_BoomerAMGDD_UnpackResidualBuffer(recv_buffers[i], compGrid, compGridCommPkg, level, i);
            }

            // clean up memory for this level
            for (i = 0; i < num_recvs; i++)
            {
               nalu_hypre_TFree(recv_buffers[i], NALU_HYPRE_MEMORY_HOST);
            }
            nalu_hypre_TFree(recv_buffers, NALU_HYPRE_MEMORY_HOST);
         }
      }
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Complex*
nalu_hypre_BoomerAMGDD_PackResidualBuffer( nalu_hypre_AMGDDCompGrid **compGrid,
                                      nalu_hypre_AMGDDCommPkg   *compGridCommPkg,
                                      NALU_HYPRE_Int             current_level,
                                      NALU_HYPRE_Int             proc )
{
   NALU_HYPRE_Complex  *buffer;
   NALU_HYPRE_Int       level, i;
   NALU_HYPRE_Int       send_elmt;
   NALU_HYPRE_Int       cnt = 0;

   buffer = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                          nalu_hypre_AMGDDCommPkgSendBufferSize(compGridCommPkg)[current_level][proc], NALU_HYPRE_MEMORY_HOST);
   for (level = current_level; level < nalu_hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumSendNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         send_elmt = nalu_hypre_AMGDDCommPkgSendFlag(compGridCommPkg)[current_level][proc][level][i];
         if (send_elmt < nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]))
         {
            buffer[cnt++] = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorOwned(nalu_hypre_AMGDDCompGridF(
                                                                               compGrid[level])))[send_elmt];
         }
         else
         {
            send_elmt -= nalu_hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
            buffer[cnt++] = nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridF(
                                                                                  compGrid[level])))[send_elmt];
         }
      }
   }

   return buffer;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDD_UnpackResidualBuffer( NALU_HYPRE_Complex        *buffer,
                                        nalu_hypre_AMGDDCompGrid **compGrid,
                                        nalu_hypre_AMGDDCommPkg   *compGridCommPkg,
                                        NALU_HYPRE_Int             current_level,
                                        NALU_HYPRE_Int             proc )
{
   NALU_HYPRE_Int  recv_elmt;
   NALU_HYPRE_Int  level, i;
   NALU_HYPRE_Int  cnt = 0;

   for (level = current_level; level < nalu_hypre_AMGDDCommPkgNumLevels(compGridCommPkg); level++)
   {
      for (i = 0; i < nalu_hypre_AMGDDCommPkgNumRecvNodes(compGridCommPkg)[current_level][proc][level]; i++)
      {
         recv_elmt = nalu_hypre_AMGDDCommPkgRecvMap(compGridCommPkg)[current_level][proc][level][i];
         nalu_hypre_VectorData(nalu_hypre_AMGDDCompGridVectorNonOwned(nalu_hypre_AMGDDCompGridF(
                                                               compGrid[level])))[recv_elmt] = buffer[cnt++];
      }
   }

   return nalu_hypre_error_flag;
}
