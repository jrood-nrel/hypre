/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * This routine serves as an alternative to the MatrixAssemble routine for the
 * semi interpolation and restriction operators.  It allows us to avoid having
 * to deal with zero boxes when figuring out communications patterns.
 *
 * The issue arises in the following scenario for process p.  In the diagram,
 * process p only owns grid points denoted by '|' and not those denoted by ':'.
 * The center of the stencil is represented by an 'x'.
 *
 *    x----> <----x----> <----x   stencil coeffs needed for P^T
 *     <----x----> <----x---->    stencil coeffs needed for P
 *     <----x<--->x<--->x---->    stencil coeffs needed for A
 *
 *    :-----:-----|-----:-----:   fine grid
 *    :-----------|-----------:   coarse grid
 *
 *     <----------x---------->    stencil coeffs to be computed for RAP
 *
 * The issue is with the grid for P, which is empty on process p.  Previously,
 * we added ghost zones to get the appropriate neighbor data, and we did this
 * even for zero boxes.  Unfortunately, dealing with zero boxes is a major pain,
 * so the below routine eliminates the need for handling zero boxes when
 * computing communication information.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructInterpAssemble( nalu_hypre_StructMatrix  *A,
                            nalu_hypre_StructMatrix  *P,
                            NALU_HYPRE_Int            P_stored_as_transpose,
                            NALU_HYPRE_Int            cdir,
                            nalu_hypre_Index          index,
                            nalu_hypre_Index          stride )
{
   nalu_hypre_StructGrid     *grid = nalu_hypre_StructMatrixGrid(A);

   nalu_hypre_BoxArrayArray  *box_aa;
   nalu_hypre_BoxArray       *box_a;
   nalu_hypre_Box            *box;

   nalu_hypre_CommInfo       *comm_info;
   nalu_hypre_CommPkg        *comm_pkg;
   nalu_hypre_CommHandle     *comm_handle;

   NALU_HYPRE_Int             num_ghost[] = {0, 0, 0, 0, 0, 0};
   NALU_HYPRE_Int             i, j, s, dim;

   if (nalu_hypre_StructMatrixConstantCoefficient(P) != 0)
   {
      return nalu_hypre_error_flag;
   }

   /* set num_ghost */
   dim = nalu_hypre_StructGridNDim(grid);
   for (j = 0; j < dim; j++)
   {
      num_ghost[2 * j]   = 1;
      num_ghost[2 * j + 1] = 1;
   }
   if (P_stored_as_transpose)
   {
      num_ghost[2 * cdir]   = 2;
      num_ghost[2 * cdir + 1] = 2;
   }

   /* comm_info <-- From fine grid grown by num_ghost */

   nalu_hypre_CreateCommInfoFromNumGhost(grid, num_ghost, &comm_info);

   /* Project and map comm_info onto coarsened index space */

   nalu_hypre_CommInfoProjectSend(comm_info, index, stride);
   nalu_hypre_CommInfoProjectRecv(comm_info, index, stride);

   for (s = 0; s < 4; s++)
   {
      switch (s)
      {
         case 0:
            box_aa = nalu_hypre_CommInfoSendBoxes(comm_info);
            nalu_hypre_SetIndex3(nalu_hypre_CommInfoSendStride(comm_info), 1, 1, 1);
            break;

         case 1:
            box_aa = nalu_hypre_CommInfoRecvBoxes(comm_info);
            nalu_hypre_SetIndex3(nalu_hypre_CommInfoRecvStride(comm_info), 1, 1, 1);
            break;

         case 2:
            box_aa = nalu_hypre_CommInfoSendRBoxes(comm_info);
            break;

         case 3:
            box_aa = nalu_hypre_CommInfoRecvRBoxes(comm_info);
            break;
      }

      nalu_hypre_ForBoxArrayI(j, box_aa)
      {
         box_a = nalu_hypre_BoxArrayArrayBoxArray(box_aa, j);
         nalu_hypre_ForBoxI(i, box_a)
         {
            box = nalu_hypre_BoxArrayBox(box_a, i);
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(box), index, stride,
                                        nalu_hypre_BoxIMin(box));
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(box), index, stride,
                                        nalu_hypre_BoxIMax(box));
         }
      }
   }

   comm_pkg = nalu_hypre_StructMatrixCommPkg(P);
   if (comm_pkg)
   {
      nalu_hypre_CommPkgDestroy(comm_pkg);
   }

   nalu_hypre_CommPkgCreate(comm_info,
                       nalu_hypre_StructMatrixDataSpace(P),
                       nalu_hypre_StructMatrixDataSpace(P),
                       nalu_hypre_StructMatrixNumValues(P), NULL, 0,
                       nalu_hypre_StructMatrixComm(P),
                       &comm_pkg);
   nalu_hypre_CommInfoDestroy(comm_info);
   nalu_hypre_StructMatrixCommPkg(P) = comm_pkg;

   nalu_hypre_InitializeCommunication(comm_pkg,
                                 nalu_hypre_StructMatrixStencilData(P)[0],//nalu_hypre_StructMatrixData(P),
                                 nalu_hypre_StructMatrixStencilData(P)[0],//nalu_hypre_StructMatrixData(P),
                                 0, 0,
                                 &comm_handle);
   nalu_hypre_FinalizeCommunication(comm_handle);

   return nalu_hypre_error_flag;
}
