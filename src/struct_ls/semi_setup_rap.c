/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "pfmg.h"

#define nalu_hypre_MapRAPMarker(indexRAP, rank)      \
   {                                            \
      NALU_HYPRE_Int imacro,jmacro,kmacro;           \
      imacro = nalu_hypre_IndexX(indexRAP);          \
      jmacro = nalu_hypre_IndexY(indexRAP);          \
      kmacro = nalu_hypre_IndexZ(indexRAP);          \
      if (imacro==-1) imacro=2;                 \
      if (jmacro==-1) jmacro=2;                 \
      if (kmacro==-1) kmacro=2;                 \
      rank = imacro + 3*jmacro + 9*kmacro;      \
   }

#define nalu_hypre_InverseMapRAPMarker(rank, indexRAP)       \
   {                                                    \
      NALU_HYPRE_Int imacro,ijmacro,jmacro,kmacro;           \
      ijmacro = (rank%9);                               \
      imacro  = (ijmacro%3);                            \
      jmacro  = (ijmacro-imacro)/3;                     \
      kmacro  = (rank-3*jmacro-imacro)/9;               \
      if (imacro==2) imacro=-1;                         \
      if (jmacro==2) jmacro=-1;                         \
      if (kmacro==2) kmacro=-1;                         \
      nalu_hypre_SetIndex3(indexRAP,imacro,jmacro,kmacro);   \
   }

/*--------------------------------------------------------------------------
 * Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_SemiCreateRAPOp( nalu_hypre_StructMatrix *R,
                       nalu_hypre_StructMatrix *A,
                       nalu_hypre_StructMatrix *P,
                       nalu_hypre_StructGrid   *coarse_grid,
                       NALU_HYPRE_Int           cdir,
                       NALU_HYPRE_Int           P_stored_as_transpose )
{
   nalu_hypre_StructMatrix    *RAP;

   nalu_hypre_Index           *RAP_stencil_shape;
   nalu_hypre_StructStencil   *RAP_stencil;
   NALU_HYPRE_Int              RAP_stencil_size;
   NALU_HYPRE_Int              dim;
   NALU_HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   NALU_HYPRE_Int             *not_cdirs;
   nalu_hypre_StructStencil   *A_stencil;
   NALU_HYPRE_Int              A_stencil_size;
   nalu_hypre_Index           *A_stencil_shape;

   nalu_hypre_Index            indexR;
   nalu_hypre_Index            indexRA;
   nalu_hypre_Index            indexRAP;
   NALU_HYPRE_Int              Rloop, Aloop;

   NALU_HYPRE_Int              j, i;
   NALU_HYPRE_Int              d;
   NALU_HYPRE_Int              stencil_rank;

   NALU_HYPRE_Int             *RAP_marker;
   NALU_HYPRE_Int              RAP_marker_size;
   NALU_HYPRE_Int              RAP_marker_rank;

   A_stencil = nalu_hypre_StructMatrixStencil(A);
   dim = nalu_hypre_StructStencilNDim(A_stencil);
   A_stencil_size = nalu_hypre_StructStencilSize(A_stencil);
   A_stencil_shape = nalu_hypre_StructStencilShape(A_stencil);

   /*-----------------------------------------------------------------------
    * Allocate RAP_marker array used to deternine which offsets are
    * present in RAP. Initialized to zero indicating no offsets present.
    *-----------------------------------------------------------------------*/

   RAP_marker_size = 1;
   for (i = 0; i < dim; i++)
   {
      RAP_marker_size *= 3;
   }
   RAP_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  RAP_marker_size, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    * Define RAP_stencil
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(indexR, 0);
   nalu_hypre_SetIndex(indexRA, 0);
   nalu_hypre_SetIndex(indexRAP, 0);

   stencil_rank = 0;

   /*-----------------------------------------------------------------------
    * Calculate RAP stencil by symbolic computation of triple matrix
    * product RAP. We keep track of index to update RAP_marker.
    *-----------------------------------------------------------------------*/
   for (Rloop = -1; Rloop < 2; Rloop++)
   {
      nalu_hypre_IndexD(indexR, cdir) = Rloop;
      for (Aloop = 0; Aloop < A_stencil_size; Aloop++)
      {
         for (d = 0; d < dim; d++)
         {
            nalu_hypre_IndexD(indexRA, d) = nalu_hypre_IndexD(indexR, d) +
                                       nalu_hypre_IndexD(A_stencil_shape[Aloop], d);
         }

         /*-----------------------------------------------------------------
          * If RA part of the path lands on C point, then P part of path
          * stays at the C point. Divide by 2 to yield to coarse index.
          *-----------------------------------------------------------------*/
         if ((nalu_hypre_IndexD(indexRA, cdir) % 2) == 0)
         {
            nalu_hypre_CopyIndex(indexRA, indexRAP);
            nalu_hypre_IndexD(indexRAP, cdir) /= 2;
            nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;
         }
         /*-----------------------------------------------------------------
          * If RA part of the path lands on F point, then P part of path
          * move +1 and -1 in cdir. Divide by 2 to yield to coarse index.
          *-----------------------------------------------------------------*/
         else
         {
            nalu_hypre_CopyIndex(indexRA, indexRAP);
            nalu_hypre_IndexD(indexRAP, cdir) += 1;
            nalu_hypre_IndexD(indexRAP, cdir) /= 2;
            nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;

            nalu_hypre_CopyIndex(indexRA, indexRAP);
            nalu_hypre_IndexD(indexRAP, cdir) -= 1;
            nalu_hypre_IndexD(indexRAP, cdir) /= 2;
            nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
            RAP_marker[RAP_marker_rank]++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    * For symmetric A, we zero out some entries of RAP_marker to yield
    * the stencil with the proper stored entries.
    * The set S of stored off diagonal entries are such that paths in
    * RAP resulting in a contribution to a entry of S arise only from
    * diagonal entries of A or entries contined in S.
    *
    * In 1d
    * =====
    * cdir = 0
    * (i) in S if
    *    i<0.
    *
    * In 2d
    * =====
    * cdir = 1                 cdir = 0
    * (i,j) in S if          (i,j) in S if
    *      i<0,                     j<0,
    * or   i=0 & j<0.          or   j=0 & i<0.
    *
    * In 3d
    * =====
    * cdir = 2                 cdir = 1                cdir = 0
    * (i,j,k) in S if          (i,j,k) in S if         (i,j,k) in S if
    *      i<0,                     k<0,                    j<0,
    * or   i=0 & j<0,          or   k=0 & i<0,              j=0 & k<0,
    * or   i=j=0 & k<0.        or   k=i=0 & j<0.            j=k=0 & i<0.
    *-----------------------------------------------------------------------*/
   if (nalu_hypre_StructMatrixSymmetric(A))
   {
      if (dim > 1)
      {
         not_cdirs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  dim - 1, NALU_HYPRE_MEMORY_HOST);
      }

      for (d = 1; d < dim; d++)
      {
         not_cdirs[d - 1] = (dim + cdir - d) % dim;
      }

      nalu_hypre_SetIndex(indexRAP, 0);
      nalu_hypre_IndexD(indexRAP, cdir) = 1;
      nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
      RAP_marker[RAP_marker_rank] = 0;

      if (dim > 1)
      {
         nalu_hypre_SetIndex(indexRAP, 0);
         nalu_hypre_IndexD(indexRAP, not_cdirs[0]) = 1;
         for (i = -1; i < 2; i++)
         {
            nalu_hypre_IndexD(indexRAP, cdir) = i;
            nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
            RAP_marker[RAP_marker_rank] = 0;
         }
      }

      if (dim > 2)
      {
         nalu_hypre_SetIndex(indexRAP, 0);
         nalu_hypre_IndexD(indexRAP, not_cdirs[1]) = 1;
         for (i = -1; i < 2; i++)
         {
            nalu_hypre_IndexD(indexRAP, not_cdirs[0]) = i;
            for (j = -1; j < 2; j++)
            {
               nalu_hypre_IndexD(indexRAP, cdir) = j;
               nalu_hypre_MapRAPMarker(indexRAP, RAP_marker_rank);
               RAP_marker[RAP_marker_rank] = 0;

            }
         }
      }

      if (dim > 1)
      {
         nalu_hypre_TFree(not_cdirs, NALU_HYPRE_MEMORY_HOST);
      }
   }

   RAP_stencil_size = 0;

   for (i = 0; i < RAP_marker_size; i++)
   {
      if ( RAP_marker[i] != 0 )
      {
         RAP_stencil_size++;
      }
   }

   RAP_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  RAP_stencil_size, NALU_HYPRE_MEMORY_HOST);

   stencil_rank = 0;
   for (i = 0; i < RAP_marker_size; i++)
   {
      if ( RAP_marker[i] != 0 )
      {
         nalu_hypre_InverseMapRAPMarker(i, RAP_stencil_shape[stencil_rank]);
         stencil_rank++;
      }
   }

   RAP_stencil = nalu_hypre_StructStencilCreate(dim, RAP_stencil_size,
                                           RAP_stencil_shape);
   RAP = nalu_hypre_StructMatrixCreate(nalu_hypre_StructMatrixComm(A),
                                  coarse_grid, RAP_stencil);

   nalu_hypre_StructStencilDestroy(RAP_stencil);

   /*-----------------------------------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------------------------------*/
   nalu_hypre_StructMatrixSymmetric(RAP) = nalu_hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------------------------------
    * Set number of ghost points - one one each boundary
    *-----------------------------------------------------------------------*/
   nalu_hypre_StructMatrixSetNumGhost(RAP, RAP_num_ghost);

   nalu_hypre_TFree(RAP_marker, NALU_HYPRE_MEMORY_HOST);

   return RAP;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiBuildRAP( nalu_hypre_StructMatrix *A,
                    nalu_hypre_StructMatrix *P,
                    nalu_hypre_StructMatrix *R,
                    NALU_HYPRE_Int           cdir,
                    nalu_hypre_Index         cindex,
                    nalu_hypre_Index         cstride,
                    NALU_HYPRE_Int           P_stored_as_transpose,
                    nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;

   nalu_hypre_StructStencil  *coarse_stencil;
   NALU_HYPRE_Int             coarse_stencil_size;
   nalu_hypre_Index          *coarse_stencil_shape;
   NALU_HYPRE_Int            *coarse_symm_elements;

   nalu_hypre_StructGrid     *fgrid;
   NALU_HYPRE_Int            *fgrid_ids;
   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   NALU_HYPRE_Int            *cgrid_ids;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           stridec;
   nalu_hypre_Index           fstart;
   nalu_hypre_IndexRef        stridef;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             fi, ci;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *P_dbox;
   nalu_hypre_Box            *R_dbox;
   nalu_hypre_Box            *RAP_dbox;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_ptr;

   NALU_HYPRE_Real           *rap_ptrS, *rap_ptrU, *rap_ptrD;

   NALU_HYPRE_Int             symm_path_multiplier;

   NALU_HYPRE_Int             COffsetA;
   NALU_HYPRE_Int             COffsetP;
   NALU_HYPRE_Int             AOffsetP;

   NALU_HYPRE_Int             RAPloop;
   NALU_HYPRE_Int             diag;
   NALU_HYPRE_Int             dim;
   NALU_HYPRE_Int             d;

   NALU_HYPRE_Real            zero = 0.0;

   coarse_stencil = nalu_hypre_StructMatrixStencil(RAP);
   coarse_stencil_size = nalu_hypre_StructStencilSize(coarse_stencil);
   coarse_symm_elements = nalu_hypre_StructMatrixSymmElements(RAP);
   coarse_stencil_shape = nalu_hypre_StructStencilShape(coarse_stencil);
   dim = nalu_hypre_StructStencilNDim(coarse_stencil);

   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = nalu_hypre_StructMatrixGrid(A);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

   /*-----------------------------------------------------------------
    *  Loop over boxes to compute entries of RAP
    *-----------------------------------------------------------------*/

   fi = 0;
   nalu_hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

      cstart = nalu_hypre_BoxIMin(cgrid_box);
      nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);
      nalu_hypre_BoxGetSize(cgrid_box, loop_size);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
      P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
      R_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R), fi);
      RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

      /*-----------------------------------------------------------------
       * Extract pointers for interpolation operator:
       * pa is pointer for weight for f-point above c-point
       * pb is pointer for weight for f-point below c-point
       *
       *   pa  "down"                      pb "up"
       *
       *                                     C
       *
       *                                     |
       *                                     v
       *
       *       F                             F
       *
       *       ^
       *       |
       *
       *       C
       *
       *-----------------------------------------------------------------*/

      nalu_hypre_SetIndex(index, 0);
      //RL:PTROFFSET
      NALU_HYPRE_Int pb_offset = 0;
      if (P_stored_as_transpose)
      {
         nalu_hypre_IndexD(index, cdir) = 1;
         pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

         nalu_hypre_IndexD(index, cdir) = -1;
         pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
      }
      else
      {
         nalu_hypre_IndexD(index, cdir) = -1;
         pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

         nalu_hypre_IndexD(index, cdir) = 1;
         pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
         pb_offset = -nalu_hypre_BoxOffsetDistance(P_dbox, index);
      }

      /*-----------------------------------------------------------------
       * Extract pointers for restriction operator:
       * ra is pointer for weight for f-point above c-point
       * rb is pointer for weight for f-point below c-point
       *
       *   rb  "down"                      ra "up"
       *
       *                                     F
       *
       *                                     |
       *                                     v
       *
       *       C                             C
       *
       *       ^
       *       |
       *
       *       F
       *
       *-----------------------------------------------------------------*/

      nalu_hypre_SetIndex(index, 0);
      NALU_HYPRE_Int rb_offset = 0;
      if (P_stored_as_transpose)
      {
         nalu_hypre_IndexD(index, cdir) = 1;
         ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

         nalu_hypre_IndexD(index, cdir) = -1;
         rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
      }
      else
      {
         nalu_hypre_IndexD(index, cdir) = -1;
         ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

         nalu_hypre_IndexD(index, cdir) = 1;
         rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
         rb_offset = -nalu_hypre_BoxOffsetDistance(P_dbox, index);
      }

      /*-----------------------------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoops below I assume iA and iP refer to data associated
       * with the point which we are building the stencil for. The below
       * Offsets (and those defined later in the switch statement) are
       * used in refering to data associated with other points.
       *-----------------------------------------------------------------*/

      nalu_hypre_SetIndex(index, 0);
      nalu_hypre_IndexD(index, cdir) = 1;
      COffsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);
      COffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);

      /*-----------------------------------------------------------------
       * Entries in RAP are calculated by accumulation, must first
       * zero out entries.
       *-----------------------------------------------------------------*/

      for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
      {
         if (coarse_symm_elements[RAPloop] == -1)
         {
            rap_ptrS = nalu_hypre_StructMatrixBoxData(RAP, ci, RAPloop);
#define DEVICE_VAR is_device_ptr(rap_ptrS)
            nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                RAP_dbox, cstart, stridec, iAc);
            {
               rap_ptrS[iAc] = zero;
            }
            nalu_hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }
      }

      /*-----------------------------------------------------------------
       * Computational loop. Written as a loop over stored entries of
       * RAP. We then get the pointer (a_ptr) for the same index in A.
       * If it exists, we then calculate all RAP paths involving this
       * entry of A.
       *-----------------------------------------------------------------*/
      for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
      {
         if (coarse_symm_elements[RAPloop] == -1)
         {
            /*-------------------------------------------------------------
             * Get pointer for A that corresponds to the current RAP index.
             * If pointer is non-null, i.e. there is a corresponding entry
             * in A, compute paths.
             *-------------------------------------------------------------*/
            nalu_hypre_CopyIndex(coarse_stencil_shape[RAPloop], index);
            a_ptr = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);
            if (a_ptr != NULL)
            {
               switch (nalu_hypre_IndexD(index, cdir))
               {
                  /*-----------------------------------------------------
                   * If A stencil index is 0 in coarsened direction, need
                   * to calculate (r,p) pairs (stay,stay) (up,up) (up,down)
                   * (down,up) and (down,down). Paths 1,3 & 4 {(s,s),(u,d),
                   * (d,u)} yield contributions to RAP with the same stencil
                   * index as A. Path 2 (u,u) contributes to RAP with
                   * index +1 in coarsened direction. Path 5 (d,d)
                   * contributes to RAP with index -1 in coarsened
                   * direction.
                   *-----------------------------------------------------*/

                  case 0:

                     nalu_hypre_IndexD(index, cdir) = 1;
                     rap_ptrU = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     nalu_hypre_IndexD(index, cdir) = -1;
                     rap_ptrD = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     nalu_hypre_IndexD(index, cdir) = 0;
                     AOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
                     rap_ptrS = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     diag = 0;
                     for (d = 0; d < dim; d++)
                     {
                        diag += nalu_hypre_IndexD(index, d) * nalu_hypre_IndexD(index, d);
                     }

                     if (diag == 0 && nalu_hypre_StructMatrixSymmetric(RAP))
                     {
                        /*--------------------------------------------------
                         * If A stencil index is (0,0,0) and RAP is symmetric,
                         * must not calculate (up,up) path. It's symmetric
                         * to the (down,down) path and calculating both paths
                         * incorrectly doubles the contribution. Additionally
                         * the (up,up) path contributes to a non-stored entry
                         * in RAP.
                         *--------------------------------------------------*/
#define DEVICE_VAR is_device_ptr(rap_ptrS,a_ptr,ra,pa,rb,pb,rap_ptrD)
                        nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                            P_dbox, cstart, stridec, iP,
                                            R_dbox, cstart, stridec, iR,
                                            A_dbox, fstart, stridef, iA,
                                            RAP_dbox, cstart, stridec, iAc);
                        {
                           NALU_HYPRE_Int iAp, iPp;
                           /* path 1 : (stay,stay) */
                           rap_ptrS[iAc] +=          a_ptr[iA]           ;

                           /* path 2 : (up,up) */

                           /* path 3 : (up,down) */
                           iAp = iA + COffsetA;
                           iPp = iP + AOffsetP;
                           rap_ptrS[iAc] += ra[iR] * a_ptr[iAp] * pa[iPp];

                           /* path 4 : (down,up) */
                           iAp = iA - COffsetA;
                           rap_ptrS[iAc] += rb[iR + rb_offset] * a_ptr[iAp] * pb[iPp + pb_offset];

                           /* path 5 : (down,down) */
                           iPp = iP - COffsetP + AOffsetP;
                           rap_ptrD[iAc] += rb[iR + rb_offset] * a_ptr[iAp] * pa[iPp];
                        }
                        nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
                     }
                     else
                     {
                        /*--------------------------------------------------
                         * If A stencil index is not (0,0,0) or RAP is
                         * nonsymmetric, all 5 paths are calculated.
                         *--------------------------------------------------*/
#define DEVICE_VAR is_device_ptr(rap_ptrS,a_ptr,rap_ptrU,ra,pb,pa,rb,rap_ptrD)
                        nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                            P_dbox, cstart, stridec, iP,
                                            R_dbox, cstart, stridec, iR,
                                            A_dbox, fstart, stridef, iA,
                                            RAP_dbox, cstart, stridec, iAc);
                        {
                           NALU_HYPRE_Int iAp, iPp;
                           /* path 1 : (stay,stay) */
                           rap_ptrS[iAc] +=          a_ptr[iA]           ;

                           /* path 2 : (up,up) */
                           iAp = iA + COffsetA;
                           iPp = iP + COffsetP + AOffsetP;
                           rap_ptrU[iAc] += ra[iR] * a_ptr[iAp] * pb[iPp + pb_offset];

                           /* path 3 : (up,down) */
                           iPp = iP + AOffsetP;
                           rap_ptrS[iAc] += ra[iR] * a_ptr[iAp] * pa[iPp];

                           /* path 4 : (down,up) */
                           iAp = iA - COffsetA;
                           rap_ptrS[iAc] += rb[iR + rb_offset] * a_ptr[iAp] * pb[iPp + pb_offset];

                           /* path 5 : (down,down) */
                           iPp = iP - COffsetP + AOffsetP;
                           rap_ptrD[iAc] += rb[iR + rb_offset] * a_ptr[iAp] * pa[iPp];
                        }
                        nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
                     }

                     break;

                  /*-----------------------------------------------------
                   * If A stencil index is -1 in coarsened direction, need
                   * to calculate (r,p) pairs (stay,up) (stay,down) (up,stay)
                   * and (down,stay). Paths 2 & 4 {(s,d),(d,s)} contribute
                   * to RAP with same stencil index as A. Paths 1 & 3
                   * {(s,u),(u,s)} contribute to RAP with index 0 in
                   * coarsened direction.
                   *-----------------------------------------------------*/

                  case -1:

                     rap_ptrD = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     nalu_hypre_IndexD(index, cdir) = 0;
                     AOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
                     rap_ptrS = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);

                     /*--------------------------------------------------
                      * If A stencil index is zero except in coarsened
                      * dirction and RAP is symmetric, must calculate
                      * symmetric paths for (stay,up) and (up,stay).
                      * These contribute to the diagonal entry of RAP.
                      * These additional paths have the same numerical
                      * contribution as the calculated path. We multiply
                      * by two to account for them.
                      *--------------------------------------------------*/
                     symm_path_multiplier = 1;
                     diag = 0;
                     for (d = 0; d < dim; d++)
                     {
                        diag += nalu_hypre_IndexD(index, d) * nalu_hypre_IndexD(index, d);
                     }
                     if (diag == 0 && nalu_hypre_StructMatrixSymmetric(RAP))
                     {
                        symm_path_multiplier = 2;
                     }

#define DEVICE_VAR is_device_ptr(rap_ptrS,a_ptr,pb,rap_ptrD,pa,ra,rb)
                     nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                         P_dbox, cstart, stridec, iP,
                                         R_dbox, cstart, stridec, iR,
                                         A_dbox, fstart, stridef, iA,
                                         RAP_dbox, cstart, stridec, iAc);
                     {
                        NALU_HYPRE_Int iAp, iPp;
                        /* Path 1 : (stay,up) & symmetric path  */
                        iPp = iP + AOffsetP;
                        rap_ptrS[iAc] += symm_path_multiplier *
                                         (a_ptr[iA]  * pb[iPp + pb_offset]);

                        /* Path 2 : (stay,down) */
                        iPp = iP - COffsetP + AOffsetP;
                        rap_ptrD[iAc] +=          a_ptr[iA]  * pa[iPp];

                        /* Path 3 : (up,stay) */
                        iAp = iA + COffsetA;
                        rap_ptrS[iAc] += symm_path_multiplier *
                                         (ra[iR] * a_ptr[iAp]          );

                        /* Path 4 : (down,stay) */
                        iAp = iA - COffsetA;
                        rap_ptrD[iAc] += rb[iR + rb_offset] * a_ptr[iAp]          ;
                     }
                     nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

                     break;

                  /*-----------------------------------------------------
                   * If A stencil index is +1 in coarsened direction, need
                   * to calculate (r,p) pairs (stay,up) (stay,down) (up,stay)
                   * and (down,stay). Paths 1 & 3 {(s,u),(u,s)} contribute
                   * to RAP with same stencil index as A. Paths 2 & 4
                   * {(s,d),(d,s)} contribute to RAP with index 0 in
                   * coarsened direction.
                   *-----------------------------------------------------*/

                  case 1:

                     rap_ptrU = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     nalu_hypre_IndexD(index, cdir) = 0;
                     AOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
                     rap_ptrS = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     /*--------------------------------------------------
                      * If A stencil index is zero except in coarsened
                      * dirction and RAP is symmetric, must calculate
                      * symmetric paths for (stay,down) and (down,stay).
                      * These contribute to the diagonal entry of RAP.
                      * These additional paths have the same numerical
                      * contribution as the calculated path. We multiply
                      * by two to account for them.
                      *--------------------------------------------------*/
                     symm_path_multiplier = 1;
                     diag = 0;
                     for (d = 0; d < dim; d++)
                     {
                        diag += nalu_hypre_IndexD(index, d) * nalu_hypre_IndexD(index, d);
                     }
                     if (diag == 0 && nalu_hypre_StructMatrixSymmetric(RAP))
                     {
                        symm_path_multiplier = 2;
                     }

#define DEVICE_VAR is_device_ptr(rap_ptrU,a_ptr,pb,rap_ptrS,pa,ra,rb)
                     nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                         P_dbox, cstart, stridec, iP,
                                         R_dbox, cstart, stridec, iR,
                                         A_dbox, fstart, stridef, iA,
                                         RAP_dbox, cstart, stridec, iAc);
                     {
                        NALU_HYPRE_Int iAp, iPp;
                        /* Path 1 : (stay,up) */
                        iPp = iP + COffsetP + AOffsetP;
                        rap_ptrU[iAc] +=          a_ptr[iA]  * pb[iPp + pb_offset];

                        /* Path 2 : (stay,down) */
                        iPp = iP + AOffsetP;
                        rap_ptrS[iAc] += symm_path_multiplier *
                                         (a_ptr[iA]  * pa[iPp]);

                        /* Path 3 : (up,stay) */
                        iAp = iA + COffsetA;
                        rap_ptrU[iAc] += ra[iR] * a_ptr[iAp]          ;

                        /* Path 4 : (down,stay) */
                        iAp = iA - COffsetA;
                        rap_ptrS[iAc] += symm_path_multiplier *
                                         (rb[iR + rb_offset] * a_ptr[iAp]          );
                     }
                     nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR

                     break;
               } /* end of switch */

            } /* end of if a_ptr != NULL */

         } /* end if coarse_symm_element == -1 */

      } /* end of RAPloop */

   } /* end ForBoxI */

   /*-----------------------------------------------------------------
    *  Loop over boxes to collapse entries of RAP when period = 1 in
    *  the coarsened direction.
    *-----------------------------------------------------------------*/

   if (nalu_hypre_IndexD(nalu_hypre_StructGridPeriodic(cgrid), cdir) == 1)
   {
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = nalu_hypre_BoxIMin(cgrid_box);
         nalu_hypre_BoxGetSize(cgrid_box, loop_size);

         RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

         /*--------------------------------------------------------------
          * Computational loop. A loop over stored entries of RAP.
          *-------------------------------------------------------------*/
         for (RAPloop = 0; RAPloop < coarse_stencil_size; RAPloop++)
         {
            if (coarse_symm_elements[RAPloop] == -1)
            {
               nalu_hypre_CopyIndex(coarse_stencil_shape[RAPloop], index);
               switch (nalu_hypre_IndexD(index, cdir))
               {
                  /*-----------------------------------------------------
                   * If RAP stencil index is 0 in coarsened direction,
                   * leave entry unchanged.
                   *-----------------------------------------------------*/

                  case 0:

                     break;

                  /*-----------------------------------------------------
                   * If RAP stencil index is +/-1 in coarsened direction,
                   * to add entry to cooresponding entry with 0 in the
                   * coarsened direction. Also zero out current index.
                   *-----------------------------------------------------*/

                  default:

                     /*---------------------------------------------------------
                      * Get pointer to the current RAP index (rap_ptrD)
                      * and cooresponding index with 0 in the coarsened
                      * direction (rap_ptrS).
                      *---------------------------------------------------------*/
                     rap_ptrD = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);
                     nalu_hypre_IndexD(index, cdir) = 0;
                     rap_ptrS = nalu_hypre_StructMatrixExtractPointerByIndex(RAP,
                                                                        ci, index);

                     /*--------------------------------------------------
                      * If RAP stencil index is zero except in coarsened
                      * direction and RAP is symmetric, must
                      * NALU_HYPRE_Real entry when modifying the diagonal.
                      *--------------------------------------------------*/
                     symm_path_multiplier = 1;
                     diag = 0;
                     for (d = 0; d < dim; d++)
                     {
                        diag += nalu_hypre_IndexD(index, d) * nalu_hypre_IndexD(index, d);
                     }
                     if (diag == 0 && nalu_hypre_StructMatrixSymmetric(RAP))
                     {
                        symm_path_multiplier = 2;
                     }
#define DEVICE_VAR is_device_ptr(rap_ptrS,rap_ptrD)
                     nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                         RAP_dbox, cstart, stridec, iAc);
                     {
                        rap_ptrS[iAc] += symm_path_multiplier *
                                         (rap_ptrD[iAc]);

                        rap_ptrD[iAc] = zero;
                     }
                     nalu_hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR

                     break;

               } /* end of switch */

            } /* end if coarse_symm_element == -1 */

         } /* end of RAPloop */

      } /* end ForBoxI */

   } /* if periodic */

   return nalu_hypre_error_flag;
}
