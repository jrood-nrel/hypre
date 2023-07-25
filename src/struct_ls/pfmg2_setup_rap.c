/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * Macro to "change coordinates".  This routine is written as though
 * coarsening is being done in the y-direction.  This macro is used to
 * allow for coarsening to be done in the x-direction also.
 *--------------------------------------------------------------------------*/

#define MapIndex(in_index, cdir, out_index)                     \
   nalu_hypre_IndexD(out_index, 2)    = nalu_hypre_IndexD(in_index, 2);   \
   nalu_hypre_IndexD(out_index, cdir) = nalu_hypre_IndexD(in_index, 1);   \
   cdir = (cdir + 1) % 2;                                       \
   nalu_hypre_IndexD(out_index, cdir) = nalu_hypre_IndexD(in_index, 0);   \
   cdir = (cdir + 1) % 2;

/*--------------------------------------------------------------------------
 * Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_PFMG2CreateRAPOp( nalu_hypre_StructMatrix *R,
                        nalu_hypre_StructMatrix *A,
                        nalu_hypre_StructMatrix *P,
                        nalu_hypre_StructGrid   *coarse_grid,
                        NALU_HYPRE_Int           cdir        )
{
   nalu_hypre_StructMatrix    *RAP;

   nalu_hypre_Index           *RAP_stencil_shape;
   nalu_hypre_StructStencil   *RAP_stencil;
   NALU_HYPRE_Int              RAP_stencil_size;
   NALU_HYPRE_Int              RAP_stencil_dim;
   NALU_HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   nalu_hypre_Index            index_temp;
   NALU_HYPRE_Int              j, i;
   NALU_HYPRE_Int              stencil_rank;

   RAP_stencil_dim = 2;

   /*-----------------------------------------------------------------------
    * Define RAP_stencil
    *-----------------------------------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------------------------------
    * non-symmetric case
    *-----------------------------------------------------------------------*/

   if (!nalu_hypre_StructMatrixSymmetric(A))
   {

      /*--------------------------------------------------------------------
       * 5 or 9 point fine grid stencil produces 9 point RAP
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 9;
      RAP_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  RAP_stencil_size, NALU_HYPRE_MEMORY_HOST);
      for (j = -1; j < 2; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Storage for 9 elements (c,w,e,n,s,sw,se,nw,se)
             *--------------------------------------------------------------*/
            nalu_hypre_SetIndex3(index_temp, i, j, 0);
            MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
            stencil_rank++;
         }
      }
   }

   /*-----------------------------------------------------------------------
    * symmetric case
    *-----------------------------------------------------------------------*/

   else
   {

      /*--------------------------------------------------------------------
       * 5 or 9 point fine grid stencil produces 9 point RAP
       * Only store the lower triangular part + diagonal = 5 entries,
       * lower triangular means the lower triangular part on the matrix
       * in the standard lexicographic ordering.
       *--------------------------------------------------------------------*/
      RAP_stencil_size = 5;
      RAP_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  RAP_stencil_size, NALU_HYPRE_MEMORY_HOST);
      for (j = -1; j < 1; j++)
      {
         for (i = -1; i < 2; i++)
         {

            /*--------------------------------------------------------------
             * Store 5 elements in (c,w,s,sw,se)
             *--------------------------------------------------------------*/
            if ( i + j <= 0 )
            {
               nalu_hypre_SetIndex3(index_temp, i, j, 0);
               MapIndex(index_temp, cdir, RAP_stencil_shape[stencil_rank]);
               stencil_rank++;
            }
         }
      }
   }

   RAP_stencil = nalu_hypre_StructStencilCreate(RAP_stencil_dim, RAP_stencil_size,
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

   return RAP;
}

/*--------------------------------------------------------------------------
 * Routines to build RAP. These routines are fairly general
 *  1) No assumptions about symmetry of A
 *  2) No assumption that R = transpose(P)
 *  3) 5 or 9-point fine grid A
 *
 * I am, however, assuming that the c-to-c interpolation is the identity.
 *
 * I've written two routines - nalu_hypre_PFMG2BuildRAPSym to build the
 * lower triangular part of RAP (including the diagonal) and
 * nalu_hypre_PFMG2BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the
 * first routine would be called. With full storage both would need to
 * be called.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPSym( nalu_hypre_StructMatrix *A,
                        nalu_hypre_StructMatrix *P,
                        nalu_hypre_StructMatrix *R,
                        NALU_HYPRE_Int           cdir,
                        nalu_hypre_Index         cindex,
                        nalu_hypre_Index         cstride,
                        nalu_hypre_StructMatrix *RAP     )
{
   nalu_hypre_StructStencil  *fine_stencil;
   NALU_HYPRE_Int             fine_stencil_size;

   nalu_hypre_StructGrid     *fgrid;
   NALU_HYPRE_Int            *fgrid_ids;
   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   NALU_HYPRE_Int            *cgrid_ids;

   NALU_HYPRE_Int             constant_coefficient;
   NALU_HYPRE_Int             constant_coefficient_A;
   NALU_HYPRE_Int             fi, ci;

   fine_stencil = nalu_hypre_StructMatrixStencil(A);
   fine_stencil_size = nalu_hypre_StructStencilSize(fine_stencil);

   fgrid = nalu_hypre_StructMatrixGrid(A);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(RAP);
   constant_coefficient_A = nalu_hypre_StructMatrixConstantCoefficient(A);
   nalu_hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(R) == constant_coefficient );
   nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(P) == constant_coefficient );
   if (constant_coefficient == 1 )
   {
      nalu_hypre_assert( constant_coefficient_A == 1 );
   }
   else
   {
      nalu_hypre_assert( constant_coefficient_A == 0 || constant_coefficient_A == 2 );
   }

   fi = 0;
   nalu_hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      /*-----------------------------------------------------------------
       * Switch statement to direct control to apropriate BoxLoop depending
       * on stencil size. Default is full 9-point.
       *-----------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for symmetric 5-point fine grid operator; produces a
          * symmetric 9-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (southwest, south, southeast,
          * west, and center).
          *--------------------------------------------------------------*/

         case 5:

            if ( constant_coefficient == 1 )
            {
               nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 9-point fine grid operator; produces a
          * symmetric 9-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (southwest, south, southeast,
          * west, and center).
          *--------------------------------------------------------------*/

         default:

            if ( constant_coefficient == 1 )
            {
               nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 5, constant coefficient 0 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC0(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           stridec;
   nalu_hypre_Index           fstart;
   nalu_hypre_IndexRef        stridef;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             constant_coefficient_A;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *P_dbox;
   nalu_hypre_Box            *R_dbox;
   nalu_hypre_Box            *RAP_dbox;
   nalu_hypre_Box            *cgrid_box;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   NALU_HYPRE_Real            a_cw_offd, a_cw_offdm1, a_cw_offdp1, a_ce_offdm1;
   NALU_HYPRE_Real            a_cs_offd, a_cs_offdm1, a_cs_offdp1, a_cn_offd, a_cn_offdm1;
   NALU_HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   NALU_HYPRE_Real           *rap_csw, *rap_cse;

   NALU_HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   NALU_HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = nalu_hypre_StructMatrixConstantCoefficient(A);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
   P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
   R_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int pbOffset = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int rbOffset = nalu_hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);


   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      yOffsetA_offd = 0;
      yOffsetA_diag = nalu_hypre_BoxOffsetDistance(A_dbox, index);
   }

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);


   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   nalu_hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_csw,rb,a_cw,pa,rap_cs,a_cc,a_cs,rap_cse,a_ce,rap_cw,pb,ra,rap_cc,a_cn)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA;

         NALU_HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] = rb[iR - rbOffset] * a_cw[iAm1] * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] = rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_cs[iAm1]
                       +                 a_cs[iA] * pa[iP1];
         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] = rb[iR - rbOffset] * a_ce[iAm1] * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw[iA]
                                +          rb[iR - rbOffset] * a_cw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw[iAp1] * pa[iP1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_cn[iAm1]
                                +          ra[iR] * a_cs[iAp1]
                                +                   a_cs[iA]   * pb[iP - pbOffset]
                                +                   a_cn[iA]   * pa[iP];
      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_ce_offdm1 = a_ce[iA_offdm1];

#define DEVICE_VAR is_device_ptr(rap_csw,rb,pa,rap_cs,a_cc,rap_cse,rap_cw,pb,ra,rap_cc)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA_diag;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA_diag;

         NALU_HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] = rb[iR - rbOffset] * a_cw_offdm1 * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] = rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_cs_offdm1
                       +                   a_cs_offd   * pa[iP1];

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] = rb[iR - rbOffset] * a_ce_offdm1 * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw_offd
                                +          rb[iR - rbOffset] * a_cw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw_offdp1 * pa[iP1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_cn_offdm1
                                +          ra[iR] * a_cs_offdp1
                                +                   a_cs_offd  * pb[iP - pbOffset]
                                +                   a_cn_offd  * pa[iP];
      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      } *//* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 5, constant coefficient 1 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPSym_onebox_FSS5_CC1(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           fstart;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;

   NALU_HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   NALU_HYPRE_Real           *rap_csw, *rap_cse;

   NALU_HYPRE_Int             iA, iAm1, iAp1;
   NALU_HYPRE_Int             iAc;
   NALU_HYPRE_Int             iP, iP1;
   NALU_HYPRE_Int             iR;
   NALU_HYPRE_Int             yOffsetA;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetA = 0;
   yOffsetP = 0;

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 9-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 5-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                 +          rb[iR] * a_cs[iAm1]
                 +                   a_cs[iA]   * pa[iP1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
                          +          rb[iR] * a_cw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cw[iAp1] * pa[iP1];

   rap_cc[iAc] =          a_cc[iA]
                          +          rb[iR] * a_cc[iAm1] * pb[iP]
                          +          ra[iR] * a_cc[iAp1] * pa[iP]
                          +          rb[iR] * a_cn[iAm1]
                          +          ra[iR] * a_cs[iAp1]
                          +                   a_cs[iA]   * pb[iP]
                          +                   a_cn[iA]   * pa[iP];

   /*      } *//* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 9, constant coefficient 0 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC0(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           stridec;
   nalu_hypre_Index           fstart;
   nalu_hypre_IndexRef        stridef;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             constant_coefficient_A;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *P_dbox;
   nalu_hypre_Box            *R_dbox;
   nalu_hypre_Box            *RAP_dbox;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   NALU_HYPRE_Real           *a_csw, *a_cse, *a_cnw;
   NALU_HYPRE_Real            a_cw_offd, a_cw_offdm1, a_cw_offdp1, a_ce_offdm1;
   NALU_HYPRE_Real            a_cs_offd, a_cs_offdm1, a_cs_offdp1, a_cn_offd, a_cn_offdm1;
   NALU_HYPRE_Real            a_csw_offd, a_csw_offdm1, a_csw_offdp1, a_cse_offd, a_cse_offdm1;
   NALU_HYPRE_Real            a_cnw_offd, a_cnw_offdm1;

   NALU_HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   NALU_HYPRE_Real           *rap_csw, *rap_cse;

   NALU_HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   NALU_HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = nalu_hypre_StructMatrixConstantCoefficient(A);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
   P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
   R_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int pbOffset = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   NALU_HYPRE_Int rbOffset = nalu_hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      yOffsetA_offd = 0;
      yOffsetA_diag = 0;
   }

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*--------------------------------------------------------------
    * Loop for symmetric 9-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   nalu_hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_csw,rb,a_cw,pa,a_csw,rap_cs,a_cc,a_cs,rap_cse,a_ce,a_cse,rap_cw,pb,ra,a_cnw,rap_cc,a_cn)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA;

         NALU_HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] = rb[iR - rbOffset] * a_cw[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_csw[iAm1]
                        +                    a_csw[iA]  * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] = rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_cs[iAm1]
                       +                   a_cs[iA]   * pa[iP1];

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] = rb[iR - rbOffset] * a_ce[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_cse[iAm1]
                        +                    a_cse[iA]  * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw[iA]
                                +          rb[iR - rbOffset] * a_cw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_cnw[iAm1]
                                +          ra[iR] * a_csw[iAp1]
                                +                   a_csw[iA]  * pb[iP1 - pbOffset]
                                +                   a_cnw[iA]  * pa[iP1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_cn[iAm1]
                                +          ra[iR] * a_cs[iAp1]
                                +                   a_cs[iA]   * pb[iP - pbOffset]
                                +                   a_cn[iA]   * pa[iP];

      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_csw_offd = a_csw[iA_offd];
      a_csw_offdm1 = a_csw[iA_offdm1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdm1 = a_cse[iA_offdm1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdm1 = a_cnw[iA_offdm1];

#define DEVICE_VAR is_device_ptr(rap_csw,rb,pa,rap_cs,a_cc,rap_cse,rap_cw,pb,ra,rap_cc)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA_diag;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA_diag;

         NALU_HYPRE_Int iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] = rb[iR - rbOffset] * a_cw_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_csw_offdm1
                        +                    a_csw_offd  * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] = rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_cs_offdm1
                       +                   a_cs_offd   * pa[iP1];

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] = rb[iR - rbOffset] * a_ce_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_cse_offdm1
                        +                    a_cse_offd  * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw_offd
                                +          rb[iR - rbOffset] * a_cw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw_offdp1 * pa[iP1]
                                +          rb[iR - rbOffset] * a_cnw_offdm1
                                +          ra[iR] * a_csw_offdp1
                                +                   a_csw_offd  * pb[iP1 - pbOffset]
                                +                   a_cnw_offd  * pa[iP1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_cn_offdm1
                                +          ra[iR] * a_cs_offdp1
                                +                   a_cs_offd   * pb[iP - pbOffset]
                                +                   a_cn_offd   * pa[iP];

      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 9, constant coefficient 1 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPSym_onebox_FSS9_CC1(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           fstart;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   NALU_HYPRE_Real           *a_csw, *a_cse, *a_cnw;
   NALU_HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   NALU_HYPRE_Real           *rap_csw, *rap_cse;

   NALU_HYPRE_Int             iA, iAm1, iAp1;
   NALU_HYPRE_Int             iAc;
   NALU_HYPRE_Int             iP, iP1;
   NALU_HYPRE_Int             iR;
   NALU_HYPRE_Int             yOffsetA;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetA = 0;
   yOffsetP = 0;

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 9-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 9-point fine grid operator; produces a
    * symmetric 9-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (southwest, south, southeast,
    * west, and center).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                  +           rb[iR] * a_csw[iAm1]
                  +                    a_csw[iA]  * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] = rb[iR] * a_cc[iAm1] * pa[iP1]
                 +          rb[iR] * a_cs[iAm1]
                 +                   a_cs[iA]   * pa[iP1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                  +           rb[iR] * a_cse[iAm1]
                  +                    a_cse[iA]  * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
                          +          rb[iR] * a_cw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cw[iAp1] * pa[iP1]
                          +          rb[iR] * a_cnw[iAm1]
                          +          ra[iR] * a_csw[iAp1]
                          +                   a_csw[iA]  * pb[iP1]
                          +                   a_cnw[iA]  * pa[iP1];

   rap_cc[iAc] =          a_cc[iA]
                          +          rb[iR] * a_cc[iAm1] * pb[iP]
                          +          ra[iR] * a_cc[iAp1] * pa[iP]
                          +          rb[iR] * a_cn[iAm1]
                          +          ra[iR] * a_cs[iAp1]
                          +                   a_cs[iA]   * pb[iP]
                          +                   a_cn[iA]   * pa[iP];



   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPNoSym( nalu_hypre_StructMatrix *A,
                          nalu_hypre_StructMatrix *P,
                          nalu_hypre_StructMatrix *R,
                          NALU_HYPRE_Int           cdir,
                          nalu_hypre_Index         cindex,
                          nalu_hypre_Index         cstride,
                          nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_StructStencil  *fine_stencil;
   NALU_HYPRE_Int             fine_stencil_size;

   nalu_hypre_StructGrid     *fgrid;
   NALU_HYPRE_Int            *fgrid_ids;
   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   NALU_HYPRE_Int            *cgrid_ids;
   NALU_HYPRE_Int             fi, ci;
   NALU_HYPRE_Int             constant_coefficient;

   fine_stencil = nalu_hypre_StructMatrixStencil(A);
   fine_stencil_size = nalu_hypre_StructStencilSize(fine_stencil);

   fgrid = nalu_hypre_StructMatrixGrid(A);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(RAP);
   if (constant_coefficient)
   {
      nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(R) );
      nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(A) );
      nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(P) );
   }
   else
   {
      /*      nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(R)==0 );
              nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(A)==0 );
              nalu_hypre_assert( nalu_hypre_StructMatrixConstantCoefficient(P)==0 );
      */
   }

   fi = 0;
   nalu_hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      /*-----------------------------------------------------------------
       * Switch statement to direct control to appropriate BoxLoop depending
       * on stencil size. Default is full 27-point.
       *-----------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for 5-point fine grid operator; produces upper triangular
          * part of 9-point coarse grid operator - excludes diagonal.
          * stencil entries: (northeast, north, northwest, and east)
          *--------------------------------------------------------------*/

         case 5:

            if ( constant_coefficient == 1 )
            {
               nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         /*--------------------------------------------------------------
          * Loop for 9-point fine grid operator; produces upper triangular
          * part of 9-point coarse grid operator - excludes diagonal.
          * stencil entries: (northeast, north, northwest, and east)
          *--------------------------------------------------------------*/

         default:

            if ( constant_coefficient == 1 )
            {
               nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 5, constant coefficient 0 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC0(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           stridec;
   nalu_hypre_Index           fstart;
   nalu_hypre_IndexRef        stridef;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             constant_coefficient_A;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *P_dbox;
   nalu_hypre_Box            *R_dbox;
   nalu_hypre_Box            *RAP_dbox;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;

   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cn;
   NALU_HYPRE_Real           a_cn_offd, a_cn_offdp1, a_cw_offdp1;
   NALU_HYPRE_Real           a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   NALU_HYPRE_Real           *rap_ce, *rap_cn;
   NALU_HYPRE_Real           *rap_cnw, *rap_cne;

   NALU_HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   NALU_HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   /*nalu_hypre_printf("nosym 5.0\n");*/
   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = nalu_hypre_StructMatrixConstantCoefficient(A);

   /*   fi = 0;
        nalu_hypre_ForBoxI(ci, cgrid_boxes)
        {
        while (fgrid_ids[fi] != cgrid_ids[ci])
        {
        fi++;
        }
   */
   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
   P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
   R_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int pbOffset = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int rbOffset = nalu_hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      nalu_hypre_assert( constant_coefficient_A == 2 );
      yOffsetA_diag = nalu_hypre_BoxOffsetDistance(A_dbox, index);
      yOffsetA_offd = 0;
   }

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);


   /*--------------------------------------------------------------
    * Loop for 5-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   nalu_hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
      /*nalu_hypre_printf("nosym 5.0.0\n");*/

#define DEVICE_VAR is_device_ptr(rap_cne,ra,a_ce,pb,rap_cn,a_cc,a_cn,rap_cnw,a_cw,rap_ce,rb,pa)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA;

         NALU_HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_cn[iAp1]
                       +                   a_cn[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce[iA]
                                +          rb[iR - rbOffset] * a_ce[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce[iAp1] * pa[iP1];
      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      nalu_hypre_assert( constant_coefficient_A == 2 );
      /*nalu_hypre_printf("nosym 5.0.2\n"); */

      iA_offd = 0;
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_cne,ra,pb,rap_cn,a_cc,rap_cnw,rap_ce,rb,pa)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAp1 = iA + yOffsetA_diag;

         NALU_HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_cn_offdp1
                       +                   a_cn_offd   * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce_offd
                                +          rb[iR - rbOffset] * a_ce_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce_offdp1 * pa[iP1];
      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 5, constant coefficient 1 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS5_CC1(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           fstart;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cn;
   NALU_HYPRE_Real           *rap_ce, *rap_cn;
   NALU_HYPRE_Real           *rap_cnw, *rap_cne;

   NALU_HYPRE_Int             iA, iAm1, iAp1;
   NALU_HYPRE_Int             iAc;
   NALU_HYPRE_Int             iP, iP1;
   NALU_HYPRE_Int             iR;
   NALU_HYPRE_Int             yOffsetA;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   /* nalu_hypre_printf("nosym 5.1\n");*/

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetA = 0;
   yOffsetP = 0;

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 5-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                 +          ra[iR] * a_cn[iAp1]
                 +                   a_cn[iA]   * pb[iP1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
                          +          rb[iR] * a_ce[iAm1] * pb[iP1]
                          +          ra[iR] * a_ce[iAp1] * pa[iP1];


   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 9, constant coefficient 0 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC0(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           stridec;
   nalu_hypre_Index           fstart;
   nalu_hypre_IndexRef        stridef;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             constant_coefficient_A;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *P_dbox;
   nalu_hypre_Box            *R_dbox;
   nalu_hypre_Box            *RAP_dbox;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cn;
   NALU_HYPRE_Real           *a_cse, *a_cnw, *a_cne;
   NALU_HYPRE_Real           a_cn_offd, a_cn_offdp1, a_cw_offdp1;
   NALU_HYPRE_Real           a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   NALU_HYPRE_Real           a_cne_offd, a_cne_offdm1, a_cne_offdp1;
   NALU_HYPRE_Real           a_cse_offd, a_cse_offdp1, a_cnw_offd, a_cnw_offdp1;
   NALU_HYPRE_Real           *rap_ce, *rap_cn;
   NALU_HYPRE_Real           *rap_cnw, *rap_cne;

   NALU_HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;
   NALU_HYPRE_Int             yOffsetA, yOffsetA_diag, yOffsetA_offd;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   /*nalu_hypre_printf("nosym 9.0\n");*/
   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = nalu_hypre_StructMatrixConstantCoefficient(A);

   /*   fi = 0;
        nalu_hypre_ForBoxI(ci, cgrid_boxes)
        {
        while (fgrid_ids[fi] != cgrid_ids[ci])
        {
        fi++;
        }
   */
   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
   P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
   R_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int pbOffset = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   NALU_HYPRE_Int rbOffset = nalu_hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      yOffsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      nalu_hypre_assert( constant_coefficient_A == 2 );
      yOffsetA_diag = nalu_hypre_BoxOffsetDistance(A_dbox, index);
      yOffsetA_offd = 0;
   }

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = nalu_hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/


   /*--------------------------------------------------------------
    * Loop for 9-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   nalu_hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
      /*nalu_hypre_printf("nosym 9.0.0\n");*/

#define DEVICE_VAR is_device_ptr(rap_cne,ra,a_ce,pb,a_cne,rap_cn,a_cc,a_cn,rap_cnw,a_cw,a_cnw,rap_ce,rb,pa,a_cse)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAm1 = iA - yOffsetA;
         NALU_HYPRE_Int iAp1 = iA + yOffsetA;

         NALU_HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_cne[iAp1]
                        +                    a_cne[iA]  * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_cn[iAp1]
                       +                   a_cn[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_cnw[iAp1]
                        +                    a_cnw[iA]  * pb[iP1 - pbOffset];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce[iA]
                                +          rb[iR - rbOffset] * a_ce[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_cne[iAm1]
                                +          ra[iR] * a_cse[iAp1]
                                +                   a_cse[iA]  * pb[iP1 - pbOffset]
                                +                   a_cne[iA]  * pa[iP1];

      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      /*nalu_hypre_printf("nosym 9.0.2\n");*/
      nalu_hypre_assert( constant_coefficient_A == 2 );
      iA_offd = 0;
      iA_offdm1 = iA_offd - yOffsetA_offd;
      iA_offdp1 = iA_offd + yOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];
      a_cne_offd = a_cne[iA_offd];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cne_offdp1 = a_cne[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdp1 = a_cnw[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_cne,ra,pb,rap_cn,a_cc,rap_cnw,rap_ce,rb,pa)
      nalu_hypre_BoxLoop4Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         NALU_HYPRE_Int iAp1 = iA + yOffsetA_diag;

         NALU_HYPRE_Int iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_cne_offdp1
                        +                    a_cne_offd  * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_cn_offdp1
                       +                   a_cn_offd   * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_cnw_offdp1
                        +                    a_cnw_offd  * pb[iP1 - pbOffset];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce_offd
                                +          rb[iR - rbOffset] * a_ce_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce_offdp1 * pa[iP1]
                                +          rb[iR - rbOffset] * a_cne_offdm1
                                +          ra[iR] * a_cse_offdp1
                                +                   a_cse_offd  * pb[iP1 - pbOffset]
                                +                   a_cne_offd  * pa[iP1];

      }
      nalu_hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}

/* for fine stencil size 9, constant coefficient 1 */
NALU_HYPRE_Int
nalu_hypre_PFMG2BuildRAPNoSym_onebox_FSS9_CC1(
   NALU_HYPRE_Int             ci,
   NALU_HYPRE_Int             fi,
   nalu_hypre_StructMatrix *A,
   nalu_hypre_StructMatrix *P,
   nalu_hypre_StructMatrix *R,
   NALU_HYPRE_Int           cdir,
   nalu_hypre_Index         cindex,
   nalu_hypre_Index         cstride,
   nalu_hypre_StructMatrix *RAP     )
{

   nalu_hypre_Index           index;
   nalu_hypre_Index           index_temp;

   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   nalu_hypre_Box            *cgrid_box;
   nalu_hypre_IndexRef        cstart;
   nalu_hypre_Index           fstart;

   NALU_HYPRE_Real           *pa, *pb;
   NALU_HYPRE_Real           *ra, *rb;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cn;
   NALU_HYPRE_Real           *a_cse, *a_cnw, *a_cne;
   NALU_HYPRE_Real           *rap_ce, *rap_cn;
   NALU_HYPRE_Real           *rap_cnw, *rap_cne;

   NALU_HYPRE_Int             iA, iAm1, iAp1;
   NALU_HYPRE_Int             iAc;
   NALU_HYPRE_Int             iP, iP1;
   NALU_HYPRE_Int             iR;
   NALU_HYPRE_Int             yOffsetA;
   NALU_HYPRE_Int             xOffsetP;
   NALU_HYPRE_Int             yOffsetP;

   /*nalu_hypre_printf("nosym 9.1\n");*/

   cgrid = nalu_hypre_StructMatrixGrid(RAP);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

   cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = nalu_hypre_BoxIMin(cgrid_box);
   nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   pa = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   pb = nalu_hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   ra = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   rb = nalu_hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point fine grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for coarse grid operator - always 9-point:
    *
    * We build only the upper triangular part.
    *
    * rap_ce is pointer for east coefficient (etc.)
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   nalu_hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetA = 0;
   yOffsetP = 0;

   nalu_hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to appropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/


   /*--------------------------------------------------------------
    * Loop for 9-point fine grid operator; produces upper triangular
    * part of 9-point coarse grid operator - excludes diagonal.
    * stencil entries: (northeast, north, northwest, and east)
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - yOffsetA;
   iAp1 = iA + yOffsetA;

   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                  +           ra[iR] * a_cne[iAp1]
                  +                    a_cne[iA]  * pb[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] = ra[iR] * a_cc[iAp1] * pb[iP1]
                 +          ra[iR] * a_cn[iAp1]
                 +                   a_cn[iA]   * pb[iP1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                  +           ra[iR] * a_cnw[iAp1]
                  +                    a_cnw[iA]  * pb[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
                          +          rb[iR] * a_ce[iAm1] * pb[iP1]
                          +          ra[iR] * a_ce[iAp1] * pa[iP1]
                          +          rb[iR] * a_cne[iAm1]
                          +          ra[iR] * a_cse[iAp1]
                          +                   a_cse[iA]  * pb[iP1]
                          +                   a_cne[iA]  * pa[iP1];



   /*      }*/ /* end ForBoxI */

   return nalu_hypre_error_flag;
}



