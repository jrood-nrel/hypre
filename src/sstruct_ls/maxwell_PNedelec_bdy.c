/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * Finds the boundary boxes for all var_grids in pgrid. Use the cell grid
 * to determine the boundary.
 * bdry[n_cellboxes, nvars+1]= boxarrayarray ptr.: nalu_hypre_BoxArrayArray ***bdry.
 * bdry[n_cellboxes, 0] is the cell-centred box.
 * Each box_arrayarray: for each variable, there are a max of 2*(ndim-1)
 * box_arrays (e.g., in 3d, the x_edges on the boundary can be the two
 * z_faces & the two y_faces of the boundary). Each of these box_arrays
 * consists of boxes that can be on the boundary.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_Maxwell_PNedelec_Bdy( nalu_hypre_StructGrid       *cell_grid,
                            nalu_hypre_SStructPGrid     *pgrid,
                            nalu_hypre_BoxArrayArray ****bdry_ptr )
{

   NALU_HYPRE_Int ierr = 0;

   NALU_HYPRE_Int              nvars    = nalu_hypre_SStructPGridNVars(pgrid);

   nalu_hypre_BoxArrayArray   *cellgrid_bdry;
   nalu_hypre_BoxArrayArray ***bdry;
   nalu_hypre_BoxArray        *box_array, *box_array2;
   nalu_hypre_BoxArray        *cell_boxes;
   nalu_hypre_Box             *box, *bdy_box, *shifted_box;

   NALU_HYPRE_Int              ndim     = nalu_hypre_SStructPGridNDim(pgrid);

   NALU_HYPRE_SStructVariable *vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
   nalu_hypre_Index            varoffset, ishift, jshift, kshift;
   nalu_hypre_Index            lower, upper;

   NALU_HYPRE_Int             *flag;
   NALU_HYPRE_Int              i, j, k, t, nboxes, bdy;

   nalu_hypre_SetIndex3(ishift, 1, 0, 0);
   nalu_hypre_SetIndex3(jshift, 0, 1, 0);
   nalu_hypre_SetIndex3(kshift, 0, 0, 1);

   cell_boxes = nalu_hypre_StructGridBoxes(cell_grid);
   nboxes    = nalu_hypre_BoxArraySize(cell_boxes);

   bdry = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray **,  nboxes, NALU_HYPRE_MEMORY_HOST);
   shifted_box = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_ForBoxI(j, cell_boxes)
   {
      box = nalu_hypre_BoxArrayBox(cell_boxes, j);

      /* find the cellgrid boundaries of box if there are any. */
      cellgrid_bdry = nalu_hypre_BoxArrayArrayCreate(2 * ndim, ndim);
      flag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * ndim, NALU_HYPRE_MEMORY_HOST);
      bdy = 0;

      for (i = 0; i < ndim; i++)
      {
         nalu_hypre_BoxBoundaryDG(box, cell_grid,
                             nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2 * i),
                             nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2 * i + 1),
                             i);
         if (nalu_hypre_BoxArraySize(nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2 * i)))
         {
            flag[2 * i] = 1;
            bdy++;
         }

         if (nalu_hypre_BoxArraySize(nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2 * i + 1)))
         {
            flag[2 * i + 1] = 1;
            bdy++;
         }
      }

      /* There are boundary boxes. Every variable of pgrid will have some */
      if (bdy)
      {
         bdry[j] = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray *,  nvars + 1, NALU_HYPRE_MEMORY_HOST);

         /* keep the cell-centred boxarrayarray of boundaries */
         bdry[j][0] = nalu_hypre_BoxArrayArrayDuplicate(cellgrid_bdry);

         k = 2 * (ndim - 1); /* 3-d requires 4 boundary faces to be checked */
         for (i = 0; i < nvars; i++)
         {
            bdry[j][i + 1] = nalu_hypre_BoxArrayArrayCreate(k, ndim); /* one for +/- directions */
         }

         for (i = 0; i < nvars; i++)
         {
            t = vartypes[i];
            nalu_hypre_SStructVariableGetOffset(vartypes[i], ndim, varoffset);

            switch (t)
            {
               case 2: /* xface, boundary i= lower, upper */
               {
                  if (flag[0]) /* boundary i= lower */
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 0);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, varoffset, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[1]) /* boundary i= upper */
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 1);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 3: /* yface, boundary j= lower, upper */
               {
                  if (flag[2]) /* boundary j= lower */
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 0);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, varoffset, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[3]) /* boundary j= upper */
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 1);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 5: /* xedge, boundary z_faces & y_faces */
               {
                  if (flag[4]) /* boundary k= lower zface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 4);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 0);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, kshift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[5]) /* boundary k= upper zface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 5);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 1);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, jshift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[2]) /* boundary j= lower yface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 2);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, jshift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[3]) /* boundary j= upper yface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 3);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, kshift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                  break;
               }

               case 6: /* yedge, boundary z_faces & x_faces */
               {
                  if (flag[4]) /* boundary k= lower zface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 4);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 0);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, kshift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[5]) /* boundary k= upper zface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 5);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 1);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, ishift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[0]) /* boundary i= lower xface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 2);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, ishift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[1]) /* boundary i= upper xface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 3);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, kshift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  break;
               }

               case 7: /* zedge, boundary y_faces & x_faces */
               {
                  if (flag[2]) /* boundary j= lower yface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 2);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 0);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, jshift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[3]) /* boundary j= upper yface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 3);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 1);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, ishift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[0]) /* boundary i= lower xface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 0);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 2);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, varoffset, ndim, lower);
                        nalu_hypre_SubtractIndexes(upper, ishift, ndim, upper);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }

                  if (flag[1]) /* boundary i= upper xface*/
                  {
                     box_array = nalu_hypre_BoxArrayArrayBoxArray(cellgrid_bdry, 1);
                     box_array2 = nalu_hypre_BoxArrayArrayBoxArray(bdry[j][i + 1], 3);
                     nalu_hypre_ForBoxI(k, box_array)
                     {
                        bdy_box = nalu_hypre_BoxArrayBox(box_array, k);

                        /* bdry boxes */
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(bdy_box), lower);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(bdy_box), upper);
                        nalu_hypre_SubtractIndexes(lower, jshift, ndim, lower);

                        nalu_hypre_BoxSetExtents(shifted_box, lower, upper);
                        nalu_hypre_AppendBox(shifted_box, box_array2);
                     }
                  }
                  break;
               }

            }  /* switch(t) */
         }     /* for (i= 0; i< nvars; i++) */
      }        /* if (bdy) */

      else
      {
         /* make an empty ptr of boxarrayarrays to avoid memory leaks when
            destroying bdry later. */
         bdry[j] = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray *,  nvars + 1, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nvars + 1; i++)
         {
            bdry[j][i] = nalu_hypre_BoxArrayArrayCreate(0, ndim);
         }
      }

      nalu_hypre_BoxArrayArrayDestroy(cellgrid_bdry);
      nalu_hypre_TFree(flag, NALU_HYPRE_MEMORY_HOST);
   }  /* nalu_hypre_ForBoxI(j, cell_boxes) */

   nalu_hypre_BoxDestroy(shifted_box);

   *bdry_ptr     = bdry;

   return ierr;
}

