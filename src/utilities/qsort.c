/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap( NALU_HYPRE_Int *v,
                 NALU_HYPRE_Int  i,
                 NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap_c( NALU_HYPRE_Complex *v,
                   NALU_HYPRE_Int      i,
                   NALU_HYPRE_Int      j )
{
   NALU_HYPRE_Complex temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap2( NALU_HYPRE_Int  *v,
                  NALU_HYPRE_Real *w,
                  NALU_HYPRE_Int   i,
                  NALU_HYPRE_Int   j )
{
   NALU_HYPRE_Int  temp;
   NALU_HYPRE_Real temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigSwap2( NALU_HYPRE_BigInt *v,
                     NALU_HYPRE_Real   *w,
                     NALU_HYPRE_Int     i,
                     NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt temp;
   NALU_HYPRE_Real   temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap2i( NALU_HYPRE_Int  *v,
                   NALU_HYPRE_Int  *w,
                   NALU_HYPRE_Int  i,
                   NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

void nalu_hypre_BigSwap2i( NALU_HYPRE_BigInt *v,
                      NALU_HYPRE_Int    *w,
                      NALU_HYPRE_Int     i,
                      NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt big_temp;
   NALU_HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


/* AB 11/04 */

void nalu_hypre_swap3i( NALU_HYPRE_Int  *v,
                   NALU_HYPRE_Int  *w,
                   NALU_HYPRE_Int  *z,
                   NALU_HYPRE_Int  i,
                   NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap3_d( NALU_HYPRE_Real *v,
                    NALU_HYPRE_Int  *w,
                    NALU_HYPRE_Int  *z,
                    NALU_HYPRE_Int   i,
                    NALU_HYPRE_Int   j )
{
   NALU_HYPRE_Int  temp;
   NALU_HYPRE_Real temp_d;

   temp_d = v[i];
   v[i] = v[j];
   v[j] = temp_d;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/* swap (v[i], v[j]), (w[i], w[j]), and (z[v[i]], z[v[j]]) - DOK */
void nalu_hypre_swap3_d_perm( NALU_HYPRE_Int  *v,
                         NALU_HYPRE_Real *w,
                         NALU_HYPRE_Int  *z,
                         NALU_HYPRE_Int  i,
                         NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Int temp;
   NALU_HYPRE_Real temp_d;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp_d = w[i];
   w[i] = w[j];
   w[j] = temp_d;
   temp = z[v[i]];
   z[v[i]] = z[v[j]];
   z[v[j]] = temp;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigSwap4_d( NALU_HYPRE_Real   *v,
                       NALU_HYPRE_BigInt *w,
                       NALU_HYPRE_Int    *z,
                       NALU_HYPRE_Int    *y,
                       NALU_HYPRE_Int     i,
                       NALU_HYPRE_Int     j )
{
   NALU_HYPRE_Int temp;
   NALU_HYPRE_BigInt big_temp;
   NALU_HYPRE_Real temp_d;

   temp_d = v[i];
   v[i] = v[j];
   v[j] = temp_d;
   big_temp = w[i];
   w[i] = w[j];
   w[j] = big_temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
   temp = y[i];
   y[i] = y[j];
   y[j] = temp;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_swap_d( NALU_HYPRE_Real *v,
                   NALU_HYPRE_Int  i,
                   NALU_HYPRE_Int  j )
{
   NALU_HYPRE_Real temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_qsort0( NALU_HYPRE_Int *v,
                   NALU_HYPRE_Int  left,
                   NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap(v, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap(v, ++last, i);
      }
   }
   nalu_hypre_swap(v, left, last);
   nalu_hypre_qsort0(v, left, last - 1);
   nalu_hypre_qsort0(v, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_qsort1( NALU_HYPRE_Int  *v,
                   NALU_HYPRE_Real *w,
                   NALU_HYPRE_Int   left,
                   NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap2(v, w, ++last, i);
      }
   }
   nalu_hypre_swap2(v, w, left, last);
   nalu_hypre_qsort1(v, w, left, last - 1);
   nalu_hypre_qsort1(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsort1( NALU_HYPRE_BigInt *v,
                      NALU_HYPRE_Real   *w,
                      NALU_HYPRE_Int     left,
                      NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwap2(v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwap2(v, w, ++last, i);
      }
   }
   nalu_hypre_BigSwap2(v, w, left, last);
   nalu_hypre_BigQsort1(v, w, left, last - 1);
   nalu_hypre_BigQsort1(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_qsort2i( NALU_HYPRE_Int *v,
                    NALU_HYPRE_Int *w,
                    NALU_HYPRE_Int  left,
                    NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap2i( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap2i(v, w, ++last, i);
      }
   }
   nalu_hypre_swap2i(v, w, left, last);
   nalu_hypre_qsort2i(v, w, left, last - 1);
   nalu_hypre_qsort2i(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsort2i( NALU_HYPRE_BigInt *v,
                       NALU_HYPRE_Int *w,
                       NALU_HYPRE_Int  left,
                       NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwap2i( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwap2i(v, w, ++last, i);
      }
   }
   nalu_hypre_BigSwap2i(v, w, left, last);
   nalu_hypre_BigQsort2i(v, w, left, last - 1);
   nalu_hypre_BigQsort2i(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*   sort on w (NALU_HYPRE_Real), move v (AB 11/04) */

void nalu_hypre_qsort2( NALU_HYPRE_Int  *v,
                   NALU_HYPRE_Real *w,
                   NALU_HYPRE_Int   left,
                   NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (w[i] < w[left])
      {
         nalu_hypre_swap2(v, w, ++last, i);
      }
   }
   nalu_hypre_swap2(v, w, left, last);
   nalu_hypre_qsort2(v, w, left, last - 1);
   nalu_hypre_qsort2(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* qsort2 based on absolute value of entries in w. */
void nalu_hypre_qsort2_abs( NALU_HYPRE_Int  *v,
                       NALU_HYPRE_Real *w,
                       NALU_HYPRE_Int   left,
                       NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;
   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (fabs(w[i]) > fabs(w[left]))
      {
         nalu_hypre_swap2(v, w, ++last, i);
      }
   }
   nalu_hypre_swap2(v, w, left, last);
   nalu_hypre_qsort2_abs(v, w, left, last - 1);
   nalu_hypre_qsort2_abs(v, w, last + 1, right);
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort on v, move w and z (AB 11/04) */

void nalu_hypre_qsort3i( NALU_HYPRE_Int *v,
                    NALU_HYPRE_Int *w,
                    NALU_HYPRE_Int *z,
                    NALU_HYPRE_Int  left,
                    NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap3i( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap3i(v, w, z, ++last, i);
      }
   }
   nalu_hypre_swap3i(v, w, z, left, last);
   nalu_hypre_qsort3i(v, w, z, left, last - 1);
   nalu_hypre_qsort3i(v, w, z, last + 1, right);
}

/* sort on v, move w and z DOK */
void nalu_hypre_qsort3ir( NALU_HYPRE_Int  *v,
                     NALU_HYPRE_Real *w,
                     NALU_HYPRE_Int  *z,
                     NALU_HYPRE_Int   left,
                     NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap3_d_perm( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap3_d_perm(v, w, z, ++last, i);
      }
   }
   nalu_hypre_swap3_d_perm(v, w, z, left, last);
   nalu_hypre_qsort3ir(v, w, z, left, last - 1);
   nalu_hypre_qsort3ir(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on real array v */
void nalu_hypre_qsort3( NALU_HYPRE_Real *v,
                   NALU_HYPRE_Int  *w,
                   NALU_HYPRE_Int  *z,
                   NALU_HYPRE_Int   left,
                   NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap3_d( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_swap3_d(v, w, z, ++last, i);
      }
   }
   nalu_hypre_swap3_d(v, w, z, left, last);
   nalu_hypre_qsort3(v, w, z, left, last - 1);
   nalu_hypre_qsort3(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on absolute value */

void nalu_hypre_qsort3_abs(NALU_HYPRE_Real *v,
                      NALU_HYPRE_Int *w,
                      NALU_HYPRE_Int *z,
                      NALU_HYPRE_Int  left,
                      NALU_HYPRE_Int  right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap3_d( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (fabs(v[i]) < fabs(v[left]))
      {
         nalu_hypre_swap3_d(v, w, z, ++last, i);
      }
   }
   nalu_hypre_swap3_d(v, w, z, left, last);
   nalu_hypre_qsort3_abs(v, w, z, left, last - 1);
   nalu_hypre_qsort3_abs(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on absolute value */

void nalu_hypre_BigQsort4_abs( NALU_HYPRE_Real   *v,
                          NALU_HYPRE_BigInt *w,
                          NALU_HYPRE_Int    *z,
                          NALU_HYPRE_Int    *y,
                          NALU_HYPRE_Int     left,
                          NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwap4_d( v, w, z, y, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (fabs(v[i]) < fabs(v[left]))
      {
         nalu_hypre_BigSwap4_d(v, w, z, y, ++last, i);
      }
   }
   nalu_hypre_BigSwap4_d(v, w, z, y, left, last);
   nalu_hypre_BigQsort4_abs(v, w, z, y, left, last - 1);
   nalu_hypre_BigQsort4_abs(v, w, z, y, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* sort min to max based on absolute value */

void nalu_hypre_qsort_abs( NALU_HYPRE_Real *w,
                      NALU_HYPRE_Int   left,
                      NALU_HYPRE_Int   right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_swap_d( w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (fabs(w[i]) < fabs(w[left]))
      {
         nalu_hypre_swap_d(w, ++last, i);
      }
   }
   nalu_hypre_swap_d(w, left, last);
   nalu_hypre_qsort_abs(w, left, last - 1);
   nalu_hypre_qsort_abs(w, last + 1, right);
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigSwapbi( NALU_HYPRE_BigInt *v,
                      NALU_HYPRE_Int    *w,
                      NALU_HYPRE_Int     i,
                      NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt big_temp;
   NALU_HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsortbi( NALU_HYPRE_BigInt *v,
                       NALU_HYPRE_Int    *w,
                       NALU_HYPRE_Int     left,
                       NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwapbi( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwapbi(v, w, ++last, i);
      }
   }
   nalu_hypre_BigSwapbi(v, w, left, last);
   nalu_hypre_BigQsortbi(v, w, left, last - 1);
   nalu_hypre_BigQsortbi(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigSwapLoc( NALU_HYPRE_BigInt *v,
                       NALU_HYPRE_Int    *w,
                       NALU_HYPRE_Int     i,
                       NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt big_temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   w[i] = j;
   w[j] = i;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsortbLoc( NALU_HYPRE_BigInt *v,
                         NALU_HYPRE_Int    *w,
                         NALU_HYPRE_Int     left,
                         NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwapLoc( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwapLoc(v, w, ++last, i);
      }
   }
   nalu_hypre_BigSwapLoc(v, w, left, last);
   nalu_hypre_BigQsortbLoc(v, w, left, last - 1);
   nalu_hypre_BigQsortbLoc(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


void nalu_hypre_BigSwapb2i( NALU_HYPRE_BigInt *v,
                       NALU_HYPRE_Int    *w,
                       NALU_HYPRE_Int    *z,
                       NALU_HYPRE_Int     i,
                       NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt big_temp;
   NALU_HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsortb2i( NALU_HYPRE_BigInt *v,
                        NALU_HYPRE_Int    *w,
                        NALU_HYPRE_Int    *z,
                        NALU_HYPRE_Int     left,
                        NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwapb2i( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwapb2i(v, w, z, ++last, i);
      }
   }
   nalu_hypre_BigSwapb2i(v, w, z, left, last);
   nalu_hypre_BigQsortb2i(v, w, z, left, last - 1);
   nalu_hypre_BigQsortb2i(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigSwap( NALU_HYPRE_BigInt *v,
                    NALU_HYPRE_Int     i,
                    NALU_HYPRE_Int     j )
{
   NALU_HYPRE_BigInt temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void nalu_hypre_BigQsort0( NALU_HYPRE_BigInt *v,
                      NALU_HYPRE_Int     left,
                      NALU_HYPRE_Int     right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   nalu_hypre_BigSwap( v, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         nalu_hypre_BigSwap(v, ++last, i);
      }
   }
   nalu_hypre_BigSwap(v, left, last);
   nalu_hypre_BigQsort0(v, left, last - 1);
   nalu_hypre_BigQsort0(v, last + 1, right);
}

// Recursive DFS search.
static void nalu_hypre_search_row(NALU_HYPRE_Int            row,
                             const NALU_HYPRE_Int     *row_ptr,
                             const NALU_HYPRE_Int     *col_inds,
                             const NALU_HYPRE_Complex *data,
                             NALU_HYPRE_Int           *visited,
                             NALU_HYPRE_Int           *ordering,
                             NALU_HYPRE_Int           *order_ind)
{
   // If this row has not been visited, call recursive DFS on nonzero
   // column entries
   if (!visited[row])
   {
      NALU_HYPRE_Int j;
      visited[row] = 1;
      for (j = row_ptr[row]; j < row_ptr[row + 1]; j++)
      {
         NALU_HYPRE_Int col = col_inds[j];
         nalu_hypre_search_row(col, row_ptr, col_inds, data,
                          visited, ordering, order_ind);
      }
      // Add node to ordering *after* it has been searched
      ordering[*order_ind] = row;
      *order_ind += 1;
   }
}


// Find topological ordering on acyclic CSR matrix. That is, find ordering
// of matrix to be triangular.
//
// INPUT
// -----
//    - rowptr[], colinds[], data[] form a CSR structure for nxn matrix
//    - ordering[] should be empty array of length n
void nalu_hypre_topo_sort( const NALU_HYPRE_Int     *row_ptr,
                      const NALU_HYPRE_Int     *col_inds,
                      const NALU_HYPRE_Complex *data,
                      NALU_HYPRE_Int           *ordering,
                      NALU_HYPRE_Int            n)
{
   NALU_HYPRE_Int *visited = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int order_ind = 0;
   NALU_HYPRE_Int temp_row = 0;
   while (order_ind < n)
   {
      nalu_hypre_search_row(temp_row, row_ptr, col_inds, data,
                       visited, ordering, &order_ind);
      temp_row += 1;
      if (temp_row == n)
      {
         temp_row = 0;
      }
   }
   nalu_hypre_TFree(visited, NALU_HYPRE_MEMORY_HOST);
}


// Recursive DFS search.
static void nalu_hypre_dense_search_row(NALU_HYPRE_Int            row,
                                   const NALU_HYPRE_Complex *L,
                                   NALU_HYPRE_Int           *visited,
                                   NALU_HYPRE_Int           *ordering,
                                   NALU_HYPRE_Int           *order_ind,
                                   NALU_HYPRE_Int            n,
                                   NALU_HYPRE_Int            is_col_major)
{
   // If this row has not been visited, call recursive DFS on nonzero
   // column entries
   if (!visited[row])
   {
      NALU_HYPRE_Int col;
      visited[row] = 1;
      for (col = 0; col < n; col++)
      {
         NALU_HYPRE_Complex val;
         if (is_col_major)
         {
            val = L[col * n + row];
         }
         else
         {
            val = L[row * n + col];
         }
         if (nalu_hypre_cabs(val) > 1e-14)
         {
            nalu_hypre_dense_search_row(col, L, visited, ordering, order_ind, n, is_col_major);
         }
      }
      // Add node to ordering *after* it has been searched
      ordering[*order_ind] = row;
      *order_ind += 1;
   }
}


// Find topological ordering of acyclic dense matrix in column major
// format. That is, find ordering of matrix to be triangular.
//
// INPUT
// -----
//    - L[] : dense nxn matrix in column major format
//    - ordering[] should be empty array of length n
//    - row is the row to start the search from
void nalu_hypre_dense_topo_sort(const NALU_HYPRE_Complex *L,
                           NALU_HYPRE_Int           *ordering,
                           NALU_HYPRE_Int            n,
                           NALU_HYPRE_Int            is_col_major)
{
   NALU_HYPRE_Int *visited = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int order_ind = 0;
   NALU_HYPRE_Int temp_row = 0;
   while (order_ind < n)
   {
      nalu_hypre_dense_search_row(temp_row, L, visited, ordering, &order_ind, n, is_col_major);
      temp_row += 1;
      if (temp_row == n)
      {
         temp_row = 0;
      }
   }
   nalu_hypre_TFree(visited, NALU_HYPRE_MEMORY_HOST);
}
