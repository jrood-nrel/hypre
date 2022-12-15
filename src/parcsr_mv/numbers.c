/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_parcsr_mv.h"

nalu_hypre_NumbersNode * nalu_hypre_NumbersNewNode()
/* makes a new node for a tree representing numbers */
{
   NALU_HYPRE_Int i;
   nalu_hypre_NumbersNode * newnode = nalu_hypre_CTAlloc( nalu_hypre_NumbersNode,  1, NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i <= 10; ++i ) { newnode->digit[i] = NULL; }
   return newnode;
}

void nalu_hypre_NumbersDeleteNode( nalu_hypre_NumbersNode * node )
/* deletes a node and the tree of which it is root */
{
   NALU_HYPRE_Int i;
   for ( i = 0; i <= 10; ++i ) if ( node->digit[i] != NULL )
      {
         nalu_hypre_NumbersDeleteNode( node->digit[i] );
         node->digit[i] = NULL;
      };
   nalu_hypre_TFree( node, NALU_HYPRE_MEMORY_HOST);
}

NALU_HYPRE_Int nalu_hypre_NumbersEnter( nalu_hypre_NumbersNode * node, const NALU_HYPRE_Int n )
/* enters a number in the tree starting with 'node'. */
{
   NALU_HYPRE_Int newN = 0;
   NALU_HYPRE_Int q = n / 10;
   NALU_HYPRE_Int r = n % 10;
   nalu_hypre_assert( n >= 0 );
   if ( node->digit[r] == NULL )
   {
      node->digit[r] = nalu_hypre_NumbersNewNode();
      newN = 1;
   };
   if ( q < 10 )  /* q is a one-digit number; point to terminal object */
   {
      if ( (node->digit[r])->digit[10] == NULL )
      {
         (node->digit[r])->digit[10] = nalu_hypre_NumbersNewNode();
      }
   }
   else    /* multidigit number; place for this digit points to next node */
   {
      newN = nalu_hypre_NumbersEnter(node->digit[r], q );
   }
   return newN;
}

NALU_HYPRE_Int nalu_hypre_NumbersNEntered( nalu_hypre_NumbersNode * node )
/* returns the number of numbers represented by the tree whose root is 'node' */
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int count = 0;
   if ( node == NULL ) { return 0; }
   for ( i = 0; i < 10; ++i ) if ( node->digit[i] != NULL )
      {
         count += nalu_hypre_NumbersNEntered( node->digit[i] );
      }
   if ( node->digit[10] != NULL ) { ++count; }
   return count;
}

NALU_HYPRE_Int nalu_hypre_NumbersQuery( nalu_hypre_NumbersNode * node, const NALU_HYPRE_Int n )
/* returns 1 if n is on the tree with root 'node', 0 otherwise */
{
   NALU_HYPRE_Int q = n / 10;
   NALU_HYPRE_Int r = n % 10;
   nalu_hypre_assert( n >= 0 );
   if ( node->digit[r] == NULL )   /* low order digit of n not on tree */
   {
      return 0;
   }
   else if ( q < 10 ) /* q is a one-digit number; check terminal object */
   {
      if ( (node->digit[r])->digit[10] == NULL )
      {
         return 0;
      }
      else
      {
         return 1;
      }
   }
   else    /* look for higher order digits of n on tree of its low order digit r */
   {
      return nalu_hypre_NumbersQuery( node->digit[r], q );
   }
}

NALU_HYPRE_Int * nalu_hypre_NumbersArray( nalu_hypre_NumbersNode * node )
/* allocates and returns an unordered array of ints as a simpler representation
   of the contents of the Numbers tree.
   For the array length, call nalu_hypre_NumbersNEntered */
{
   NALU_HYPRE_Int i, j, Ntemp;
   NALU_HYPRE_Int k = 0;
   NALU_HYPRE_Int N = nalu_hypre_NumbersNEntered(node);
   NALU_HYPRE_Int * array, * temp;
   array = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  N, NALU_HYPRE_MEMORY_HOST);
   if ( node == NULL ) { return array; }
   for ( i = 0; i < 10; ++i ) if ( node->digit[i] != NULL )
      {
         Ntemp = nalu_hypre_NumbersNEntered( node->digit[i] );
         temp = nalu_hypre_NumbersArray( node->digit[i] );
         for ( j = 0; j < Ntemp; ++j )
         {
            array[k++] = temp[j] * 10 + i;
         }
         nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
      }
   if ( node->digit[10] != NULL ) { array[k++] = 0; }
   nalu_hypre_assert( k == N );
   return array;
}
