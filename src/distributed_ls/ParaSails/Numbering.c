/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Numbering - An object that maintains a mapping to and from global indices
 * and local indices.  The local indices are numbered:
 *
 * 0 .. num_loc - 1         <--- locally owned indices
 * num_loc .. num_ind - 1   <--- external indices
 *
 * Implementation:  Mapping from a local index to a global index is performed
 * through an array.  Mapping from a global index to a local index is more
 * difficult.  If the global index is determined to be owned by the local
 * processor, then a conversion is performed; else the local index is
 * looked up in a hash table.
 *
 *****************************************************************************/

#include <stdlib.h>
//#include <memory.h>
#include "Common.h"
#include "Numbering.h"
#include "OrderStat.h"

/*--------------------------------------------------------------------------
 * NumberingCreate - Return (a pointer to) a numbering object
 * for a given matrix.  The "size" parameter is the initial number of
 * external indices that can be stored, and will grow if necessary.
 * (Implementation note: the hash table size is kept approximately twice
 * this number.)
 *
 * The numbering is created such that the local indices from a given processor
 * are contiguous.  This is required by the mat-vec routine.
 *--------------------------------------------------------------------------*/

Numbering *NumberingCreate(Matrix *mat, NALU_HYPRE_Int size)
{
   Numbering *numb = nalu_hypre_TAlloc(Numbering, 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int row, i, len, *ind;
   NALU_HYPRE_Real *val;
   NALU_HYPRE_Int num_external = 0;

   numb->size    = size;
   numb->beg_row = mat->beg_row;
   numb->end_row = mat->end_row;
   numb->num_loc = mat->end_row - mat->beg_row + 1;
   numb->num_ind = mat->end_row - mat->beg_row + 1;

   numb->local_to_global = nalu_hypre_TAlloc(NALU_HYPRE_Int, (numb->num_loc+size) , NALU_HYPRE_MEMORY_HOST);
   numb->hash            = HashCreate(2*size+1);

   /* Set up the local part of local_to_global */
   for (i=0; i<numb->num_loc; i++)
      numb->local_to_global[i] = mat->beg_row + i;

   /* Fill local_to_global array */
   for (row=0; row<=mat->end_row - mat->beg_row; row++)
   {
      MatrixGetRow(mat, row, &len, &ind, &val);

      for (i=0; i<len; i++)
      {
         /* Only interested in external indices */
         if (ind[i] < mat->beg_row || ind[i] > mat->end_row)
         {
            if (HashLookup(numb->hash, ind[i]) == HASH_NOTFOUND)
            {
               if (num_external >= numb->size)
               {
                  Hash *newHash;

                  /* allocate more space for numbering */
                  numb->size *= 2;
                  numb->local_to_global = (NALU_HYPRE_Int *)
                     nalu_hypre_TReAlloc(numb->local_to_global,NALU_HYPRE_Int,
                           (numb->num_loc+numb->size), NALU_HYPRE_MEMORY_HOST);
                  newHash = HashCreate(2*numb->size+1);
                  HashRehash(numb->hash, newHash);
                  HashDestroy(numb->hash);
                  numb->hash = newHash;
               }

               HashInsert(numb->hash, ind[i], num_external);
               numb->local_to_global[numb->num_loc+num_external] = ind[i];
               num_external++;
            }
         }
      }
   }

   /* Sort the indices */
   nalu_hypre_shell_sort(num_external, &numb->local_to_global[numb->num_loc]);

   /* Redo the hash table for the sorted indices */
   HashReset(numb->hash);

   for (i=0; i<num_external; i++)
      HashInsert(numb->hash,
            numb->local_to_global[i+numb->num_loc], i+numb->num_loc);

   numb->num_ind += num_external;

   return numb;
}

/*--------------------------------------------------------------------------
 * NumberingCreateCopy - Create a new numbering object, and take as initial
 * contents the "orig" numbering object.
 *--------------------------------------------------------------------------*/

Numbering *NumberingCreateCopy(Numbering *orig)
{
   Numbering *numb = nalu_hypre_TAlloc(Numbering, 1, NALU_HYPRE_MEMORY_HOST);

   numb->size    = orig->size;
   numb->beg_row = orig->beg_row;
   numb->end_row = orig->end_row;
   numb->num_loc = orig->num_loc;
   numb->num_ind = orig->num_ind;

   numb->local_to_global =
      nalu_hypre_TAlloc(NALU_HYPRE_Int, (numb->num_loc+numb->size) , NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(numb->local_to_global,  orig->local_to_global,
         NALU_HYPRE_Int, numb->num_ind, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

   numb->hash = HashCreate(2*numb->size+1);
   HashRehash(orig->hash, numb->hash);

   return numb;
}

/*--------------------------------------------------------------------------
 * NumberingDestroy - Destroy a numbering object.
 *--------------------------------------------------------------------------*/

void NumberingDestroy(Numbering *numb)
{
   nalu_hypre_TFree(numb->local_to_global,NALU_HYPRE_MEMORY_HOST);
   HashDestroy(numb->hash);

   nalu_hypre_TFree(numb,NALU_HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * NumberingLocalToGlobal - Convert an array of indices to global coordinate
 * numbering.  May be done in place.
 *--------------------------------------------------------------------------*/

void NumberingLocalToGlobal(Numbering *numb, NALU_HYPRE_Int len, NALU_HYPRE_Int *local, NALU_HYPRE_Int *global)
{
   NALU_HYPRE_Int i;

   for (i=0; i<len; i++)
      global[i] = numb->local_to_global[local[i]];
}

/*--------------------------------------------------------------------------
 * NumberingGlobalToLocal - Convert an array of indices to local coordinate
 * numbering.  If the global coordinate number does not exist, it is added
 * to the numbering object.  May be done in place.
 *--------------------------------------------------------------------------*/

void NumberingGlobalToLocal(Numbering *numb, NALU_HYPRE_Int len, NALU_HYPRE_Int *global, NALU_HYPRE_Int *local)
{
   NALU_HYPRE_Int i, l;

   for (i=0; i<len; i++)
   {
      if (global[i] < numb->beg_row || global[i] > numb->end_row)
      {
         l = HashLookup(numb->hash, global[i]);

         if (l == HASH_NOTFOUND)
         {
            if (numb->num_ind >= numb->num_loc + numb->size)
            {
               Hash *newHash;

               /* allocate more space for numbering */
               numb->size *= 2;
#ifdef PARASAILS_DEBUG
               nalu_hypre_printf("Numbering resize %d\n", numb->size);
#endif
               numb->local_to_global = nalu_hypre_TReAlloc(numb->local_to_global,
                                                      NALU_HYPRE_Int,
                                                      numb->num_loc + numb->size,
                                                      NALU_HYPRE_MEMORY_HOST);

               newHash = HashCreate(2*numb->size+1);
               HashRehash(numb->hash, newHash);
               HashDestroy(numb->hash);
               numb->hash = newHash;
            }

            HashInsert(numb->hash, global[i], numb->num_ind);
            numb->local_to_global[numb->num_ind] = global[i];
            local[i] = numb->num_ind;
            numb->num_ind++;
         }
         else
         {
            local[i] = l;
         }
      }
      else
      {
         local[i] = global[i] - numb->beg_row;
      }
   }
}
