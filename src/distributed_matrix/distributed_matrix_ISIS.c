/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* THIS IS A C++ FILE, since it needs to call ISIS++ with objects */

#ifdef ISIS_AVAILABLE
#include "iostream.h"
#include "RowMatrix.h"  /* ISIS++ header file */
#endif

#include "./distributed_matrix.h"

#ifdef ISIS_AVAILABLE
extern "C" {

typedef struct
{
    NALU_HYPRE_BigInt *ind;
    NALU_HYPRE_Real *val;
} 
RowBuf;
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_InitializeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

NALU_HYPRE_Int 
nalu_hypre_InitializeDistributedMatrixISIS(nalu_hypre_DistributedMatrix *dm)
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) nalu_hypre_DistributedMatrixLocalStorage(dm);

   const Map& map = mat->getMap();

   NALU_HYPRE_BigInt num_rows = mat->getMap().n();
   NALU_HYPRE_BigInt num_cols = mat->getMap().n();
   
   nalu_hypre_DistributedMatrixM(dm) = num_rows;
   nalu_hypre_DistributedMatrixN(dm) = num_cols;

   /* allocate space for row buffers */

   RowBuf *rowbuf = new RowBuf;
   rowbuf->ind = new NALU_HYPRE_BigInt[num_cols];
   rowbuf->val = new NALU_HYPRE_Real[num_cols];

   dm->auxiliary_data = (void *) rowbuf;
#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FreeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_FreeDistributedMatrixISIS( nalu_hypre_DistributedMatrix *dm)
{
#ifdef ISIS_AVAILABLE
   RowBuf *rowbuf = (RowBuf *) dm->auxiliary_data;

   delete rowbuf->ind;
   delete rowbuf->val;
   delete rowbuf;
#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PrintDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_PrintDistributedMatrixISIS( nalu_hypre_DistributedMatrix *matrix )
{
#ifdef ISIS_AVAILABLE
   cout << "nalu_hypre_PrintDistributedMatrixISIS not implemented" << endl;
#endif

   return -1;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GetDistributedMatrixLocalRangeISIS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_GetDistributedMatrixLocalRangeISIS( nalu_hypre_DistributedMatrix *dm,
                             NALU_HYPRE_BigInt *start,
                             NALU_HYPRE_BigInt *end )
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) nalu_hypre_DistributedMatrixLocalStorage(dm);

   *start = mat->getMap().startRow() - 1;  /* convert to 0-based */
   *end = mat->getMap().endRow(); /* endRow actually returns 1 less */
   
   cout << "LocalRangeISIS " << *start << "  " << *end << endl;
#endif
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GetDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

/* semantics: buffers returned will be overwritten on next call to 
 this get function */

NALU_HYPRE_Int 
nalu_hypre_GetDistributedMatrixRowISIS( nalu_hypre_DistributedMatrix *dm,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
#ifdef ISIS_AVAILABLE
   RowMatrix *mat = (RowMatrix *) nalu_hypre_DistributedMatrixLocalStorage(dm);
   RowBuf *rowbuf;
   NALU_HYPRE_Int i, temp;

   rowbuf = (RowBuf *) dm->auxiliary_data;

   mat->getRow(row+1, temp, rowbuf->val, rowbuf->ind);

#if 0
   /* add diagonal element if necessary */
   {
       NALU_HYPRE_Int *p;
       NALU_HYPRE_Int found = 0;

       for (i=0, p=rowbuf->ind; i<temp; i++, p++)
       {
	   if (*p == row+1) 
	       found = 1;
       }

       if (!found)
       {
	   rowbuf->ind[temp] = row+1;
	   rowbuf->val[temp] = 1.; /* pick a value */
	   temp++;
       }
   }
#endif

   /* set pointers to local buffers */
   if (col_ind != NULL)
   {
       NALU_HYPRE_BigInt *p;

       *size = temp;
       *col_ind = rowbuf->ind;

       /* need to convert to 0-based indexing for output */
       for (i=0, p=*col_ind; i<temp; i++, p++)
	   (*p)--;
   }

   if (values != NULL)
   {
       *values = rowbuf->val;
       *size = temp;
   }

#endif

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_RestoreDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_RestoreDistributedMatrixRowISIS( nalu_hypre_DistributedMatrix *dm,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
  /* does nothing, since we use local buffers */

   return 0;
}

#ifdef ISIS_AVAILABLE
} /* extern "C" */
#endif
