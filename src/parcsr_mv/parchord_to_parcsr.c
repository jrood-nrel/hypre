/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* ----------------------------------------------------------------------- */
/*                                                                         */
/*                     ParCSRMatrix to ParChordMatrix                      */
/*                                 and                                     */
/*                     ParCSRMatrix to ParChordMatrix:                     */
/*                                                                         */
/* ----------------------------------------------------------------------- */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_parcsr_mv.h"

void nalu_hypre_ParChordMatrix_RowStarts(
   nalu_hypre_ParChordMatrix *Ac, MPI_Comm comm,
   NALU_HYPRE_BigInt ** row_starts, NALU_HYPRE_BigInt * global_num_cols )
/* This function computes the ParCSRMatrix-style row_starts from a chord matrix.
   It requires the the idofs of the chord matrix be partitioned among
   processors, so their numbering is monotonic with the processor number;
   see below.

   The algorithm: each proc. p knows its min & max global row & col numbers.
   Mins are first_index_rdof[p], first_index_idof[p]
   ***IF*** these were in proper order (see below),
   first_index_rdof[p] is row_starts[p].
   Add num_rdofs-1 to get the max, i.e. add num_rdofs
   to get row_starts[p+1] (IF the processors are ordered thus).
   Compute these, then broadcast to the other processors to form
   row_starts.
   (We also could get global_num_rows by an AllReduce num_idofs.)
   We get global_num_cols by taking the min and max over processors of
   the min and max col no.s on each processor.

   If the chord matrix is not ordered so the above will work, then we
   would need to to completely move matrices around sometimes, a very expensive
   operation.
   The problem is that the chord matrix format makes no assumptions about
   processor order, but the ParCSR format assumes that
   p<q => (local row numbers of p) < (local row numbers of q)
   Maybe instead I could change the global numbering scheme as part of this
   conversion.
   A closely related ordering-type problem to watch for: row_starts must be
   a partition for a ParCSRMatrix.  In a ChordMatrix, the struct itself
   makes no guarantees, but Panayot said, in essence, that row_starts will
   be a partition.
   col_starts should be NULL; later we shall let the Create function compute one.
*/
{
   NALU_HYPRE_BigInt * fis_idof = nalu_hypre_ParChordMatrixFirstindexIdof(Ac);
   NALU_HYPRE_BigInt * fis_rdof = nalu_hypre_ParChordMatrixFirstindexRdof(Ac);
   NALU_HYPRE_Int my_id, num_procs;
   NALU_HYPRE_Int num_idofs = nalu_hypre_ParChordMatrixNumIdofs(Ac);
   NALU_HYPRE_Int num_rdofs = nalu_hypre_ParChordMatrixNumRdofs(Ac);
   NALU_HYPRE_BigInt min_rdof, max_rdof, global_min_rdof, global_max_rdof;
   NALU_HYPRE_Int p, lens[2], lastlens[2];
   nalu_hypre_MPI_Status *status;
   nalu_hypre_MPI_Request *request;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   request = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  1, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  1, NALU_HYPRE_MEMORY_HOST);

   min_rdof = fis_rdof[my_id];
   max_rdof = min_rdof + num_rdofs;
   lens[0] = num_idofs;
   lens[1] = num_rdofs;

   /* row_starts (except last value */
   *row_starts = nalu_hypre_CTAlloc( NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   for ( p = 0; p < num_procs; ++p )
   {
      (*row_starts)[p] = fis_idof[p];
   }

   /* check that ordering and partitioning of rows is as expected
      (much is missing, and even then not perfect)... */
   if ( my_id < num_procs - 1 )
   {
      nalu_hypre_MPI_Isend( lens, 2, NALU_HYPRE_MPI_INT, my_id + 1, 0, comm, request );
   }
   if ( my_id > 0 )
   {
      nalu_hypre_MPI_Recv( lastlens, 2, NALU_HYPRE_MPI_INT, my_id - 1, 0, comm, status );
   }
   if ( my_id < num_procs - 1 )
   {
      nalu_hypre_MPI_Waitall( 1, request, status);
   }
   if ( my_id > 0 )
   {
      nalu_hypre_assert( (*row_starts)[my_id] == (*row_starts)[my_id - 1] + (NALU_HYPRE_BigInt)lastlens[0] );
   }
   nalu_hypre_TFree( request, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( status, NALU_HYPRE_MEMORY_HOST);

   /* Get the upper bound for all the rows */
   nalu_hypre_MPI_Bcast( lens, 2, NALU_HYPRE_MPI_INT, num_procs - 1, comm );
   (*row_starts)[num_procs] = (*row_starts)[num_procs - 1] + (NALU_HYPRE_Int)lens[0];

   /* Global number of columns */
   /*   nalu_hypre_MPI_Allreduce( &num_rdofs, global_num_cols, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm );*/
   nalu_hypre_MPI_Allreduce( &min_rdof, &global_min_rdof, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_MIN, comm );
   nalu_hypre_MPI_Allreduce( &max_rdof, &global_max_rdof, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_MAX, comm );
   *global_num_cols = global_max_rdof - global_min_rdof;
}

NALU_HYPRE_Int
nalu_hypre_ParChordMatrixToParCSRMatrix(
   nalu_hypre_ParChordMatrix *Ac,
   MPI_Comm comm,
   nalu_hypre_ParCSRMatrix **pAp )
{
   /* Some parts of this function are copied from nalu_hypre_CSRMatrixToParCSRMatrix. */

   nalu_hypre_ParCSRMatrix *Ap;
   NALU_HYPRE_BigInt *row_starts, *col_starts;
   NALU_HYPRE_BigInt global_num_rows, global_num_cols;
   NALU_HYPRE_Int my_id, num_procs;
   NALU_HYPRE_Int num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   NALU_HYPRE_Int *local_num_rows;
   /* not computed   NALU_HYPRE_Int          *local_num_nonzeros; */
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_BigInt first_col_diag, last_col_diag;
   NALU_HYPRE_Int i, ic, ij, ir, ilocal, p, r, r_p, r_global, r_local, jlen;
   NALU_HYPRE_Int *a_i, *a_j, *ilen;
   NALU_HYPRE_Int **rdofs, **ps;
   NALU_HYPRE_Complex data;
   NALU_HYPRE_Complex *a_data;
   NALU_HYPRE_Complex **datas;
   nalu_hypre_CSRMatrix *local_A;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_ParChordMatrix_RowStarts
   ( Ac, comm, &row_starts, &global_num_cols );
   /* ... this function works correctly only under some assumptions;
      see the function definition for details */
   global_num_rows = row_starts[num_procs] - row_starts[0];

   col_starts = NULL;
   /* The offd and diag blocks aren't defined until we have both row
      and column partitions... */
   num_cols_offd = 0;
   num_nonzeros_diag = 0;
   num_nonzeros_offd = 0;

   Ap  = nalu_hypre_ParCSRMatrixCreate( comm, global_num_rows, global_num_cols,
                                   row_starts, col_starts,
                                   num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   *pAp = Ap;

   row_starts = nalu_hypre_ParCSRMatrixRowStarts(Ap);
   col_starts = nalu_hypre_ParCSRMatrixColStarts(Ap);

   local_num_rows = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs; i++)
   {
      local_num_rows[i] = (NALU_HYPRE_Int)(row_starts[i + 1] - row_starts[i]);
   }

   num_nonzeros = 0;
   for ( p = 0; p < nalu_hypre_ParChordMatrixNumInprocessors(Ac); ++p )
   {
      num_nonzeros += nalu_hypre_ParChordMatrixNumInchords(Ac)[p];
   };

   local_A = nalu_hypre_CSRMatrixCreate( local_num_rows[my_id], (NALU_HYPRE_Int)global_num_cols,
                                    num_nonzeros );

   /* Compute local CSRMatrix-like i,j arrays for this processor. */

   ps = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  nalu_hypre_ParChordMatrixNumIdofs(Ac), NALU_HYPRE_MEMORY_HOST);
   rdofs = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  nalu_hypre_ParChordMatrixNumIdofs(Ac), NALU_HYPRE_MEMORY_HOST);
   datas = nalu_hypre_CTAlloc( NALU_HYPRE_Complex*,  nalu_hypre_ParChordMatrixNumIdofs(Ac), NALU_HYPRE_MEMORY_HOST);
   ilen  = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  nalu_hypre_ParChordMatrixNumIdofs(Ac), NALU_HYPRE_MEMORY_HOST);
   jlen = 0;
   for ( i = 0; i < nalu_hypre_ParChordMatrixNumIdofs(Ac); ++i )
   {
      ilen[i] = 0;
      ps[i] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  nalu_hypre_ParChordMatrixNumRdofs(Ac), NALU_HYPRE_MEMORY_HOST);
      rdofs[i] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  nalu_hypre_ParChordMatrixNumRdofs(Ac), NALU_HYPRE_MEMORY_HOST);
      datas[i] = nalu_hypre_CTAlloc( NALU_HYPRE_Complex,  nalu_hypre_ParChordMatrixNumRdofs(Ac), NALU_HYPRE_MEMORY_HOST);
      /* ... rdofs[i], datas[i] will generally, not always, be much too big */
   }
   for ( p = 0; p < nalu_hypre_ParChordMatrixNumInprocessors(Ac); ++p )
   {
      for ( ic = 0; ic < nalu_hypre_ParChordMatrixNumInchords(Ac)[p]; ++ic )
      {
         ilocal = nalu_hypre_ParChordMatrixInchordIdof(Ac)[p][ic];
         r = nalu_hypre_ParChordMatrixInchordRdof(Ac)[p][ic];
         data = nalu_hypre_ParChordMatrixInchordData(Ac)[p][ic];
         ps[ilocal][ ilen[ilocal] ] = p;
         rdofs[ilocal][ ilen[ilocal] ] = r;
         datas[ilocal][ ilen[ilocal] ] = data;
         ++ilen[ilocal];
         ++jlen;
      }
   };

   a_i = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  nalu_hypre_ParChordMatrixNumIdofs(Ac) + 1, NALU_HYPRE_MEMORY_HOST);
   a_j = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  jlen, NALU_HYPRE_MEMORY_HOST);
   a_data = nalu_hypre_CTAlloc( NALU_HYPRE_Complex,  jlen, NALU_HYPRE_MEMORY_HOST);
   a_i[0] = 0;
   for ( ilocal = 0; ilocal < nalu_hypre_ParChordMatrixNumIdofs(Ac); ++ilocal )
   {
      a_i[ilocal + 1] = a_i[ilocal] + ilen[ilocal];
      ir = 0;
      for ( ij = a_i[ilocal]; ij < a_i[ilocal + 1]; ++ij )
      {
         p = ps[ilocal][ir];
         r_p = rdofs[ilocal][ir];  /* local in proc. p */
         r_global = r_p + nalu_hypre_ParChordMatrixFirstindexRdof(Ac)[p];
         r_local = r_global - nalu_hypre_ParChordMatrixFirstindexRdof(Ac)[my_id];
         a_j[ij] = r_local;
         a_data[ij] = datas[ilocal][ir];
         ir++;
      };
   };

   for ( i = 0; i < nalu_hypre_ParChordMatrixNumIdofs(Ac); ++i )
   {
      nalu_hypre_TFree( ps[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree( rdofs[i], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree( datas[i], NALU_HYPRE_MEMORY_HOST);
   };
   nalu_hypre_TFree( ps, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( rdofs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( datas, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( ilen, NALU_HYPRE_MEMORY_HOST);

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id + 1] - 1;

   nalu_hypre_CSRMatrixData(local_A) = a_data;
   nalu_hypre_CSRMatrixI(local_A) = a_i;
   nalu_hypre_CSRMatrixJ(local_A) = a_j;
   nalu_hypre_CSRMatrixOwnsData(local_A) = 0;

   GenerateDiagAndOffd(local_A, Ap, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {
      nalu_hypre_TFree(a_data, NALU_HYPRE_MEMORY_HOST);
      /* ... the data has been copied into different diag & offd arrays of Ap */
      nalu_hypre_TFree(a_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(a_i, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixData(local_A) = NULL;
      nalu_hypre_CSRMatrixI(local_A) = NULL;
      nalu_hypre_CSRMatrixJ(local_A) = NULL;
   }
   nalu_hypre_CSRMatrixDestroy(local_A);
   nalu_hypre_TFree(local_num_rows, NALU_HYPRE_MEMORY_HOST);
   /*   nalu_hypre_TFree(csr_matrix_datatypes);*/
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixToParChordMatrix(
   nalu_hypre_ParCSRMatrix *Ap,
   MPI_Comm comm,
   nalu_hypre_ParChordMatrix **pAc )
{
   NALU_HYPRE_BigInt * row_starts = nalu_hypre_ParCSRMatrixRowStarts(Ap);
   NALU_HYPRE_BigInt * col_starts = nalu_hypre_ParCSRMatrixColStarts(Ap);
   nalu_hypre_CSRMatrix * diag = nalu_hypre_ParCSRMatrixDiag(Ap);
   nalu_hypre_CSRMatrix * offd = nalu_hypre_ParCSRMatrixOffd(Ap);
   NALU_HYPRE_Int * offd_j = nalu_hypre_CSRMatrixJ(offd);
   NALU_HYPRE_Int * diag_j = nalu_hypre_CSRMatrixJ(diag);
   NALU_HYPRE_BigInt * col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(Ap);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(Ap);

   nalu_hypre_ParChordMatrix * Ac;
   nalu_hypre_NumbersNode * rdofs, * offd_cols_me;
   nalu_hypre_NumbersNode ** offd_cols;
   NALU_HYPRE_Int ** offd_col_array;
   NALU_HYPRE_Int * len_offd_col_array, * offd_col_array_me;
   NALU_HYPRE_Int len_offd_col_array_me;
   NALU_HYPRE_Int num_idofs, num_rdofs, j_local, j_global, row_global;
   NALU_HYPRE_Int i, j, jj, p, pto, q, qto, my_id, my_q, row, ireq;
   NALU_HYPRE_Int num_inprocessors, num_toprocessors, num_procs, len_num_rdofs_toprocessor;
   NALU_HYPRE_Int *inprocessor, *toprocessor, *pcr, *qcr, *num_inchords, *chord, *chordto;
   NALU_HYPRE_Int *inproc, *toproc, *num_rdofs_toprocessor;
   NALU_HYPRE_Int **inchord_idof, **inchord_rdof, **rdof_toprocessor;
   NALU_HYPRE_Complex **inchord_data;
   NALU_HYPRE_Complex data;
   NALU_HYPRE_Int *first_index_idof, *first_index_rdof;
   nalu_hypre_MPI_Request * request;
   nalu_hypre_MPI_Status * status;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   num_idofs = row_starts[my_id + 1] - row_starts[my_id];
   num_rdofs = col_starts[my_id + 1] - col_starts[my_id];

   nalu_hypre_ParChordMatrixCreate( pAc, comm, num_idofs, num_rdofs );
   Ac = *pAc;

   /* The following block sets Inprocessor:
      On each proc. my_id, we find the columns in the offd and diag blocks
      (global no.s).  The columns are rdofs (contrary to what I wrote in
      ChordMatrix.txt).
      For each such col/rdof r, find the proc. p which owns row/idof r.
      We set the temporary array pcr[p]=1 for such p.
      An MPI all-to-all will exchange such arrays so my_id's array qcr has
      qcr[q]=1 iff, on proc. q, pcr[my_id]=1.  In other words, qcr[q]=1 if
      my_id owns a row/idof i which is the same as a col/rdof owned by q.
      Collect all such q's into in the array Inprocessor.
      While constructing pcr, we also construct pj such that for any index jj
      into offd_j,offd_data, pj[jj] is the processor which owns jj as a row (idof)
      (the number jj is local to this processor).
      */
   pcr = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   qcr = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   for ( p = 0; p < num_procs; ++p ) { pcr[p] = 0; }
   for ( jj = 0; jj < nalu_hypre_CSRMatrixNumNonzeros(offd); ++jj )
   {
      j_local = offd_j[jj];
      j_global =  col_map_offd[j_local];
      for ( p = 0; p < num_procs; ++p )
      {
         if ( j_global >= row_starts[p] && j_global < row_starts[p + 1] )
         {
            pcr[p] = 1;
            /* not used yet...            pj[jj] = p;*/
            break;
         }
      }
   }
   /*   jjd = jj; ...not used yet */

   /* pcr[my_id] = 1; ...for square matrices (with nonzero diag block)
      this one line  would do the job of the following nested loop.
      For non-square matrices, the data distribution is too arbitrary. */
   for ( jj = 0; jj < nalu_hypre_CSRMatrixNumNonzeros(diag); ++jj )
   {
      j_local = diag_j[jj];
      j_global = j_local + first_col_diag;
      for ( p = 0; p < num_procs; ++p )
      {
         if ( j_global >= row_starts[p] && j_global < row_starts[p + 1] )
         {
            pcr[p] = 1;
            /* not used yet...            pj[jj+jjd] = p;*/
            break;
         }
      }
   }


   /* Now pcr[p]=1 iff my_id owns a col/rdof r which proc. p owns as a row/idof */
   nalu_hypre_MPI_Alltoall( pcr, 1, NALU_HYPRE_MPI_INT, qcr, 1, NALU_HYPRE_MPI_INT, comm );
   /* Now qcr[q]==1 if my_id owns a row/idof i which is a col/rdof of proc. q
    The array of such q's is the array Inprocessor. */

   num_inprocessors = 0;
   for ( q = 0; q < num_procs; ++q ) if ( qcr[q] == 1 ) { ++num_inprocessors; }
   inprocessor = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   p = 0;
   for ( q = 0; q < num_procs; ++q ) if ( qcr[q] == 1 ) { inprocessor[ p++ ] = q; }
   num_toprocessors = 0;
   for ( q = 0; q < num_procs; ++q ) if ( pcr[q] == 1 ) { ++num_toprocessors; }
   toprocessor = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_toprocessors, NALU_HYPRE_MEMORY_HOST);
   p = 0;
   for ( q = 0; q < num_procs; ++q ) if ( pcr[q] == 1 ) { toprocessor[ p++ ] = q; }

   nalu_hypre_ParChordMatrixNumInprocessors(Ac) = num_inprocessors;
   nalu_hypre_ParChordMatrixInprocessor(Ac) = inprocessor;
   nalu_hypre_ParChordMatrixNumToprocessors(Ac) = num_toprocessors;
   nalu_hypre_ParChordMatrixToprocessor(Ac) = toprocessor;
   nalu_hypre_TFree( qcr, NALU_HYPRE_MEMORY_HOST);

   /* FirstIndexIdof[p] is the global index of proc. p's row 0 */
   /* FirstIndexRdof[p] is the global index of proc. p's col 0 */
   /* Fir FirstIndexIdof, we copy the array row_starts rather than its pointers,
      because the chord matrix will think it's free to delete FirstIndexIdof */
   /* col_starts[p] contains the global index of the first column
      in the diag block of p.  But for first_index_rdof we want the global
      index of the first column in p (whether that's in the diag or offd block).
      So it's more involved than row/idof: we also check the offd block, and
      have to do a gather to get first_index_rdof for every proc. on every proc. */
   first_index_idof = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   first_index_rdof = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   for ( p = 0; p <= num_procs; ++p )
   {
      first_index_idof[p] = row_starts[p];
      first_index_rdof[p] = col_starts[p];
   };
   if ( nalu_hypre_CSRMatrixNumRows(offd) > 0  && nalu_hypre_CSRMatrixNumCols(offd) > 0 )
      first_index_rdof[my_id] =
         col_starts[my_id] < col_map_offd[0] ? col_starts[my_id] : col_map_offd[0];
   nalu_hypre_MPI_Allgather( &first_index_rdof[my_id], 1, NALU_HYPRE_MPI_INT,
                        first_index_rdof, 1, NALU_HYPRE_MPI_INT, comm );

   /* Set num_inchords: num_inchords[p] is no. chords on my_id connected to p.
      Set each chord (idof,jdof,data).
      We go through each matrix element in the diag block, find what processor
      owns its column no. as a row, then update num_inchords[p], inchord_idof[p],
      inchord_rdof[p], inchord_data[p].
   */

   inchord_idof = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   inchord_rdof = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   inchord_data = nalu_hypre_CTAlloc( NALU_HYPRE_Complex*,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   num_inchords = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   chord = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   chordto = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_toprocessors, NALU_HYPRE_MEMORY_HOST);
   num_rdofs = 0;
   for ( q = 0; q < num_inprocessors; ++q ) { num_inchords[q] = 0; }
   my_q = -1;
   for ( q = 0; q < num_inprocessors; ++q ) if ( inprocessor[q] == my_id ) { my_q = q; }
   nalu_hypre_assert( my_q >= 0 );

   /* diag block: first count chords (from my_id to my_id),
      then set them from diag block's CSR data structure */
   num_idofs = nalu_hypre_CSRMatrixNumRows(diag);
   rdofs = nalu_hypre_NumbersNewNode();
   for ( row = 0; row < nalu_hypre_CSRMatrixNumRows(diag); ++row )
   {
      for ( i = nalu_hypre_CSRMatrixI(diag)[row]; i < nalu_hypre_CSRMatrixI(diag)[row + 1]; ++i )
      {
         j_local = nalu_hypre_CSRMatrixJ(diag)[i];
         nalu_hypre_NumbersEnter( rdofs, j_local );
         ++num_inchords[my_q];
      }
   };
   num_rdofs = nalu_hypre_NumbersNEntered( rdofs );
   inchord_idof[my_q] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inchords[my_q], NALU_HYPRE_MEMORY_HOST);
   inchord_rdof[my_q] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inchords[my_q], NALU_HYPRE_MEMORY_HOST);
   inchord_data[my_q] = nalu_hypre_CTAlloc( NALU_HYPRE_Complex,  num_inchords[my_q], NALU_HYPRE_MEMORY_HOST);
   chord[0] = 0;
   for ( row = 0; row < nalu_hypre_CSRMatrixNumRows(diag); ++row )
   {
      for ( i = nalu_hypre_CSRMatrixI(diag)[row]; i < nalu_hypre_CSRMatrixI(diag)[row + 1]; ++i )
      {
         j_local = nalu_hypre_CSRMatrixJ(diag)[i];
         data = nalu_hypre_CSRMatrixData(diag)[i];
         inchord_idof[my_q][chord[0]] = row;
         /* Here We need to convert from j_local - a column local to
            the diag of this proc., to a j which is local only to this
            processor - a column (rdof) numbering scheme to be shared by the
            diag and offd blocks...  */
         j_global = j_local + nalu_hypre_ParCSRMatrixColStarts(Ap)[my_q];
         j = j_global - first_index_rdof[my_q];
         inchord_rdof[my_q][chord[0]] = j;
         inchord_data[my_q][chord[0]] = data;
         nalu_hypre_assert( chord[0] < num_inchords[my_q] );
         ++chord[0];
      }
   };
   nalu_hypre_NumbersDeleteNode(rdofs);


   /* offd block: */

   /* offd_cols_me duplicates rdofs */
   offd_cols_me = nalu_hypre_NumbersNewNode();
   for ( row = 0; row < nalu_hypre_CSRMatrixNumRows(offd); ++row )
   {
      for ( i = nalu_hypre_CSRMatrixI(offd)[row]; i < nalu_hypre_CSRMatrixI(offd)[row + 1]; ++i )
      {
         j_local = nalu_hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         nalu_hypre_NumbersEnter( offd_cols_me, j_global );
      }
   }
   offd_col_array = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   len_offd_col_array = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   offd_col_array_me = nalu_hypre_NumbersArray( offd_cols_me );
   len_offd_col_array_me = nalu_hypre_NumbersNEntered( offd_cols_me );
   request = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  2 * num_procs, NALU_HYPRE_MEMORY_HOST);
   ireq = 0;
   for ( q = 0; q < num_inprocessors; ++q )
      nalu_hypre_MPI_Irecv( &len_offd_col_array[q], 1, NALU_HYPRE_MPI_INT,
                       inprocessor[q], 0, comm, &request[ireq++] );
   for ( p = 0; p < num_procs; ++p ) if ( pcr[p] == 1 )
      {
         nalu_hypre_MPI_Isend( &len_offd_col_array_me, 1, NALU_HYPRE_MPI_INT, p, 0, comm, &request[ireq++] );
      }
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  ireq, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Waitall( ireq, request, status );
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   ireq = 0;
   for ( q = 0; q < num_inprocessors; ++q )
   {
      offd_col_array[q] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  len_offd_col_array[q], NALU_HYPRE_MEMORY_HOST);
   }
   for ( q = 0; q < num_inprocessors; ++q )
      nalu_hypre_MPI_Irecv( offd_col_array[q], len_offd_col_array[q], NALU_HYPRE_MPI_INT,
                       inprocessor[q], 0, comm, &request[ireq++] );
   for ( p = 0; p < num_procs; ++p ) if ( pcr[p] == 1 )
      {
         nalu_hypre_MPI_Isend( offd_col_array_me, len_offd_col_array_me,
                          NALU_HYPRE_MPI_INT, p, 0, comm, &request[ireq++] );
      }
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  ireq, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Waitall( ireq, request, status );
   nalu_hypre_TFree(request, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   offd_cols = nalu_hypre_CTAlloc( nalu_hypre_NumbersNode *,  num_inprocessors, NALU_HYPRE_MEMORY_HOST);
   for ( q = 0; q < num_inprocessors; ++q )
   {
      offd_cols[q] = nalu_hypre_NumbersNewNode();
      for ( i = 0; i < len_offd_col_array[q]; ++i )
      {
         nalu_hypre_NumbersEnter( offd_cols[q], offd_col_array[q][i] );
      }
   }

   len_num_rdofs_toprocessor = 1 + nalu_hypre_CSRMatrixI(offd)
                               [nalu_hypre_CSRMatrixNumRows(offd)];
   inproc = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  len_num_rdofs_toprocessor, NALU_HYPRE_MEMORY_HOST);
   toproc = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  len_num_rdofs_toprocessor, NALU_HYPRE_MEMORY_HOST);
   num_rdofs_toprocessor = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  len_num_rdofs_toprocessor, NALU_HYPRE_MEMORY_HOST);
   for ( qto = 0; qto < len_num_rdofs_toprocessor; ++qto )
   {
      inproc[qto] = -1;
      toproc[qto] = -1;
      num_rdofs_toprocessor[qto] = 0;
   };
   rdofs = nalu_hypre_NumbersNewNode();
   for ( row = 0; row < nalu_hypre_CSRMatrixNumRows(offd); ++row )
   {
      for ( i = nalu_hypre_CSRMatrixI(offd)[row]; i < nalu_hypre_CSRMatrixI(offd)[row + 1]; ++i )
      {
         j_local = nalu_hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         nalu_hypre_NumbersEnter( rdofs, j_local );

         /* TO DO: find faster ways to do the two processor lookups below.*/
         /* Find a processor p (local index q) from the inprocessor list,
            which owns the column(rdof) whichis the same as this processor's
            row(idof) row. Update num_inchords for p.
            Save q as inproc[i] for quick recall later.  It represents
            an inprocessor (not unique) connected to a chord i.
         */
         inproc[i] = -1;
         for ( q = 0; q < num_inprocessors; ++q ) if (q != my_q)
            {
               p = inprocessor[q];
               if ( nalu_hypre_NumbersQuery( offd_cols[q],
                                        row + nalu_hypre_ParCSRMatrixFirstRowIndex(Ap) )
                    == 1 )
               {
                  /* row is one of the offd columns of p */
                  ++num_inchords[q];
                  inproc[i] = q;
                  break;
               }
            }
         if ( inproc[i] < 0 )
         {
            /* For square matrices, we would have found the column in some
               other processor's offd.  But for non-square matrices it could
               exist only in some other processor's diag...*/
            /* Note that all data in a diag block is stored.  We don't check
               whether the value of a data entry is zero. */
            for ( q = 0; q < num_inprocessors; ++q ) if (q != my_q)
               {
                  p = inprocessor[q];
                  row_global = row + nalu_hypre_ParCSRMatrixFirstRowIndex(Ap);
                  if ( row_global >= col_starts[p] &&
                       row_global < col_starts[p + 1] )
                  {
                     /* row is one of the diag columns of p */
                     ++num_inchords[q];
                     inproc[i] = q;
                     break;
                  }
               }
         }
         nalu_hypre_assert( inproc[i] >= 0 );

         /* Find the processor pto (local index qto) from the toprocessor list,
            which owns the row(idof) which is the  same as this processor's
            column(rdof) j_global. Update num_rdofs_toprocessor for pto.
            Save pto as toproc[i] for quick recall later. It represents
            the toprocessor connected to a chord i. */
         for ( qto = 0; qto < num_toprocessors; ++qto )
         {
            pto = toprocessor[qto];
            if ( j_global >= row_starts[pto] && j_global < row_starts[pto + 1] )
            {
               nalu_hypre_assert( qto < len_num_rdofs_toprocessor );
               ++num_rdofs_toprocessor[qto];
               /* ... an overestimate, as if two chords share an rdof, that
                  rdof will be counted twice in num_rdofs_toprocessor.
                  It can be fixed up later.*/
               toproc[i] = qto;
               break;
            }
         }
      }
   };
   num_rdofs += nalu_hypre_NumbersNEntered(rdofs);
   nalu_hypre_NumbersDeleteNode(rdofs);

   for ( q = 0; q < num_inprocessors; ++q ) if (q != my_q)
      {
         inchord_idof[q] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inchords[q], NALU_HYPRE_MEMORY_HOST);
         inchord_rdof[q] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_inchords[q], NALU_HYPRE_MEMORY_HOST);
         inchord_data[q] = nalu_hypre_CTAlloc( NALU_HYPRE_Complex,  num_inchords[q], NALU_HYPRE_MEMORY_HOST);
         chord[q] = 0;
      };
   for ( q = 0; q < num_inprocessors; ++q ) if (q != my_q)
      {
         for ( i = 0; i < num_inchords[q]; ++i )
         {
            inchord_idof[q][i] = -1;
         }
      };
   rdof_toprocessor = nalu_hypre_CTAlloc( NALU_HYPRE_Int*,  num_toprocessors, NALU_HYPRE_MEMORY_HOST);
   for ( qto = 0; qto < num_toprocessors; ++qto ) /*if (qto!=my_q)*/
   {
      nalu_hypre_assert( qto < len_num_rdofs_toprocessor );
      rdof_toprocessor[qto] = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_rdofs_toprocessor[qto], NALU_HYPRE_MEMORY_HOST);
      chordto[qto] = 0;
   };
   for ( row = 0; row < nalu_hypre_CSRMatrixNumRows(offd); ++row )
   {
      for ( i = nalu_hypre_CSRMatrixI(offd)[row]; i < nalu_hypre_CSRMatrixI(offd)[row + 1]; ++i )
      {
         j_local = nalu_hypre_CSRMatrixJ(offd)[i];
         j_global =  col_map_offd[j_local];
         data = nalu_hypre_CSRMatrixData(offd)[i];
         qto = toproc[i];
         q = inproc[i];
         nalu_hypre_assert( q != my_q );
         nalu_hypre_assert( chord[q] < num_inchords[q] );
         inchord_idof[q][chord[q]] = row;
         j = j_global - first_index_rdof[q];
         inchord_rdof[q][chord[q]] = j;
         inchord_data[q][chord[q]] = data;
         /* Note that although inchord_* is organized according to the
            inprocessors, the rdof has the local number of a toprocessor -
            the only thing which makes sense and fits with what I've been
            told about chord matrices. */
         nalu_hypre_assert( chord[q] < num_inchords[q] );
         ++chord[q];
         if ( qto >= 0 )
         {
            /* There is an rdof processor for this chord */
            rdof_toprocessor[qto][chordto[qto]] = j;
            ++chordto[qto];
         }
      }
   };
   /* fix up overestimate of num_rdofs_toprocessor.  We're not going to
      bother to fix the excessive size which has been allocated to
      rdof_toprocessor... */
   for ( qto = 0; qto < num_toprocessors; ++qto ) /*if (qto!=my_q)*/
   {
      num_rdofs_toprocessor[qto] = chordto[qto] - 1;
   }
   nalu_hypre_NumbersDeleteNode( offd_cols_me );
   for ( q = 0; q < num_inprocessors; ++q )
   {
      nalu_hypre_NumbersDeleteNode( offd_cols[q]);
   }
   nalu_hypre_TFree( offd_cols, NALU_HYPRE_MEMORY_HOST);
   for ( q = 0; q < num_inprocessors; ++q )
   {
      nalu_hypre_TFree( offd_col_array[q], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree( offd_col_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( len_offd_col_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( chordto, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( inproc, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( toproc, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( chord, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree( pcr, NALU_HYPRE_MEMORY_HOST);


   nalu_hypre_ParChordMatrixFirstindexIdof(Ac) = first_index_idof;
   nalu_hypre_ParChordMatrixFirstindexRdof(Ac) = first_index_rdof;

   nalu_hypre_ParChordMatrixNumInchords(Ac) = num_inchords;
   nalu_hypre_ParChordMatrixInchordIdof(Ac) = inchord_idof;
   nalu_hypre_ParChordMatrixInchordRdof(Ac) = inchord_rdof;
   nalu_hypre_ParChordMatrixInchordData(Ac) = inchord_data;
   nalu_hypre_ParChordMatrixNumIdofs(Ac) = num_idofs;
   nalu_hypre_ParChordMatrixNumRdofs(Ac) = num_rdofs;
   nalu_hypre_ParChordMatrixNumRdofsToprocessor(Ac) = num_rdofs_toprocessor;
   nalu_hypre_ParChordMatrixRdofToprocessor(Ac) = rdof_toprocessor;


   /* >>> to set...

      nalu_hypre_ParChordMatrixNumIdofsInprocessor(Ac)  (low priority - not used);
      nalu_hypre_ParChordMatrixIdofInprocessor(Ac)  (low priority - not used);
   */

   return 0;
}


