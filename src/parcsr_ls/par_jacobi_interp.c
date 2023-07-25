/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"

/* #define NALU_HYPRE_JACINT_PRINT_ROW_SUMS*/
/* #define NALU_HYPRE_JACINT_PRINT_SOME_ROWS */
/* #define NALU_HYPRE_JACINT_PRINT_MATRICES*/
#define NALU_HYPRE_MAX_PRINTABLE_MATRIX 125
/*#define NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS*/

void nalu_hypre_BoomerAMGJacobiInterp( nalu_hypre_ParCSRMatrix * A,
                                  nalu_hypre_ParCSRMatrix ** P,
                                  nalu_hypre_ParCSRMatrix * S,
                                  NALU_HYPRE_Int num_functions, NALU_HYPRE_Int * dof_func,
                                  NALU_HYPRE_Int * CF_marker, NALU_HYPRE_Int level,
                                  NALU_HYPRE_Real truncation_threshold,
                                  NALU_HYPRE_Real truncation_threshold_minus )
/* nji steps of Jacobi interpolation, with nji presently just set in the code.*/
{
   NALU_HYPRE_Real weight_AF = 1.0;  /* weight multiplied by A's fine row elements */
   NALU_HYPRE_Int * dof_func_offd = NULL;
   NALU_HYPRE_Int nji = 1;
   NALU_HYPRE_Int iji;

   nalu_hypre_ParCSRMatrix_dof_func_offd( A,
                                     num_functions,
                                     dof_func,
                                     &dof_func_offd );

   for ( iji = 0; iji < nji; ++iji )
   {
      nalu_hypre_BoomerAMGJacobiInterp_1( A, P, S, CF_marker, level,
                                     truncation_threshold, truncation_threshold_minus,
                                     dof_func, dof_func_offd,
                                     weight_AF );
   }

   if ( dof_func_offd != NULL )
   {
      nalu_hypre_TFree( dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   }
}

void nalu_hypre_BoomerAMGJacobiInterp_1( nalu_hypre_ParCSRMatrix * A,
                                    nalu_hypre_ParCSRMatrix ** P,
                                    nalu_hypre_ParCSRMatrix * S,
                                    NALU_HYPRE_Int * CF_marker, NALU_HYPRE_Int level,
                                    NALU_HYPRE_Real truncation_threshold,
                                    NALU_HYPRE_Real truncation_threshold_minus,
                                    NALU_HYPRE_Int * dof_func, NALU_HYPRE_Int * dof_func_offd,
                                    NALU_HYPRE_Real weight_AF)
/* One step of Jacobi interpolation:
   A is the linear system.
   P is an interpolation matrix, input and output
   CF_marker identifies coarse and fine points
   If we imagine P and A as split into coarse and fine submatrices,

       [ AFF  AFC ]   [ AF ]            [ IFC ]
   A = [          ] = [    ] ,      P = [     ]
       [ ACF  ACC ]   [ AC ]            [ ICC ]
   (note that ICC is an identity matrix, applied to coarse points only)
   then this function computes

   IFCnew = IFCold - DFF(-1) * ( AFF*IFCold + AFC )
          = IFCold - DFF(-1) * AF * Pold)
   where DFF is the diagonal of AFF, (-1) represents the inverse, and
   where "old" denotes a value on entry to this function, "new" a returned value.

*/
{
   nalu_hypre_ParCSRMatrix * Pnew;
   nalu_hypre_ParCSRMatrix * C;
   nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(*P);
   /*nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(*P);
   NALU_HYPRE_Real      *P_diag_data = nalu_hypre_CSRMatrixData(P_diag);*/
   NALU_HYPRE_Int       *P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
   /*NALU_HYPRE_Int             *P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
   NALU_HYPRE_Real      *P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
   NALU_HYPRE_Int             *P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   nalu_hypre_CSRMatrix *C_diag;
   nalu_hypre_CSRMatrix *C_offd;
   nalu_hypre_CSRMatrix *Pnew_diag;
   nalu_hypre_CSRMatrix *Pnew_offd;*/
   NALU_HYPRE_Int   num_rows_diag_P = nalu_hypre_CSRMatrixNumRows(P_diag);
   NALU_HYPRE_Int i;
   /*NALU_HYPRE_Int Jnochanges=0, Jchanges, Pnew_num_nonzeros*/;
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   NALU_HYPRE_Int CF_coarse = 0;
   NALU_HYPRE_Int nc1;
#endif
   NALU_HYPRE_Int * J_marker = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_rows_diag_P, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int nc, ncmax, ncmin;
   NALU_HYPRE_Int num_procs, my_id;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm( A );
#ifdef NALU_HYPRE_JACINT_PRINT_ROW_SUMS
   NALU_HYPRE_Int m, nmav, npav;
   NALU_HYPRE_Real PIi, PIimax, PIimin, PIimav, PIipav, randthresh;
   NALU_HYPRE_Real eps = 1.0e-17;
#endif
#ifdef NALU_HYPRE_JACINT_PRINT_MATRICES
   char filename[80];
   NALU_HYPRE_Int i_dummy, j_dummy;
   NALU_HYPRE_Int *base_i_ptr = &i_dummy;
   NALU_HYPRE_Int *base_j_ptr = &j_dummy;
#endif
#ifdef NALU_HYPRE_JACINT_PRINT_SOME_ROWS
   NALU_HYPRE_Int sample_rows[50], n_sample_rows = 0, isamp;
#endif

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);


   for ( i = 0; i < num_rows_diag_P; ++i )
   {
      J_marker[i] = CF_marker[i];
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
      if (CF_marker[i] >= 0) { ++CF_coarse; }
#endif
   }
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   nalu_hypre_printf("%i %i Jacobi_Interp_1, P has %i+%i=%i nonzeros, local sum %e\n", my_id, level,
                nalu_hypre_CSRMatrixNumNonzeros(P_diag), nalu_hypre_CSRMatrixNumNonzeros(P_offd),
                nalu_hypre_CSRMatrixNumNonzeros(P_diag) + nalu_hypre_CSRMatrixNumNonzeros(P_offd),
                nalu_hypre_ParCSRMatrixLocalSumElts(*P) );
#endif

   /* row sum computations, for output */
#ifdef NALU_HYPRE_JACINT_PRINT_ROW_SUMS
   PIimax = -1.0e12, PIimin = 1.0e12, PIimav = 0, PIipav = 0;
   nmav = 0, npav = 0;
   for ( i = 0; i < num_rows_diag_P; ++i )
   {
      PIi = 0;  /* i-th value of P*1, i.e. sum of row i of P */
      for ( m = P_diag_i[i]; m < P_diag_i[i + 1]; ++m )
      {
         PIi += P_diag_data[m];
      }
      for ( m = P_offd_i[i]; m < P_offd_i[i + 1]; ++m )
      {
         PIi += P_offd_data[m];
      }
      if (CF_marker[i] < 0)
      {
         PIimax = nalu_hypre_max( PIimax, PIi );
         PIimin = nalu_hypre_min( PIimin, PIi );
         if (PIi <= 1 - eps) { PIimav += PIi; ++nmav; };
         if (PIi >= 1 + eps) { PIipav += PIi; ++npav; };
      }
   }
   if ( nmav > 0 ) { PIimav = PIimav / nmav; }
   if ( npav > 0 ) { PIipav = PIipav / npav; }
   nalu_hypre_printf("%i %i P in max,min row sums %e %e\n", my_id, level, PIimax, PIimin );
#endif

   ncmax = 0; ncmin = num_rows_diag_P;
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   nc1 = 0;
#endif
   for ( i = 0; i < num_rows_diag_P; ++i )
      if (CF_marker[i] < 0)
      {
         nc = P_diag_i[i + 1] - P_diag_i[i];
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
         if (nc <= 1)
         {
            ++nc1;
         }
#endif
         ncmax = nalu_hypre_max( nc, ncmax );
         ncmin = nalu_hypre_min( nc, ncmin );
      }
#if 0
   /* a very agressive reduction in how much the Jacobi step does: */
   for ( i = 0; i < num_rows_diag_P; ++i )
      if (CF_marker[i] < 0)
      {
         nc = P_diag_i[i + 1] - P_diag_i[i];
         if (nc > ncmin + 1)
            /*if ( nc > ncmin + 0.5*(ncmax-ncmin) )*/
         {
            J_marker[i] = 1;
            ++Jnochanges;
         }
      }
#endif

#ifdef NALU_HYPRE_JACINT_PRINT_SOME_ROWS
   Jchanges = num_rows_diag_P - Jnochanges - CF_coarse;
   nalu_hypre_printf("some rows to be changed: ");
   randthresh = 15 / (NALU_HYPRE_Real)Jchanges;
   for ( i = 0; i < num_rows_diag_P; ++i )
   {
      if ( J_marker[i] < 0 )
      {
         if ( ((NALU_HYPRE_Real)nalu_hypre_Rand()) < randthresh )
         {
            nalu_hypre_printf( "%i: ", i );
            for ( m = P_diag_i[i]; m < P_diag_i[i + 1]; ++m )
            {
               nalu_hypre_printf( " %i %f, ", P_diag_j[m], P_diag_data[m] );
            }
            nalu_hypre_printf(";  ");
            sample_rows[n_sample_rows] = i;
            ++n_sample_rows;
         }
      }
   }
   nalu_hypre_printf("\n");
#endif
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   nalu_hypre_printf("%i %i P has %i rows, %i changeable, %i don't change-good, %i coarse\n",
                my_id, level, num_rows_diag_P, Jchanges, Jnochanges, CF_coarse );
   nalu_hypre_printf("%i %i min,max diag cols per row: %i, %i;  no.rows w.<=1 col: %i\n", my_id, level,
                ncmin, ncmax, nc1 );
#endif
#ifdef NALU_HYPRE_JACINT_PRINT_MATRICES
   if ( num_rows_diag_P <= NALU_HYPRE_MAX_PRINTABLE_MATRIX )
   {
      nalu_hypre_sprintf( filename, "Ain%i", level );
      nalu_hypre_ParCSRMatrixPrintIJ( A, 0, 0, filename);
      nalu_hypre_sprintf( filename, "Sin%i", level );
      nalu_hypre_ParCSRMatrixPrintIJ( S, 0, 0, filename);
      nalu_hypre_sprintf( filename, "Pin%i", level );
      nalu_hypre_ParCSRMatrixPrintIJ( *P, 0, 0, filename);
   }
#endif

   C = nalu_hypre_ParMatmul_FC( A, *P, J_marker, dof_func, dof_func_offd );
   /* nalu_hypre_parMatmul_FC creates and returns C, a variation of the
      matrix product A*P in which only the "Fine"-designated rows have
      been computed.  (all columns are Coarse because all columns of P
      are).  "Fine" is defined solely by the marker array, and for
      example could be a proper subset of the fine points of a
      multigrid hierarchy.
      As a matrix, C is the size of A*P.  But only the marked rows have
      been computed.
   */
#ifdef NALU_HYPRE_JACINT_PRINT_MATRICES
   nalu_hypre_sprintf( filename, "C%i", level );
   if ( num_rows_diag_P <= NALU_HYPRE_MAX_PRINTABLE_MATRIX ) { nalu_hypre_ParCSRMatrixPrintIJ( C, 0, 0, filename); }
#endif
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   C_diag = nalu_hypre_ParCSRMatrixDiag(C);
   C_offd = nalu_hypre_ParCSRMatrixOffd(C);
   nalu_hypre_printf("%i %i Jacobi_Interp_1 after matmul, C has %i+%i=%i nonzeros, local sum %e\n",
                my_id, level, nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                nalu_hypre_CSRMatrixNumNonzeros(C_offd),
                nalu_hypre_CSRMatrixNumNonzeros(C_diag) + nalu_hypre_CSRMatrixNumNonzeros(C_offd),
                nalu_hypre_ParCSRMatrixLocalSumElts(C) );
#endif

   nalu_hypre_ParMatScaleDiagInv_F( C, A, weight_AF, J_marker );
   /* nalu_hypre_ParMatScaleDiagInv scales of its first argument by premultiplying with
      a submatrix of the inverse of the diagonal of its second argument.
      The marker array determines which diagonal elements are used.  The marker
      array should select exactly the right number of diagonal elements (the number
      of rows of AP_FC).
   */
#ifdef NALU_HYPRE_JACINT_PRINT_MATRICES
   nalu_hypre_sprintf( filename, "Cout%i", level );
   if ( num_rows_diag_P <= NALU_HYPRE_MAX_PRINTABLE_MATRIX ) { nalu_hypre_ParCSRMatrixPrintIJ( C, 0, 0, filename); }
#endif

   Pnew = nalu_hypre_ParMatMinus_F( *P, C, J_marker );
   /* nalu_hypre_ParMatMinus_F subtracts rows of its second argument from selected rows
      of its first argument.  The marker array determines which rows of the first
      argument are affected, and they should exactly correspond to all the rows
      of the second argument.
   */
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   Pnew_diag = nalu_hypre_ParCSRMatrixDiag(Pnew);
   Pnew_offd = nalu_hypre_ParCSRMatrixOffd(Pnew);
   Pnew_num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(Pnew_diag) + nalu_hypre_CSRMatrixNumNonzeros(Pnew_offd);
   nalu_hypre_printf("%i %i Jacobi_Interp_1 after MatMinus, Pnew has %i+%i=%i nonzeros, local sum %e\n",
                my_id, level, nalu_hypre_CSRMatrixNumNonzeros(Pnew_diag),
                nalu_hypre_CSRMatrixNumNonzeros(Pnew_offd), Pnew_num_nonzeros,
                nalu_hypre_ParCSRMatrixLocalSumElts(Pnew) );
#endif

   nalu_hypre_ParCSRMatrixDestroy( C );
   nalu_hypre_ParCSRMatrixDestroy( *P );

   /* Note that I'm truncating all the fine rows, not just the J-marked ones. */
#if 0
   if ( Pnew_num_nonzeros < 10000 )  /* a fixed number like this makes it no.procs.-depdendent */
   {
      /* ad-hoc attempt to reduce zero-matrix problems seen in testing..*/
      truncation_threshold = 1.0e-6 * truncation_threshold;
      truncation_threshold_minus = 1.0e-6 * truncation_threshold_minus;
   }
#endif
   nalu_hypre_BoomerAMGTruncateInterp( Pnew, truncation_threshold,
                                  truncation_threshold_minus, CF_marker );

   nalu_hypre_MatvecCommPkgCreate ( Pnew );


   *P = Pnew;

   P_diag = nalu_hypre_ParCSRMatrixDiag(*P);
   P_diag_i = nalu_hypre_CSRMatrixI(P_diag);

   /* row sum computations, for output */
#ifdef NALU_HYPRE_JACINT_PRINT_ROW_SUMS
   P_offd = nalu_hypre_ParCSRMatrixOffd(*P);
   P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
   P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
   P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
   P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   PIimax = -1.0e12, PIimin = 1.0e12, PIimav = 0, PIipav = 0;
   nmav = 0, npav = 0;
   for ( i = 0; i < num_rows_diag_P; ++i )
   {
      PIi = 0;  /* i-th value of P*1, i.e. sum of row i of P */
      for ( m = P_diag_i[i]; m < P_diag_i[i + 1]; ++m )
      {
         PIi += P_diag_data[m];
      }
      for ( m = P_offd_i[i]; m < P_offd_i[i + 1]; ++m )
      {
         PIi += P_offd_data[m];
      }
      if (CF_marker[i] < 0)
      {
         PIimax = nalu_hypre_max( PIimax, PIi );
         PIimin = nalu_hypre_min( PIimin, PIi );
         if (PIi <= 1 - eps) { PIimav += PIi; ++nmav; };
         if (PIi >= 1 + eps) { PIipav += PIi; ++npav; };
      }
   }
   if ( nmav > 0 ) { PIimav = PIimav / nmav; }
   if ( npav > 0 ) { PIipav = PIipav / npav; }
   nalu_hypre_printf("%i %i P out max,min row sums %e %e\n", my_id, level, PIimax, PIimin );
#endif

#ifdef NALU_HYPRE_JACINT_PRINT_SOME_ROWS
   nalu_hypre_printf("some changed rows: ");
   for ( isamp = 0; isamp < n_sample_rows; ++isamp )
   {
      i = sample_rows[isamp];
      nalu_hypre_printf( "%i: ", i );
      for ( m = P_diag_i[i]; m < P_diag_i[i + 1]; ++m )
      {
         nalu_hypre_printf( " %i %f, ", P_diag_j[m], P_diag_data[m] );
      }
      nalu_hypre_printf(";  ");
   }
   nalu_hypre_printf("\n");
#endif
   ncmax = 0; ncmin = num_rows_diag_P;
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   nc1 = 0;
#endif
   for ( i = 0; i < num_rows_diag_P; ++i )
      if (CF_marker[i] < 0)
      {
         nc = P_diag_i[i + 1] - P_diag_i[i];
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
         if (nc <= 1) { ++nc1; }
#endif
         ncmax = nalu_hypre_max( nc, ncmax );
         ncmin = nalu_hypre_min( nc, ncmin );
      }
#ifdef NALU_HYPRE_JACINT_PRINT_DIAGNOSTICS
   nalu_hypre_printf("%i %i P has %i rows, %i changeable, %i too good, %i coarse\n",
                my_id, level, num_rows_diag_P, num_rows_diag_P - Jnochanges - CF_coarse, Jnochanges, CF_coarse );
   nalu_hypre_printf("%i %i min,max diag cols per row: %i, %i;  no.rows w.<=1 col: %i\n", my_id, level,
                ncmin, ncmax, nc1 );

   nalu_hypre_printf("%i %i Jacobi_Interp_1 after truncation (%e), Pnew has %i+%i=%i nonzeros, local sum %e\n",
                my_id, level, truncation_threshold,
                nalu_hypre_CSRMatrixNumNonzeros(Pnew_diag), nalu_hypre_CSRMatrixNumNonzeros(Pnew_offd),
                nalu_hypre_CSRMatrixNumNonzeros(Pnew_diag) + nalu_hypre_CSRMatrixNumNonzeros(Pnew_offd),
                nalu_hypre_ParCSRMatrixLocalSumElts(Pnew) );
#endif

   /* Programming Notes:
      1. Judging by around line 299 of par_interp.c, they typical use of CF_marker
      is that CF_marker>=0 means Coarse, CF_marker<0 means Fine.
   */
#ifdef NALU_HYPRE_JACINT_PRINT_MATRICES
   nalu_hypre_sprintf( filename, "Pout%i", level );
   if ( num_rows_diag_P <= NALU_HYPRE_MAX_PRINTABLE_MATRIX ) { nalu_hypre_ParCSRMatrixPrintIJ( *P, 0, 0, filename); }
#endif

   nalu_hypre_TFree( J_marker, NALU_HYPRE_MEMORY_HOST);

}

void nalu_hypre_BoomerAMGTruncateInterp( nalu_hypre_ParCSRMatrix *P,
                                    NALU_HYPRE_Real eps, NALU_HYPRE_Real dlt,
                                    NALU_HYPRE_Int * CF_marker )
/* Truncate the interpolation matrix P, but only in rows for which the
   marker is <0.  Truncation means that an element P(i,j) is set to 0 if
   P(i,j)>0 and P(i,j)<eps*max( P(i,j) )  or if
   P(i,j)>0 and P(i,j)<dlt*max( -P(i,j) )  or if
   P(i,j)<0 and P(i,j)>dlt*min( -P(i,j) )  or if
   P(i,j)<0 and P(i,j)>eps*min( P(i,j) )
      ( 0<eps,dlt<1, typically 0.1=dlt<eps=0.2, )
   The min and max are only computed locally, as I'm guessing that there isn't
   usually much to be gained (in the way of improved performance) by getting
   them perfectly right.
*/

/* The function nalu_hypre_BoomerAMGInterpTruncation in par_interp.c is
   very similar.  It looks at fabs(value) rather than separately
   dealing with value<0 and value>0 as recommended by Klaus Stuben,
   thus as this function does.  In this function, only "marked" rows
   are affected.  Lastly, in nalu_hypre_BoomerAMGInterpTruncation, if any
   element gets discarded, it reallocates arrays to the new size.
*/
{
   nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   NALU_HYPRE_Real      *P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
   NALU_HYPRE_Int       *P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
   NALU_HYPRE_Int       *P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
   NALU_HYPRE_Real      *P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
   NALU_HYPRE_Int       *P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int       *P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
   NALU_HYPRE_Int       *new_P_diag_i;
   NALU_HYPRE_Int       *new_P_offd_i;
   NALU_HYPRE_Int   num_rows_diag_P = nalu_hypre_CSRMatrixNumRows(P_diag);
   NALU_HYPRE_Int   num_rows_offd_P = nalu_hypre_CSRMatrixNumRows(P_offd);
   NALU_HYPRE_Int num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(P_diag);
   NALU_HYPRE_Int num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(P_offd);
#if 0
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm( P );
   NALU_HYPRE_Real vmax1, vmin1;
#endif
   NALU_HYPRE_Real vmax = 0.0;
   NALU_HYPRE_Real vmin = 0.0;
   NALU_HYPRE_Real v, old_sum, new_sum, scale, wmax, wmin;
   NALU_HYPRE_Int i1, m, m1d, m1o;

   /* compute vmax = eps*max(P(i,j)), vmin = eps*min(P(i,j)) */
   for ( i1 = 0; i1 < num_rows_diag_P; i1++ )
   {
      for ( m = P_diag_i[i1]; m < P_diag_i[i1 + 1]; ++m )
      {
         v = P_diag_data[m];
         vmax = nalu_hypre_max( v, vmax );
         vmin = nalu_hypre_min( v, vmin );
      }
      for ( m = P_offd_i[i1]; m < P_offd_i[i1 + 1]; ++m )
      {
         v = P_offd_data[m];
         vmax = nalu_hypre_max( v, vmax );
         vmin = nalu_hypre_min( v, vmin );
      }
   }
#if 0
   /* This can make max,min global so results don't depend on no. processors
      We don't want this except for testing, or maybe this could be put
      someplace better.  I don't like adding communication here, for a minor reason.
   */
   vmax1 = vmax; vmin1 = vmin;
   nalu_hypre_MPI_Allreduce( &vmax1, &vmax, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, comm );
   nalu_hypre_MPI_Allreduce( &vmin1, &vmin, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MIN, comm );
#endif
   if ( vmax <= 0.0 ) { vmax =  1.0; }  /* make sure no v is v>vmax if no v is v>0 */
   if ( vmin >= 0.0 ) { vmin = -1.0; }  /* make sure no v is v<vmin if no v is v<0 */
   wmax = - dlt * vmin;
   wmin = - dlt * vmax;
   vmax *= eps;
   vmin *= eps;

   /* Repack the i,j,and data arrays so as to discard the small elements of P.
      Elements of Coarse rows (CF_marker>=0) are always kept.
      The arrays are not re-allocated, so there will generally be unused space
      at the ends of the arrays. */
   new_P_diag_i = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_rows_diag_P + 1, NALU_HYPRE_MEMORY_HOST);
   new_P_offd_i = nalu_hypre_CTAlloc( NALU_HYPRE_Int,  num_rows_offd_P + 1, NALU_HYPRE_MEMORY_HOST);
   m1d = P_diag_i[0];
   m1o = P_offd_i[0];
   for ( i1 = 0; i1 < num_rows_diag_P; i1++ )
   {
      old_sum = 0;
      new_sum = 0;
      for ( m = P_diag_i[i1]; m < P_diag_i[i1 + 1]; ++m )
      {
         v = P_diag_data[m];
         old_sum += v;
         if ( CF_marker[i1] >= 0 || ( v >= vmax && v >= wmax ) || ( v <= vmin && v <= wmin ) )
         {
            /* keep v */
            new_sum += v;
            P_diag_j[m1d] = P_diag_j[m];
            P_diag_data[m1d] = P_diag_data[m];
            ++m1d;
         }
         else
         {
            /* discard v */
            --num_nonzeros_diag;
         }
      }
      for ( m = P_offd_i[i1]; m < P_offd_i[i1 + 1]; ++m )
      {
         v = P_offd_data[m];
         old_sum += v;
         if ( CF_marker[i1] >= 0 || ( v >= vmax && v >= wmax ) || ( v <= vmin && v <= wmin ) )
         {
            /* keep v */
            new_sum += v;
            P_offd_j[m1o] = P_offd_j[m];
            P_offd_data[m1o] = P_offd_data[m];
            ++m1o;
         }
         else
         {
            /* discard v */
            --num_nonzeros_offd;
         }
      }

      new_P_diag_i[i1 + 1] = m1d;
      if ( i1 < num_rows_offd_P ) { new_P_offd_i[i1 + 1] = m1o; }

      /* rescale to keep row sum the same */
      if (new_sum != 0) { scale = old_sum / new_sum; }
      else { scale = 1.0; }
      for ( m = new_P_diag_i[i1]; m < new_P_diag_i[i1 + 1]; ++m )
      {
         P_diag_data[m] *= scale;
      }
      if ( i1 < num_rows_offd_P ) /* this test fails when there is no offd block */
         for ( m = new_P_offd_i[i1]; m < new_P_offd_i[i1 + 1]; ++m )
         {
            P_offd_data[m] *= scale;
         }

   }

   for ( i1 = 1; i1 <= num_rows_diag_P; i1++ )
   {
      P_diag_i[i1] = new_P_diag_i[i1];
      if ( i1 <= num_rows_offd_P && num_nonzeros_offd > 0 ) { P_offd_i[i1] = new_P_offd_i[i1]; }
   }
   nalu_hypre_TFree( new_P_diag_i, NALU_HYPRE_MEMORY_HOST);
   if ( num_rows_offd_P > 0 ) { nalu_hypre_TFree( new_P_offd_i, NALU_HYPRE_MEMORY_HOST); }

   nalu_hypre_CSRMatrixNumNonzeros(P_diag) = num_nonzeros_diag;
   nalu_hypre_CSRMatrixNumNonzeros(P_offd) = num_nonzeros_offd;
   nalu_hypre_ParCSRMatrixSetDNumNonzeros( P );
   nalu_hypre_ParCSRMatrixSetNumNonzeros( P );

}



/*
  nalu_hypre_ParCSRMatrix_dof_func_offd allocates, computes and returns dof_func_offd.
  The caller is responsible for freeing dof_func_offd.
  This function has code copied from nalu_hypre_BoomerAMGCreateS and nalu_hypre_BoomerAMGCreateSabs
  They should be retrofitted to call this function.  Or, better, call this function separately
  and pass the result into them through an argument (less communication, less computation).
*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrix_dof_func_offd(
   nalu_hypre_ParCSRMatrix    *A,
   NALU_HYPRE_Int                    num_functions,
   NALU_HYPRE_Int                   *dof_func,
   NALU_HYPRE_Int                  **dof_func_offd )
{
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   nalu_hypre_CSRMatrix    *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_Int             num_cols_offd = 0;
   NALU_HYPRE_Int                 Solve_err_flag = 0;
   NALU_HYPRE_Int         num_sends;
   NALU_HYPRE_Int             *int_buf_data;
   NALU_HYPRE_Int         index, start, i, j;

   num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   *dof_func_offd = NULL;
   if (num_cols_offd)
   {
      if (num_functions > 1)
      {
         *dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      }
   }


   /*-------------------------------------------------------------------
     * Get the dof_func data for the off-processor columns
     *-------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   if (num_functions > 1)
   {
      int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                              num_sends), NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                  *dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

   return (Solve_err_flag);
}
