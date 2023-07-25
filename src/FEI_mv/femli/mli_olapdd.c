/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

extern int MLI_Smoother_Apply_Schwarz(void *smoother_obj,nalu_hypre_ParCSRMatrix *A,
                                        nalu_hypre_ParVector *f,nalu_hypre_ParVector *u);

/******************************************************************************
 * Schwarz relaxation scheme 
 *****************************************************************************/

typedef struct MLI_Smoother_Schwarz_Struct
{
   nalu_hypre_ParCSRMatrix *Amat;
   ParaSails          *ps;
   int                factorized;
} MLI_Smoother_Schwarz;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Create_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Create_Schwarz(void **smoother_obj) 
{
   MLI_Smoother_Schwarz *smoother;

   smoother = nalu_hypre_CTAlloc( MLI_Smoother_Schwarz,  1 , NALU_HYPRE_MEMORY_HOST);
   if ( smoother == NULL ) { (*smoother_obj) = NULL; return 1; }
   smoother->Amat = NULL;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Destroy_Schwarz(void *smoother_obj)
{
   MLI_Smoother_Schwarz *smoother;

   smoother = (MLI_Smoother_Schwarz *) smoother_obj;
   if ( smoother != NULL ) nalu_hypre_TFree( smoother , NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_Schwarz
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_Schwarz(void *smoother_obj, 
                               int (**smoother_func)(void *smoother_obj, 
                                nalu_hypre_ParCSRMatrix *A,nalu_hypre_ParVector *f,
                                nalu_hypre_ParVector *u), nalu_hypre_ParCSRMatrix *A, 
{
   int                    *partition, mypid, start_row, end_row;
   int                    row, row_length, *col_indices;
   double                 *col_values;
   Matrix                 *mat;
   ParaSails              *ps;
   MLI_Smoother_ParaSails *smoother;
   MPI_Comm               comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning((NALU_HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;

   /*-----------------------------------------------------------------
    * construct a ParaSails matrix
    *-----------------------------------------------------------------*/

   mat = MatrixCreate(comm, start_row, end_row);
   for (row = start_row; row <= end_row; row++)
   {
      nalu_hypre_ParCSRMatrixGetRow(A, row, &row_length, &col_indices, &col_values);
      MatrixSetRow(mat, row, row_length, col_indices, col_values);
      nalu_hypre_ParCSRMatrixRestoreRow(A,row,&row_length,&col_indices,&col_values);
   }
   MatrixComplete(mat);

   /*-----------------------------------------------------------------
    * construct a ParaSails smoother object
    *-----------------------------------------------------------------*/

   smoother = nalu_hypre_CTAlloc( MLI_Smoother_ParaSails,  1 , NALU_HYPRE_MEMORY_HOST);
   if ( smoother == NULL ) { (*smoother_obj) = NULL; return 1; }
   ps = ParaSailsCreate(comm, start_row, end_row, parasails_factorized);
   ps->loadbal_beta = parasails_loadbal;
   ParaSailsSetupPattern(ps, mat, thresh, num_levels);
   ParaSailsStatsPattern(ps, mat);
   ParaSailsSetupValues(ps, mat, filter);
   ParaSailsStatsValues(ps, mat);
   smoother->factorized = parasails_factorized;
   smoother->ps = ps;
   smoother->Amat = A;

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   MatrixDestroy(mat);
   (*smoother_obj) = (void *) smoother;
   if ( trans ) (*smoother_func) = MLI_Smoother_Apply_ParaSailsTrans;
   else         (*smoother_func) = MLI_Smoother_Apply_ParaSails;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_ParaSails
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_ParaSails(void *smoother_obj, nalu_hypre_ParCSRMatrix *A,
                                 nalu_hypre_ParVector *f, nalu_hypre_ParVector    *u)
{
   nalu_hypre_CSRMatrix        *A_diag;
   nalu_hypre_ParVector        *Vtemp;
   nalu_hypre_Vector           *u_local, *Vtemp_local;
   double                 *u_data, *Vtemp_data;
   int                    i, n, relax_error = 0, global_size;
   int                    num_procs, *partition1, *partition2;
   int                    parasails_factorized;
   double                 *tmp_data;
   MPI_Comm               comm;
   MLI_Smoother_ParaSails *smoother;
   ParaSails              *ps;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat;
   comm          = nalu_hypre_ParCSRMatrixComm(A);
   A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   n             = nalu_hypre_CSRMatrixNumRows(A_diag);
   u_local       = nalu_hypre_ParVectorLocalVector(u);
   u_data        = nalu_hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size = nalu_hypre_ParVectorGlobalSize(f);
   partition1  = nalu_hypre_ParVectorPartitioning(f);
   partition2  = nalu_hypre_CTAlloc( int,  num_procs+1 , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = nalu_hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = nalu_hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   nalu_hypre_ParVectorCopy(f, Vtemp);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = nalu_hypre_CTAlloc( double,  n , NALU_HYPRE_MEMORY_HOST);

   parasails_factorized = smoother->factorized;

   if (!parasails_factorized)
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   nalu_hypre_TFree( tmp_data , NALU_HYPRE_MEMORY_HOST);

   return(relax_error); 
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_ParaSailsTrans
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_ParaSailsTrans(void *smoother_obj,nalu_hypre_ParCSRMatrix *A,
                                      nalu_hypre_ParVector *f,nalu_hypre_ParVector *u)
{
   nalu_hypre_CSRMatrix        *A_diag;
   nalu_hypre_ParVector        *Vtemp;
   nalu_hypre_Vector           *u_local, *Vtemp_local;
   double                 *u_data, *Vtemp_data;
   int                    i, n, relax_error = 0, global_size;
   int                    num_procs, *partition1, *partition2;
   int                    parasails_factorized;
   double                 *tmp_data;
   MPI_Comm               comm;
   MLI_Smoother_ParaSails *smoother;
   ParaSails              *ps;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   smoother      = (MLI_Smoother_ParaSails *) smoother_obj;
   A             = smoother->Amat;
   comm          = nalu_hypre_ParCSRMatrixComm(A);
   A_diag        = nalu_hypre_ParCSRMatrixDiag(A);
   n             = nalu_hypre_CSRMatrixNumRows(A_diag);
   u_local       = nalu_hypre_ParVectorLocalVector(u);
   u_data        = nalu_hypre_VectorData(u_local);

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size = nalu_hypre_ParVectorGlobalSize(f);
   partition1  = nalu_hypre_ParVectorPartitioning(f);
   partition2  = nalu_hypre_CTAlloc( int,  num_procs+1 , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = nalu_hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = nalu_hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   nalu_hypre_ParVectorCopy(f, Vtemp);
   nalu_hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = nalu_hypre_CTAlloc( double,  n , NALU_HYPRE_MEMORY_HOST);

   parasails_factorized = smoother->factorized;

   if (!parasails_factorized)
   {
      MatrixMatvecTrans(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   nalu_hypre_TFree( tmp_data , NALU_HYPRE_MEMORY_HOST);

   return(relax_error); 
}
#endif

