/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define habs(x) ((x > 0 ) ? x : -(x))

/*-------------------------------------------------------------------------*/
/* parcsr_mv.h is put here instead of in NALU_HYPRE_LinSysCore.h     */
/* because it gives warning when compiling cfei.cc                         */
/*-------------------------------------------------------------------------*/

#include "utilities/_hypre_utilities.h"

#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
/* RDF: What is MPIAPI? */
#ifndef MPIAPI
#define MPIAPI
#endif

int  NALU_HYPRE_DummySetup(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                      NALU_HYPRE_ParVector x_csr, NALU_HYPRE_ParVector y_csr ){return 0;}

void NALU_HYPRE_LSI_Get_IJAMatrixFromFile(double**,int**,int**,int*,double**,
                                     char*,char*);
extern int MPIAPI MPI_Comm_split(MPI_Comm, int, int, MPI_Comm *);

/***************************************************************************/
/***************************************************************************/
/* This section investigates the use of domain decomposition preconditioner*/
/* using AMG.                                                              */
/***************************************************************************/
/***************************************************************************/

/***************************************************************************/
/* local variables for preconditioning (bad idea, but...)                  */
/***************************************************************************/

NALU_HYPRE_IJMatrix localA;
NALU_HYPRE_IJVector localb;
NALU_HYPRE_IJVector localx;
int            myBegin, myEnd, myRank;
int            interior_nrows, *offRowLengths;
int            **offColInd;
int            *remap_array;
double         **offColVal;
MPI_Comm       parComm;
NALU_HYPRE_Solver   cSolver;
NALU_HYPRE_Solver   cPrecon;

/***************************************************************************/
/* Apply [I   ]                                                            */
/*       [E_ob] vb                                                         */
/***************************************************************************/

int NALU_HYPRE_LocalAMGSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x_csr,
                        NALU_HYPRE_ParVector y_csr )
{
   int                i, local_nrows, *temp_list;
   NALU_HYPRE_ParCSRMatrix LA_csr;
   NALU_HYPRE_ParVector    Lx_csr;
   NALU_HYPRE_ParVector    Lb_csr;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   double             *x_par_data ;
   double             *y_par_data ;
   double             *temp_vect;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;

   /* --------------------------------------------------------*/
   /* fetch data pointer of input and output vectors          */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);

   /* --------------------------------------------------------*/
   /* create localb & localx of length = no. of interior nodes*/
   /* --------------------------------------------------------*/

   temp_list = hypre_TAlloc(int, interior_nrows , NALU_HYPRE_MEMORY_HOST);
   temp_vect = hypre_TAlloc(double, interior_nrows , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < interior_nrows; i++) temp_list[i] = i;
   for (i = 0; i < local_nrows; i++)
   {
      if (remap_array[i] >= 0) temp_vect[remap_array[i]] = x_par_data[i];
   }
   NALU_HYPRE_IJVectorSetValues(localb,interior_nrows,(const int *) temp_list,
                           temp_vect);
   hypre_TFree(temp_list, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(temp_vect, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* perform one cycle of AMG to subdomain (internal nodes)  */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);

   NALU_HYPRE_BoomerAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   /* --------------------------------------------------------*/
   /* update interior nodes, leave boundary nodes unchanged   */
   /* --------------------------------------------------------*/

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i = 0; i < local_nrows; i++)
   {
      if (remap_array[i] >= 0) y_par_data[i] = Lx_data[remap_array[i]];
   }
   return 0;
}

/***************************************************************************/
/* Apply [I   ]                                                            */
/*       [E_ob] vb                                                         */
/***************************************************************************/

int NALU_HYPRE_ApplyExtension(NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x_csr,
                         NALU_HYPRE_ParVector y_csr )
{
   int                i, j, index, local_nrows, global_nrows, *temp_list;
   NALU_HYPRE_ParCSRMatrix LA_csr;
   NALU_HYPRE_ParVector    Lx_csr;
   NALU_HYPRE_ParVector    Lb_csr;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   double             *x_par_data ;
   double             *y_par_data ;
   double             *temp_vect;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;

   /* --------------------------------------------------------*/
   /* get local and global size of vectors                    */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows,&global_nrows,1,MPI_INT,MPI_SUM,parComm);

   /* --------------------------------------------------------*/
   /* fetch data pointer of input and output vectors          */
   /* --------------------------------------------------------*/

   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);

   /* --------------------------------------------------------*/
   /* copy from x to temporary vector                         */
   /* --------------------------------------------------------*/

   index = 0;
   for (i = 0; i < local_nrows; i++)
   {
      if ( remap_array[i] < 0 ) y_par_data[i] = x_par_data[index++];
      else                      y_par_data[i] = 0.0;
   }

   /* --------------------------------------------------------*/
   /* create localb & localx of length = no. of interior nodes*/
   /* --------------------------------------------------------*/

   temp_list = hypre_TAlloc(int,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   temp_vect = hypre_TAlloc(double,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < interior_nrows; i++) temp_list[i] = i;
   for (i = 0; i < local_nrows; i++)
   {
      if (remap_array[i] >= 0 && remap_array[i] < interior_nrows)
      {
         temp_vect[remap_array[i]] = 0.0;
         for (j = 0; j < offRowLengths[i]; j++)
            temp_vect[remap_array[i]] +=
               (offColVal[i][j] * y_par_data[offColInd[i][j]]);
      } else if ( remap_array[i] >= interior_nrows)
        printf("WARNING : index out of range.\n");
   }
   NALU_HYPRE_IJVectorSetValues(localb,interior_nrows,(const int*) temp_list,
                           temp_vect);
   hypre_TFree(temp_list, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(temp_vect, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* perform one cycle of AMG to subdomain (internal nodes)  */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);
   NALU_HYPRE_BoomerAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   /* --------------------------------------------------------*/
   /* update interior nodes, leave boundary nodes unchanged   */
   /* --------------------------------------------------------*/

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++)
   {
      if (remap_array[i] >= 0) y_par_data[i] = -Lx_data[remap_array[i]];
   }
   return 0;
}

/***************************************************************************/
/* Apply [I E_ob^T] v                                                      */
/***************************************************************************/

int NALU_HYPRE_ApplyExtensionTranspose(NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x_csr,
                                  NALU_HYPRE_ParVector y_csr )
{
   int                i, j, index, local_nrows, global_nrows, *temp_list;
   NALU_HYPRE_IJVector     tvec;
   NALU_HYPRE_ParCSRMatrix LA_csr;
   NALU_HYPRE_ParVector    Lx_csr;
   NALU_HYPRE_ParVector    Lb_csr;
   NALU_HYPRE_ParVector    t_csr;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_ParVector    *t_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   hypre_Vector       *t_par_local;
   double             *x_par_data ;
   double             *y_par_data ;
   double             *t_par_data ;
   double             *temp_vect;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;

   /* --------------------------------------------------------*/
   /* get local and global size of vectors                    */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows,&global_nrows,1,MPI_INT,MPI_SUM,parComm);

   /* --------------------------------------------------------*/
   /* create a temporary long vector                          */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &tvec);
   NALU_HYPRE_IJVectorSetObjectType(tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(tvec);
   NALU_HYPRE_IJVectorAssemble(tvec);
   NALU_HYPRE_IJVectorGetObject(tvec, (void **) &t_csr);
   t_par       = (hypre_ParVector *) t_csr;
   t_par_local = hypre_ParVectorLocalVector(t_par);
   t_par_data  = hypre_VectorData(t_par_local);

   /* --------------------------------------------------------*/
   /* fetch data pointer of input and output vectors          */
   /* --------------------------------------------------------*/

   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);

   /* --------------------------------------------------------*/
   /* create localb & localx of length = no. of interior nodes*/
   /* --------------------------------------------------------*/

   temp_list = hypre_TAlloc(int,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   temp_vect = hypre_TAlloc(double,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   for (i=0; i<interior_nrows; i++) temp_list[i] = i;
   for (i=0; i<local_nrows; i++)
   {
      if (remap_array[i] >= 0 && remap_array[i] < interior_nrows)
         temp_vect[remap_array[i]] = x_par_data[i];
   }
   NALU_HYPRE_IJVectorSetValues(localb,interior_nrows,(const int*) temp_list,
                           temp_vect);
   hypre_TFree(temp_list, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(temp_vect, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* perform one cycle of AMG to subdomain (internal nodes)  */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);

   NALU_HYPRE_BoomerAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   /* --------------------------------------------------------*/
   /* update boundary nodes                                   */
   /* --------------------------------------------------------*/

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++)
   {
      if ( remap_array[i] >= 0 )
      {
         for (j=0; j<offRowLengths[i]; j++)
         {
            index = offColInd[i][j];
            t_par_data[index] -= (Lx_data[remap_array[i]] * offColVal[i][j]);
         }
      }
   }

   /* --------------------------------------------------------*/
   /* extract boundary nodes                                  */
   /* --------------------------------------------------------*/

   index = 0;
   for (i=0; i<local_nrows; i++)
   {
      if (remap_array[i] < 0)
         y_par_data[index++] = x_par_data[i] - t_par_data[i];
   }

   /* --------------------------------------------------------*/
   /* clean up                                                */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorDestroy(tvec);

   return 0;
}

/***************************************************************************/
/* Apply E to an incoming vector                                           */
/***************************************************************************/

int NALU_HYPRE_ApplyTransform( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x_csr,
                  NALU_HYPRE_ParVector y_csr )
{
   int                i, j, local_nrows, *temp_list;
   NALU_HYPRE_ParCSRMatrix LA_csr;
   NALU_HYPRE_ParVector    Lx_csr;
   NALU_HYPRE_ParVector    Lb_csr;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   double             *x_par_data ;
   double             *y_par_data ;
   double             *temp_vect;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;

   /* --------------------------------------------------------*/
   /* get local and global size of vectors                    */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;

   /* --------------------------------------------------------*/
   /* fetch data pointer of input and output vectors          */
   /* --------------------------------------------------------*/

   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);

   /* --------------------------------------------------------*/
   /* copy from x to temporary vector                         */
   /* --------------------------------------------------------*/

   for (i = 0; i < local_nrows; i++) y_par_data[i] = x_par_data[i];

   /* --------------------------------------------------------*/
   /* create localb & localx of length = no. of interior nodes*/
   /* --------------------------------------------------------*/

   temp_list = hypre_TAlloc(int,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   temp_vect = hypre_TAlloc(double,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < interior_nrows; i++) temp_list[i] = i;
   for (i = 0; i < local_nrows; i++)
   {
      if ( remap_array[i] >= 0 && remap_array[i] < interior_nrows)
      {
         temp_vect[remap_array[i]] = 0.0;
         for (j = 0; j < offRowLengths[i]; j++)
            temp_vect[remap_array[i]] +=
               (offColVal[i][j] * x_par_data[offColInd[i][j]]);
      } else if ( remap_array[i] >= interior_nrows)
        printf("WARNING : index out of range.\n");
   }
   NALU_HYPRE_IJVectorSetValues(localb,interior_nrows,(const int*) temp_list,
                           temp_vect);
   hypre_TFree(temp_list, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(temp_vect, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* perform one cycle of AMG to subdomain (internal nodes)  */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);

   NALU_HYPRE_BoomerAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   /* --------------------------------------------------------*/
   /* update interior nodes, leave boundary nodes unchanged   */
   /* --------------------------------------------------------*/

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++)
   {
      if (remap_array[i] >= 0) y_par_data[i] -= Lx_data[remap_array[i]];
   }
   return 0;
}

/***************************************************************************/
/* Apply E^T to an incoming vector                                         */
/***************************************************************************/

int NALU_HYPRE_ApplyTransformTranspose(NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector x_csr,
                                  NALU_HYPRE_ParVector y_csr )
{
   int                i, j, index, local_nrows, *temp_list;
   NALU_HYPRE_ParCSRMatrix LA_csr;
   NALU_HYPRE_ParVector    Lx_csr;
   NALU_HYPRE_ParVector    Lb_csr;
   hypre_ParVector    *x_par;
   hypre_ParVector    *y_par;
   hypre_Vector       *x_par_local;
   hypre_Vector       *y_par_local;
   double             *x_par_data ;
   double             *y_par_data ;
   double             *temp_vect;
   hypre_ParVector    *Lx_par;
   hypre_Vector       *Lx_local;
   double             *Lx_data;

   /* --------------------------------------------------------*/
   /* get local and global size of vectors                    */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;

   /* --------------------------------------------------------*/
   /* fetch data pointer of input and output vectors          */
   /* --------------------------------------------------------*/

   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   y_par       = (hypre_ParVector *) y_csr;
   y_par_local = hypre_ParVectorLocalVector(y_par);
   y_par_data  = hypre_VectorData(y_par_local);

   /* --------------------------------------------------------*/
   /* copy from x to temporary vector                         */
   /* --------------------------------------------------------*/

   for (i = 0; i < local_nrows; i++) y_par_data[i] = x_par_data[i];

   /* --------------------------------------------------------*/
   /* create localb & localx of length = no. of interior nodes*/
   /* --------------------------------------------------------*/

   temp_list = hypre_TAlloc(int,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   temp_vect = hypre_TAlloc(double,  interior_nrows , NALU_HYPRE_MEMORY_HOST);
   for (i=0; i<interior_nrows; i++) temp_list[i] = i;
   for (i=0; i<local_nrows; i++)
   {
      if (remap_array[i] >= 0 && remap_array[i] < interior_nrows)
         temp_vect[remap_array[i]] = x_par_data[i];
   }
   NALU_HYPRE_IJVectorSetValues(localb,interior_nrows,(const int*) temp_list,
                           temp_vect);
   hypre_TFree(temp_list, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(temp_vect, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* perform one cycle of AMG to subdomain (internal nodes)  */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);

   NALU_HYPRE_BoomerAMGSolve( solver, LA_csr, Lb_csr, Lx_csr );

   /* --------------------------------------------------------*/
   /* update boundary nodes                                   */
   /* --------------------------------------------------------*/

   Lx_par   = (hypre_ParVector *) Lx_csr;
   Lx_local = hypre_ParVectorLocalVector(Lx_par);
   Lx_data  = hypre_VectorData(Lx_local);
   for (i=0; i<local_nrows; i++)
   {
      if ( remap_array[i] >= 0 )
      {
         for (j=0; j<offRowLengths[i]; j++)
         {
            index = offColInd[i][j];
            y_par_data[index] -= (Lx_data[remap_array[i]] * offColVal[i][j]);
         }
      }
   }
   return 0;
}

/***************************************************************************/
/* use CG to solve the interface problem                                   */
/***************************************************************************/

int NALU_HYPRE_IntfaceSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                        NALU_HYPRE_ParVector b_csr, NALU_HYPRE_ParVector x_csr )
{
   int                i, j, k, k1, local_nrows, global_nrows, index, num_procs;
   int                local_intface_nrows, global_intface_nrows;
   int                myBegin_int, myEnd_int, *itemp_vec, *itemp_vec2;
   int                icnt, icnt2, its, maxiter=500, mlen=100;
   double             init_norm, eps1, **ws, rnorm, t, one=1.0;
   double             **HH, *RS, *S, *C, ror, *darray, gam, epsmac=1.0e-10;
   double             rnorm2;

   NALU_HYPRE_IJVector     pvec, tvec, uvec, rvec, fvec, Tvec, T2vec;
   NALU_HYPRE_ParVector    p_csr, t_csr, u_csr, r_csr, f_csr, T_csr, T2_csr;
   hypre_ParVector    *x_par, *t_par, *p_par, *u_par, *r_par;

   hypre_ParVector    *b_par, *f_par;
   hypre_Vector       *f_par_local, *x_par_local, *b_par_local, *u_par_local;
   hypre_Vector       *t_par_local, *p_par_local, *r_par_local;
   double             *f_par_data, *x_par_data, *b_par_data, *u_par_data;
   double             *t_par_data, *p_par_data, *r_par_data;

   /* --------------------------------------------------------*/
   /* compose length of vector in the CG solve                */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,parComm);
   local_intface_nrows = myEnd - myBegin + 1 - interior_nrows;
   MPI_Allreduce(&local_intface_nrows, &global_intface_nrows, 1,MPI_INT,
                 MPI_SUM,parComm);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   itemp_vec  = hypre_TAlloc(int,  num_procs , NALU_HYPRE_MEMORY_HOST);
   itemp_vec2 = hypre_TAlloc(int,  num_procs , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs; i++) itemp_vec[i] = 0;
   itemp_vec[myRank] = local_intface_nrows;
   MPI_Allreduce(itemp_vec, itemp_vec2, num_procs, MPI_INT, MPI_SUM, parComm);
   myBegin_int = 0;
   for (i = 0; i < myRank; i++) myBegin_int += itemp_vec2[i];
   myEnd_int = myBegin_int + local_intface_nrows - 1;
   hypre_TFree(itemp_vec, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(itemp_vec2, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* copy input to output vectors                            */
   /* --------------------------------------------------------*/

   x_par       = (hypre_ParVector *) x_csr;
   x_par_local = hypre_ParVectorLocalVector(x_par);
   x_par_data  = hypre_VectorData(x_par_local);
   b_par       = (hypre_ParVector *) b_csr;
   b_par_local = hypre_ParVectorLocalVector(b_par);
   b_par_data  = hypre_VectorData(b_par_local);
   for (i = 0; i < local_nrows; i++) x_par_data[i] = b_par_data[i];
   if ( global_intface_nrows <= 0 ) return 0;

   /* --------------------------------------------------------*/
   /* create temporary vectors for GMRES                      */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &pvec);
   NALU_HYPRE_IJVectorSetObjectType(pvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(pvec);
   NALU_HYPRE_IJVectorAssemble(pvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &rvec);
   NALU_HYPRE_IJVectorSetObjectType(rvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(rvec);
   NALU_HYPRE_IJVectorAssemble(rvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &uvec);
   NALU_HYPRE_IJVectorSetObjectType(uvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(uvec);
   NALU_HYPRE_IJVectorAssemble(uvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &fvec);
   NALU_HYPRE_IJVectorSetObjectType(fvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(fvec);
   NALU_HYPRE_IJVectorAssemble(fvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &tvec);
   NALU_HYPRE_IJVectorSetObjectType(tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(tvec);
   NALU_HYPRE_IJVectorAssemble(tvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &Tvec);
   NALU_HYPRE_IJVectorSetObjectType(Tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(Tvec);
   NALU_HYPRE_IJVectorAssemble(Tvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &T2vec);
   NALU_HYPRE_IJVectorSetObjectType(T2vec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(T2vec);
   NALU_HYPRE_IJVectorAssemble(T2vec);
   /* ----------------------------------------------------------*/

   /* --------------------------------------------------------*/
   /* copy from x (long vector) to u (short vector)           */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorGetObject(fvec, (void **) &f_csr);
   f_par       = (hypre_ParVector *) f_csr;
   f_par_local = hypre_ParVectorLocalVector(f_par);
   f_par_data  = hypre_VectorData(f_par_local);

   index = 0;
   for (i = 0; i < local_nrows; i++)
   {
      if (remap_array[i] < 0) f_par_data[index++] = b_par_data[i];
   }

   /* --------------------------------------------------------*/
   /* get parcsr pointers for GMRES                           */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorGetObject(rvec, (void **) &r_csr);
   NALU_HYPRE_IJVectorGetObject(Tvec, (void **) &T_csr);
   NALU_HYPRE_IJVectorGetObject(T2vec, (void **) &T2_csr);
   NALU_HYPRE_IJVectorGetObject(tvec, (void **) &t_csr);
   NALU_HYPRE_IJVectorGetObject(pvec, (void **) &p_csr);
   NALU_HYPRE_IJVectorGetObject(uvec, (void **) &u_csr);

   p_par  = (hypre_ParVector *) p_csr;
   u_par  = (hypre_ParVector *) u_csr;
   t_par  = (hypre_ParVector *) t_csr;
   r_par  = (hypre_ParVector *) r_csr;
   t_par_local = hypre_ParVectorLocalVector(t_par);
   u_par_local = hypre_ParVectorLocalVector(u_par);
   p_par_local = hypre_ParVectorLocalVector(p_par);
   r_par_local = hypre_ParVectorLocalVector(r_par);
   t_par_data  = hypre_VectorData(t_par_local);
   u_par_data  = hypre_VectorData(u_par_local);
   p_par_data  = hypre_VectorData(p_par_local);
   r_par_data  = hypre_VectorData(r_par_local);

   /* --------------------------------------------------------*/
   /* allocate temporary memory for GMRES                     */
   /* --------------------------------------------------------*/

   darray = hypre_TAlloc(double, (mlen+1), NALU_HYPRE_MEMORY_HOST);
   HH = hypre_TAlloc(double*, (mlen+2), NALU_HYPRE_MEMORY_HOST);
   for (i=0; i<=mlen+1; i++)
      HH[i] = hypre_TAlloc(double, (mlen+2), NALU_HYPRE_MEMORY_HOST);
   RS = hypre_TAlloc(double, (mlen+2), NALU_HYPRE_MEMORY_HOST);
   S  = hypre_TAlloc(double, (mlen+2), NALU_HYPRE_MEMORY_HOST);
   C  = hypre_TAlloc(double, (mlen+2), NALU_HYPRE_MEMORY_HOST);
   ws = hypre_TAlloc(double*, (mlen+3), NALU_HYPRE_MEMORY_HOST);
   for (i=0; i<=mlen+2; i++)
      ws[i] = hypre_TAlloc(double, local_intface_nrows, NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* solve using GMRES                                       */
   /* --------------------------------------------------------*/

   NALU_HYPRE_ParVectorCopy( f_csr, r_csr );
   NALU_HYPRE_ParVectorInnerProd(r_csr, r_csr, &rnorm);
   init_norm = rnorm = rnorm2 = sqrt( rnorm );
   if ( myRank == 0 )
      printf("    Interface GMRES initial norm = %e\n", init_norm);

   its = 0;
   eps1 = 1.0E-8 * init_norm;
   while ( rnorm / init_norm > 1.0E-8 && its < maxiter )
   {
      ror = 1.0 / rnorm;
      for (i = 0; i < local_intface_nrows; i++) ws[0][i] = ror * r_par_data[i];
      RS[1] = rnorm2;
      icnt = 0;
      rnorm2 = rnorm;
      while (icnt < mlen && (rnorm2/init_norm) > 1.0E-8)
      {
         icnt++;
         its++;
         icnt2 = icnt + 1;
         for (i = 0; i < local_intface_nrows; i++)
            t_par_data[i] = ws[icnt-1][i];
         NALU_HYPRE_ApplyExtension( solver, t_csr, T_csr );
         NALU_HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, T_csr, 0.0, T2_csr );
         NALU_HYPRE_ApplyExtensionTranspose( solver, T2_csr, t_csr );
         for (i = 0; i < local_intface_nrows; i++) ws[icnt][i] = t_par_data[i];
         for (j = 1; j <= icnt; j++)
         {
            for (i=0; i<local_intface_nrows; i++) t_par_data[i] = ws[j-1][i];
            for (i=0; i<local_intface_nrows; i++) p_par_data[i] = ws[icnt2-1][i];
            NALU_HYPRE_ParVectorInnerProd(t_csr, p_csr, &darray[j-1]);
            t = darray[j-1];
            HH[j][icnt] = t;  t = - t;
            for (i=0; i<local_intface_nrows; i++)
               ws[icnt2-1][i] += (t*ws[j-1][i]);
         }
         for (i=0; i<local_intface_nrows; i++) t_par_data[i] = ws[icnt2-1][i];
         NALU_HYPRE_ParVectorInnerProd(t_csr, t_csr, &t);
         t = sqrt(t);
         HH[icnt2][icnt] = t;
         if (t != 0.0) {
            t = 1.0 / t;
            for (i=0; i<local_intface_nrows; i++) ws[icnt2-1][i] *= t;
         }
         if (icnt != 1) {
            for (k=2; k<=icnt; k++) {
               k1 = k - 1;
               t = HH[k1][icnt];
               HH[k1][icnt] =  C[k1] * t + S[k1] * HH[k][icnt];
               HH[k][icnt]  = -S[k1] * t + C[k1] * HH[k][icnt];
            }
         }
         gam=sqrt(HH[icnt][icnt]*HH[icnt][icnt]+
                  HH[icnt2][icnt]*HH[icnt2][icnt]);
         if (gam == 0.0) gam = epsmac;
         C[icnt] = HH[icnt][icnt] / gam;
         S[icnt] = HH[icnt2][icnt] / gam;
         RS[icnt2] = -S[icnt] * RS[icnt];
         RS[icnt]  = C[icnt] * RS[icnt];
         HH[icnt][icnt] = C[icnt] * HH[icnt][icnt] +
                          S[icnt] * HH[icnt2][icnt];
         rnorm2 = habs(RS[icnt2]);
         if ( myRank == 0 && its % 20 == 0 )
            printf("   Interface GMRES : iter %4d - res. norm = %e (%e)\n",its,
                       rnorm2, eps1);
      }
      rnorm = rnorm2;
      RS[icnt] = RS[icnt] / HH[icnt][icnt];
      for (i=2; i<=icnt; i++) {
         k = icnt - i + 1;
         k1 = k + 1;
         t = RS[k];
         for (j=k1; j<=icnt; j++) t = t - HH[k][j] * RS[j];
         RS[k] = t / HH[k][k];
      }
      t = RS[1];
      for (i=0; i<local_intface_nrows; i++) ws[0][i] *= t;
      for (j=2; j<=icnt; j++)
      {
         t = RS[j];
         for (i=0; i<local_intface_nrows; i++) ws[0][i] += (t * ws[j-1][i]);
      }
      for (i=0; i<local_intface_nrows; i++) u_par_data[i] += ws[0][i];

      NALU_HYPRE_ApplyExtension( solver, u_csr, T_csr );
      NALU_HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, T_csr, 0.0, T2_csr );
      NALU_HYPRE_ApplyExtensionTranspose( solver, T2_csr, r_csr );
      hypre_ParVectorScale(-one, r_par);
      hypre_ParVectorAxpy(one, f_par, r_par);
      NALU_HYPRE_ParVectorInnerProd(r_csr, r_csr, &rnorm);
      rnorm = sqrt( rnorm );
      /*if ( myRank == 0 )
         printf("   Interface GMRES : true res. norm = %e \n", rnorm);
      */
   }

   /* --------------------------------------------------------*/
   /* copy from u (short vector) to x (long vector)           */
   /* --------------------------------------------------------*/

   index = 0;
   for (i = 0; i < local_nrows; i++)
   {
      if (remap_array[i] < 0) x_par_data[i] = u_par_data[index++];
   }

   /* --------------------------------------------------------*/
   /* clean up                                                */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorDestroy(rvec);
   NALU_HYPRE_IJVectorDestroy(tvec);
   NALU_HYPRE_IJVectorDestroy(Tvec);
   NALU_HYPRE_IJVectorDestroy(T2vec);
   NALU_HYPRE_IJVectorDestroy(uvec);
   NALU_HYPRE_IJVectorDestroy(fvec);
   NALU_HYPRE_IJVectorDestroy(pvec);
   for (i=0; i<=mlen+2; i++)
      hypre_TFree(ws[i], NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(ws, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(darray, NALU_HYPRE_MEMORY_HOST);
   for (i=1; i<=mlen+1; i++)
      hypre_TFree(HH[i], NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(HH, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(RS, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(S, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(C, NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/***************************************************************************/
/* Compute y = E^T A E x where A is the global matrix and x and y are      */
/* global vectors                                                          */
/***************************************************************************/

int NALU_HYPRE_DDAMGSolve(NALU_HYPRE_Solver solver, NALU_HYPRE_ParCSRMatrix A_csr,
                     NALU_HYPRE_ParVector x_csr, NALU_HYPRE_ParVector y_csr )
{
   int             local_nrows, global_nrows;
   NALU_HYPRE_IJVector  tvec;
   NALU_HYPRE_ParVector t_csr;

   /* --------------------------------------------------------*/
   /* initialize and fetch double arrays for b and x (global) */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,parComm);
   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &tvec);
   NALU_HYPRE_IJVectorSetObjectType(tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(tvec);
   NALU_HYPRE_IJVectorAssemble(tvec);
   NALU_HYPRE_IJVectorGetObject(tvec, (void **) &t_csr);

   /* --------------------------------------------------------*/
   /* apply E^T                                               */
   /* --------------------------------------------------------*/

   NALU_HYPRE_ApplyTransformTranspose( solver, x_csr, y_csr );

   /* --------------------------------------------------------*/
   /* solve for E^T A E using CG                              */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IntfaceSolve(solver, A_csr, y_csr, t_csr);
   NALU_HYPRE_LocalAMGSolve(solver, t_csr, t_csr );

   /* --------------------------------------------------------*/
   /* apply E                                                 */
   /* --------------------------------------------------------*/

   NALU_HYPRE_ApplyTransform( solver, t_csr, y_csr );

   /* --------------------------------------------------------*/
   /* clean up                                                */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorDestroy( tvec );

   return 0;
}

/***************************************************************************/
/* solve the linear system using domain decomposed AMG                     */
/***************************************************************************/

int NALU_HYPRE_LSI_DDAMGSolve(NALU_HYPRE_ParCSRMatrix A_csr, NALU_HYPRE_ParVector x_csr,
                  NALU_HYPRE_ParVector b_csr)
{
   int             i, j, k, *row_partition, local_nrows, num_procs, rowSize;
   int             *colInd, *newColInd, rowCnt, eqnNum, *rowLengths;
   int             nnz=0, relaxType[4], maxRowSize, global_nrows;
   int             myBegin_int, myEnd_int, *itemp_vec, *itemp_vec2;
   int             local_intface_nrows, global_intface_nrows;
   int             num_iterations;
   double          *colVal, *newColVal;
   NALU_HYPRE_ParCSRMatrix  LA_csr;
   NALU_HYPRE_IJVector  tvec, Tvec, T2vec;
   NALU_HYPRE_ParVector t_csr, T_csr, T2_csr, Lx_csr, Lb_csr;
   MPI_Comm        newComm, dummyComm;
   NALU_HYPRE_Solver    PSolver, SeqPrecon;

   /* --------------------------------------------------------*/
   /* construct local range                                   */
   /* --------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(A_csr, &row_partition);
   myBegin = row_partition[myRank];
   myEnd   = row_partition[myRank+1] - 1;
   hypre_TFree( row_partition , NALU_HYPRE_MEMORY_HOST);

   /* --------------------------------------------------------*/
   /* create and load a local matrix                          */
   /* --------------------------------------------------------*/

   local_nrows = myEnd - myBegin + 1;
   for ( i = 0; i < num_procs; i++ )
   {
      if ( myRank == i )
         MPI_Comm_split(MPI_COMM_WORLD, i+1, 0, &newComm);
      else
         MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 1, &dummyComm);
   }
   MPI_Comm_rank(newComm, &i);
   MPI_Comm_size(newComm, &j);
   parComm = MPI_COMM_WORLD;

   /* --------------------------------------------------------*/
   /* find out how many rows are interior rows (remap[i] >= 0)*/
   /* --------------------------------------------------------*/

   remap_array = hypre_TAlloc(int, local_nrows , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < local_nrows; i++ ) remap_array[i] = 0;
   for ( i = myBegin; i <= myEnd; i++ )
   {
      NALU_HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      for ( j = 0; j < rowSize; j++ )
         if ( colInd[j] < myBegin || colInd[j] > myEnd )
            {remap_array[i-myBegin] = -1; break;}
      NALU_HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }
   interior_nrows = 0;
   for ( i = 0; i < local_nrows; i++ )
      if ( remap_array[i] == 0 ) remap_array[i] = interior_nrows++;

   /* --------------------------------------------------------*/
   /* construct the local matrix (only the border nodes)      */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixCreate(newComm, 0, 0+interior_nrows-1,
			0, 0+interior_nrows-1, &localA);
   NALU_HYPRE_IJMatrixSetObjectType(localA, NALU_HYPRE_PARCSR);

   rowLengths = hypre_TAlloc(int, interior_nrows , NALU_HYPRE_MEMORY_HOST);
   offRowLengths = hypre_TAlloc(int, local_nrows , NALU_HYPRE_MEMORY_HOST);
   rowCnt = 0;
   maxRowSize = 0;
   for ( i = myBegin; i <= myEnd; i++ )
   {
      offRowLengths[i-myBegin] = 0;
      if ( remap_array[i-myBegin] >= 0 )
      {
         rowLengths[rowCnt] = 0;
         NALU_HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
         for ( j = 0; j < rowSize; j++ )
         {
            if ( colInd[j] >= myBegin && colInd[j] <= myEnd )
            {
               if (remap_array[colInd[j]-myBegin] >= 0) rowLengths[rowCnt]++;
               else offRowLengths[i-myBegin]++;
            }
         }
         nnz += rowLengths[rowCnt];
         maxRowSize = (rowLengths[rowCnt] > maxRowSize) ?
                       rowLengths[rowCnt] : maxRowSize;
         NALU_HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
         rowCnt++;
      }
   }
   NALU_HYPRE_IJMatrixSetRowSizes(localA, rowLengths);
   NALU_HYPRE_IJMatrixInitialize(localA);
   newColInd = hypre_TAlloc(int, maxRowSize , NALU_HYPRE_MEMORY_HOST);
   newColVal = hypre_TAlloc(double, maxRowSize , NALU_HYPRE_MEMORY_HOST);
   rowCnt = 0;
   offColInd = hypre_TAlloc(int*, local_nrows , NALU_HYPRE_MEMORY_HOST);
   offColVal = hypre_TAlloc(double*, local_nrows , NALU_HYPRE_MEMORY_HOST);
   for ( i = 0; i < local_nrows; i++ )
   {
      if ( offRowLengths[i] > 0 )
      {
         offColInd[i] = hypre_TAlloc(int, offRowLengths[i] , NALU_HYPRE_MEMORY_HOST);
         offColVal[i] = hypre_TAlloc(double, offRowLengths[i] , NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         offColInd[i] = NULL;
         offColVal[i] = NULL;
      }
   }
   for ( i = 0; i < local_nrows; i++ )
   {
      eqnNum = myBegin + i;
      if  ( remap_array[i] >= 0 )
      {
         NALU_HYPRE_ParCSRMatrixGetRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
         nnz = 0;
         k = 0;
         for ( j = 0; j < rowSize; j++ )
         {
            if ( colInd[j] >= myBegin && colInd[j] <= myEnd )
            {
               if ( remap_array[colInd[j]-myBegin] >= 0 )
               {
                  newColInd[nnz] = remap_array[colInd[j]-myBegin];
                  newColVal[nnz++] = colVal[j];
               }
               else
               {
                  offColInd[i][k] = colInd[j]-myBegin;
                  offColVal[i][k++] = colVal[j];
               }
            }
         }
         if ( k != offRowLengths[i] )
            printf("WARNING : k != offRowLengths[i]\n");
         NALU_HYPRE_ParCSRMatrixRestoreRow(A_csr,eqnNum,&rowSize,&colInd,&colVal);
         NALU_HYPRE_IJMatrixSetValues(localA,1,&nnz,&rowCnt,newColInd,newColVal);
         rowCnt++;
      }
   }
   hypre_TFree(newColInd , NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(newColVal , NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_IJMatrixAssemble(localA);

   /* --------------------------------------------------------*/
   /* create and load local vectors                           */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJVectorCreate(newComm, 0, interior_nrows-1, &localx);
   NALU_HYPRE_IJVectorSetObjectType(localx, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(localx);
   NALU_HYPRE_IJVectorAssemble(localx);
   NALU_HYPRE_IJVectorCreate(newComm, 0, interior_nrows-1, &localb);
   NALU_HYPRE_IJVectorSetObjectType(localb, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(localb);
   NALU_HYPRE_IJVectorAssemble(localb);

   /* --------------------------------------------------------*/
   /* create an AMG context                                   */
   /* --------------------------------------------------------*/

   NALU_HYPRE_BoomerAMGCreate(&SeqPrecon);
   NALU_HYPRE_BoomerAMGSetMaxIter(SeqPrecon, 1);
   NALU_HYPRE_BoomerAMGSetCycleType(SeqPrecon, 1);
   NALU_HYPRE_BoomerAMGSetMaxLevels(SeqPrecon, 25);
   relaxType[0] = relaxType[1] = relaxType[2] = 5;
   relaxType[3] = 9;
   NALU_HYPRE_BoomerAMGSetGridRelaxType(SeqPrecon, relaxType);
   NALU_HYPRE_BoomerAMGSetTol(SeqPrecon, 1.0E-16);
   NALU_HYPRE_BoomerAMGSetMeasureType(SeqPrecon, 0);

   NALU_HYPRE_IJMatrixGetObject(localA, (void**) &LA_csr);
   NALU_HYPRE_IJVectorGetObject(localx, (void**) &Lx_csr);
   NALU_HYPRE_IJVectorGetObject(localb, (void**) &Lb_csr);

   /*NALU_HYPRE_BoomerAMGSetPrintLevel(SeqPrecon, 2);*/
   /*NALU_HYPRE_BoomerAMGSetDebugFlag(SeqPrecon, 1);*/
   NALU_HYPRE_BoomerAMGSetup( SeqPrecon, LA_csr, Lb_csr, Lx_csr);
   MPI_Barrier(MPI_COMM_WORLD);

   /* --------------------------------------------------------*/
   /* diagnostics                                             */
   /* --------------------------------------------------------*/

/* small code to check symmetry
NALU_HYPRE_ParVectorSetRandomValues( x_csr, 10345 );
NALU_HYPRE_ParVectorSetRandomValues( b_csr, 24893 );
NALU_HYPRE_DDAMGSolve( SeqPrecon, A_csr, x_csr, r_csr);
NALU_HYPRE_ParVectorInnerProd( b_csr, r_csr, &ddata);
printf("CHECK 1 = %e\n", ddata);
NALU_HYPRE_DDAMGSolve( SeqPrecon, A_csr, b_csr, r_csr);
NALU_HYPRE_ParVectorInnerProd( x_csr, r_csr, &ddata);
printf("CHECK 2 = %e\n", ddata);
*/

   MPI_Allreduce(&local_nrows, &global_nrows,1,MPI_INT,MPI_SUM,parComm);
   local_intface_nrows = myEnd - myBegin + 1 - interior_nrows;
   MPI_Allreduce(&local_intface_nrows, &global_intface_nrows, 1,MPI_INT,
                 MPI_SUM,parComm);
   itemp_vec  = hypre_TAlloc(int,  num_procs , NALU_HYPRE_MEMORY_HOST);
   itemp_vec2 = hypre_TAlloc(int,  num_procs , NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs; i++) itemp_vec[i] = 0;
   itemp_vec[myRank] = local_intface_nrows;
   MPI_Allreduce(itemp_vec, itemp_vec2, num_procs, MPI_INT, MPI_SUM, parComm);
   myBegin_int = 0;
   for (i = 0; i < myRank; i++) myBegin_int += itemp_vec2[i];
   myEnd_int = myBegin_int + local_intface_nrows - 1;
   hypre_TFree(itemp_vec, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(itemp_vec2, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin_int, myEnd_int, &tvec);
   NALU_HYPRE_IJVectorSetObjectType(tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(tvec);
   NALU_HYPRE_IJVectorAssemble(tvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &Tvec);
   NALU_HYPRE_IJVectorSetObjectType(Tvec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(Tvec);
   NALU_HYPRE_IJVectorAssemble(Tvec);

   NALU_HYPRE_IJVectorCreate(parComm, myBegin, myEnd, &T2vec);
   NALU_HYPRE_IJVectorSetObjectType(T2vec, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJVectorInitialize(T2vec);
   NALU_HYPRE_IJVectorAssemble(T2vec);

   NALU_HYPRE_IJVectorGetObject(Tvec, (void **) &T_csr);
   NALU_HYPRE_IJVectorGetObject(T2vec, (void **) &T2_csr);
   NALU_HYPRE_IJVectorGetObject(tvec, (void **) &t_csr);

/*
   for ( i = 0; i < global_intface_nrows; i++ )
   {
      MPI_Barrier(MPI_COMM_WORLD);
      NALU_HYPRE_IJVectorZeroLocalComponents(tvec);
      if ( i >= myBegin_int && i <= myEnd_int )
         t_par_data[i-myBegin_int] = 1.0;
      NALU_HYPRE_ApplyExtension( SeqPrecon, t_csr, T_csr );
      NALU_HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, T_csr, 0.0, T2_csr );
      NALU_HYPRE_ApplyExtensionTranspose( SeqPrecon, T2_csr, t_csr );
      for ( k1 = 0; k1 < local_intface_nrows; k1++ )
         if ( t_par_data[k1] != 0.0 )
            printf("RA(%4d,%4d) = %e;\n",i+1,myBegin_int+k1+1,t_par_data[k1]);
   }
*/
   MPI_Barrier(MPI_COMM_WORLD);

   /* --------------------------------------------------------*/
   /* solve using GMRES                                       */
   /* --------------------------------------------------------*/

   NALU_HYPRE_ParCSRGMRESCreate(parComm, &PSolver);
   NALU_HYPRE_ParCSRGMRESSetPrecond(PSolver,NALU_HYPRE_DDAMGSolve,NALU_HYPRE_DummySetup,
                               SeqPrecon);
   NALU_HYPRE_ParCSRGMRESSetKDim(PSolver, 100);
   NALU_HYPRE_ParCSRGMRESSetMaxIter(PSolver, 100);
   NALU_HYPRE_ParCSRGMRESSetTol(PSolver, 1.0E-8);
   NALU_HYPRE_ParCSRGMRESSetup(PSolver, A_csr, b_csr, x_csr);
   NALU_HYPRE_ParCSRGMRESSolve(PSolver, A_csr, b_csr, x_csr);
   NALU_HYPRE_ParCSRGMRESGetNumIterations(PSolver, &num_iterations);
   /*NALU_HYPRE_ParCSRPCGCreate(parComm, &PSolver);
     NALU_HYPRE_ParCSRPCGSetPrecond(PSolver,NALU_HYPRE_DDAMGSolve,NALU_HYPRE_DummySetup,
                              SeqPrecon);
     NALU_HYPRE_ParCSRPCGSetMaxIter(PSolver, 100);
     NALU_HYPRE_ParCSRPCGSetTol(PSolver, 1.0E-8);
     NALU_HYPRE_ParCSRPCGSetup(PSolver, A_csr, b_csr, x_csr);
     NALU_HYPRE_ParCSRPCGSolve(PSolver, A_csr, b_csr, x_csr);
     NALU_HYPRE_ParCSRPCGGetNumIterations(PSolver, &num_iterations);
   */
   if ( myRank == 0 )
      printf("GMRES iteration count = %d \n", num_iterations);

   /* --------------------------------------------------------*/
   /* clean up                                                */
   /* --------------------------------------------------------*/

   NALU_HYPRE_IJMatrixDestroy(localA);
   NALU_HYPRE_IJVectorDestroy(localx);
   NALU_HYPRE_IJVectorDestroy(localb);
   NALU_HYPRE_BoomerAMGDestroy(SeqPrecon);
   NALU_HYPRE_ParCSRGMRESDestroy( PSolver );
   return 0;
}

