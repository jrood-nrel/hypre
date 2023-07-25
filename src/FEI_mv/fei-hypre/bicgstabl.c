/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * BiCGSTABL 
 *
 *****************************************************************************/

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"

#include "_nalu_hypre_FEI.h"
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      size;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *s;
   void  *y;
   void  *t;
   void  *tt;
   void  *st;
   void  *asm1;
   void  *as;
   void  *awt;
   void  *wt;
   void  *wh;
   void  *at;
   void  *xt;
   void  *t2;

   void  *matvec_data;

   int    (*precond)(void*, void*, void*, void*);
   int    (*precond_setup)(void*, void*, void*, void*);
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} nalu_hypre_BiCGSTABLData;

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLCreate
 *--------------------------------------------------------------------------*/
 
void * nalu_hypre_BiCGSTABLCreate( )
{
   nalu_hypre_BiCGSTABLData *bicgstab_data;
 
   bicgstab_data = nalu_hypre_CTAlloc(nalu_hypre_BiCGSTABLData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> size)           = 2;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (bicgstab_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> y)              = NULL;
   (bicgstab_data -> t)              = NULL;
   (bicgstab_data -> tt)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> asm1)           = NULL;
   (bicgstab_data -> as)             = NULL;
   (bicgstab_data -> awt)            = NULL;
   (bicgstab_data -> wt)             = NULL;
   (bicgstab_data -> wh)             = NULL;
   (bicgstab_data -> at)             = NULL;
   (bicgstab_data -> xt)             = NULL;
   (bicgstab_data -> t2)             = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLDestroy( void *bicgstab_vdata )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int ierr = 0;
 
   if (bicgstab_data)
   {
      if ((bicgstab_data -> logging) > 0)
      {
         nalu_hypre_TFree(bicgstab_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
 
      nalu_hypre_ParKrylovMatvecDestroy(bicgstab_data -> matvec_data);
 
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> r);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> s);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> y);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> t);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> tt);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> st);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> as);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> asm1);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> awt);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> wt);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> wh);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> at);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> xt);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> t2);
 
      nalu_hypre_TFree(bicgstab_data, NALU_HYPRE_MEMORY_HOST);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data     = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)(void*, void*, void*, void*) = (bicgstab_data -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> s) == NULL)
      (bicgstab_data -> s) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> y) == NULL)
      (bicgstab_data -> y) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> t) == NULL)
      (bicgstab_data -> t) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> tt) == NULL)
      (bicgstab_data -> tt) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> st) == NULL)
      (bicgstab_data -> st) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> asm1) == NULL)
      (bicgstab_data -> asm1) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> as) == NULL)
      (bicgstab_data -> as) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> awt) == NULL)
      (bicgstab_data -> awt) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> wt) == NULL)
      (bicgstab_data -> wt) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> wh) == NULL)
      (bicgstab_data -> wh) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> at) == NULL)
      (bicgstab_data -> at) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> xt) == NULL)
      (bicgstab_data -> xt) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> t2) == NULL)
      (bicgstab_data -> t2) = nalu_hypre_ParKrylovCreateVector(b);
 
   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data -> logging) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = nalu_hypre_CTAlloc(double,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
      if ((bicgstab_data -> log_file_name) == NULL)
		  (bicgstab_data -> log_file_name) = (char*) "bicgstab.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x)
{
	nalu_hypre_BiCGSTABLData  *bicgstab_data   = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   void              *matvec_data  = (bicgstab_data -> matvec_data);

   void             *r            = (bicgstab_data -> r);
   void             *s            = (bicgstab_data -> s);
   void             *y            = (bicgstab_data -> y);
   void             *t            = (bicgstab_data -> t);
   void             *tt           = (bicgstab_data -> tt);
   void             *wt           = (bicgstab_data -> wt);
   void             *awt          = (bicgstab_data -> awt);
   void             *asm1         = (bicgstab_data -> asm1);
   void             *as           = (bicgstab_data -> as);
   void             *wh           = (bicgstab_data -> wh);
   void             *xt           = (bicgstab_data -> xt);
   void             *at           = (bicgstab_data -> at);
   void             *st           = (bicgstab_data -> st);
   void             *t2           = (bicgstab_data -> t2);
   int 	           (*precond)(void*, void*, void*, void*)   = (bicgstab_data -> precond);
   int 	            *precond_data = (int*) (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   double         *norms          = (bicgstab_data -> norms);
   
   int        ierr = 0;
   int        iter, flag; 
   int        my_id, num_procs;
   double     eta, chi, xi, psi, dtmp, dtmp2, r_norm, b_norm;
   double     A11, A12, A21, A22, B1, B2, omega; 
   double     epsilon, phi, delta, deltam1, omegam1;

   nalu_hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgstab_data -> norms);
   }

   /* initialize work arrays */
nalu_hypre_ParKrylovClearVector(x);
   nalu_hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(nalu_hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("BiCGSTABL : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGSTABL : Initial L2 norm of residual = %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
      epsilon = accuracy * b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
      epsilon = accuracy * r_norm;
   };

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   nalu_hypre_ParKrylovCopyVector(r,s);
   nalu_hypre_ParKrylovCopyVector(r,y);
   delta = nalu_hypre_ParKrylovInnerProd(r,y);
   precond(precond_data, A, s, t);
   nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,as);
   phi = nalu_hypre_ParKrylovInnerProd(y,as) / delta;
   omega = 0.0;
   
   while ( iter < max_iter && r_norm > epsilon )
   {
      iter++;

      omegam1 = omega;
      omega   = 1.0 / phi;

      if ( iter >= 2 ) 
      {
         nalu_hypre_ParKrylovCopyVector(awt,at);
         dtmp = - psi;
         nalu_hypre_ParKrylovAxpy(dtmp,asm1,at);
         nalu_hypre_ParKrylovCopyVector(wt,wh);
         dtmp = - omega;
         nalu_hypre_ParKrylovAxpy(dtmp,at,wh);
      }

      nalu_hypre_ParKrylovCopyVector(r,wt);
      dtmp = - omega;
      nalu_hypre_ParKrylovAxpy(dtmp,as,wt);

      if ( iter % 2 == 1 )
      {
         precond(precond_data, A, wt, t);
         nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,awt);
         dtmp = nalu_hypre_ParKrylovInnerProd(wt,awt);
         dtmp2 = nalu_hypre_ParKrylovInnerProd(awt,awt);
         chi = dtmp / dtmp2;
         nalu_hypre_ParKrylovCopyVector(wt,r);
         dtmp = - chi;
         nalu_hypre_ParKrylovAxpy(dtmp,awt,r);
         nalu_hypre_ParKrylovCopyVector(x,xt);
         nalu_hypre_ParKrylovAxpy(omega,s,x);
         nalu_hypre_ParKrylovAxpy(chi,wt,x);
         deltam1 = delta;
         delta = nalu_hypre_ParKrylovInnerProd(r,y);
         psi = - omega * delta / ( deltam1 * chi);
         nalu_hypre_ParKrylovCopyVector(s,st);
         nalu_hypre_ParKrylovCopyVector(s,t);
         dtmp = - chi;
         nalu_hypre_ParKrylovAxpy(dtmp,as,t);
         nalu_hypre_ParKrylovCopyVector(r,s);
         dtmp = - psi;
         nalu_hypre_ParKrylovAxpy(dtmp,t,s);
      }
      else
      {
         dtmp = - 1.0;
         nalu_hypre_ParKrylovCopyVector(wt,t2);
         nalu_hypre_ParKrylovAxpy(dtmp,wh,t2);
         precond(precond_data, A, wt, t);
         nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,awt);
         A11 = nalu_hypre_ParKrylovInnerProd(t2,t2);
         A12 = nalu_hypre_ParKrylovInnerProd(t2,awt);
         A21 = A12;
         A22 = nalu_hypre_ParKrylovInnerProd(awt,awt);
         B1  = nalu_hypre_ParKrylovInnerProd(t2,wh);
         B2  = nalu_hypre_ParKrylovInnerProd(awt,wh);
         flag = 0;
         if ( A21 > A11 )
         {
            dtmp = A11; A11 = A21; A21 = dtmp;
            dtmp = A12; A12 = A22; A22 = dtmp;
            flag = 1;
         }
         A21 = A12 / A11;
         A22 = A22 - A12 * A12 / A11;
         xi = B1; 
         eta = B2 - A21 * xi;
         eta = eta / A22;
         xi = (xi - A12 * eta) / A11;
         xi = - xi; 
         eta = -eta;
         if ( flag == 1 ) { dtmp = eta; eta = xi; xi = dtmp;}
         dtmp = 1.0 - xi;
         nalu_hypre_ParKrylovCopyVector(wh,r);
         nalu_hypre_ParKrylovScaleVector(dtmp,r);
         nalu_hypre_ParKrylovAxpy(xi,wt,r);
         nalu_hypre_ParKrylovAxpy(eta,awt,r);
         nalu_hypre_ParKrylovCopyVector(x,t);
         nalu_hypre_ParKrylovAxpy(omega,s,t);
         nalu_hypre_ParKrylovCopyVector(xt,x);
         nalu_hypre_ParKrylovAxpy(omegam1,st,x);
         nalu_hypre_ParKrylovAxpy(omega,tt,x);
         dtmp = 1.0 - xi;
         nalu_hypre_ParKrylovScaleVector(dtmp,x);
         nalu_hypre_ParKrylovAxpy(xi,t,x);
         dtmp = - eta;
         nalu_hypre_ParKrylovAxpy(dtmp,wt,x);
         deltam1 = delta;
         delta  = nalu_hypre_ParKrylovInnerProd(r,y);
         psi = omega * delta / ( deltam1 * eta);
         nalu_hypre_ParKrylovCopyVector(s,st);
         dtmp = 1.0 - xi;
         nalu_hypre_ParKrylovCopyVector(tt,t);
         nalu_hypre_ParKrylovAxpy(xi,s,t);
         nalu_hypre_ParKrylovAxpy(eta,as,t);
         nalu_hypre_ParKrylovCopyVector(r,s);
         dtmp = - psi;
         nalu_hypre_ParKrylovAxpy(dtmp,t,s);
      }

      nalu_hypre_ParKrylovCopyVector(wt,tt);
      dtmp = - psi;
      nalu_hypre_ParKrylovAxpy(dtmp,st,tt);
      nalu_hypre_ParKrylovCopyVector(as,asm1);
      precond(precond_data, A, s, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,as);
      phi = nalu_hypre_ParKrylovInnerProd(as,y) / delta;
        
      precond(precond_data, A, x, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, t, 1.0, r);
      r_norm = nalu_hypre_ParKrylovInnerProd(r,r);
      if ( my_id == 0 && logging )
         printf(" BiCGSTAB2 : iter %4d - res. norm = %e \n", iter, r_norm);
   }
   precond(precond_data, A, x, t);
   nalu_hypre_ParKrylovCopyVector(t,x);

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetTol( void *bicgstab_vdata, double tol )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetMinIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> size) = size;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetStopCrit
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(void*, void*, void*, void*),
                       int  (*precond_setup)(void*, void*, void*, void*), void  *precond_data )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> precond)        = precond;
   (bicgstab_data -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging)
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations)
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 

/******************************************************************************
 ******************************************************************************
 ******************************************************************************
  haven't been verified to work yet
 *****************************************************************************/

#ifdef OLDSTUFF

/******************************************************************************
 *
 * BiCGSTABL 
 *
 *****************************************************************************/

#include "utilities/_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "IJ_mv/NALU_HYPRE_IJ_mv.h"
#include "parcsr_mv/NALU_HYPRE_parcsr_mv.h"
#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_ls/NALU_HYPRE_parcsr_ls.h"
#include "headers.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLData
 *--------------------------------------------------------------------------*/

typedef struct
{
   int      size;
   int      max_iter;
   int      stop_crit;
   double   tol;
   double   rel_residual_norm;

   void  *A;
   void  *r;
   void  *rh;
   void  *rt;
   void  *rt1;
   void  *rt2;
   void  *rt3;
   void  *ut;
   void  *ut1;
   void  *ut2;
   void  *ut3;
   void  *t;
   void  *xh;

   void  *matvec_data;

	int    (*precond)(void*,void*,void*,void*);
	int    (*precond_setup)(void*,void*,void*,void*);
   void    *precond_data;

   /* log info (always logged) */
   int      num_iterations;
 
   /* additional log info (logged when `logging' > 0) */
   int      logging;
   double  *norms;
   char    *log_file_name;

} nalu_hypre_BiCGSTABLData;

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLCreate
 *--------------------------------------------------------------------------*/
 
void * nalu_hypre_BiCGSTABLCreate( )
{
   nalu_hypre_BiCGSTABLData *bicgstab_data;
 
   bicgstab_data = nalu_hypre_CTAlloc(nalu_hypre_BiCGSTABLData,  1, NALU_HYPRE_MEMORY_HOST);
 
   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> size)           = 2;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> precond)        = nalu_hypre_ParKrylovIdentity;
   (bicgstab_data -> precond_setup)  = nalu_hypre_ParKrylovIdentitySetup;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> rh)             = NULL;
   (bicgstab_data -> rt)             = NULL;
   (bicgstab_data -> rt1)            = NULL;
   (bicgstab_data -> rt2)            = NULL;
   (bicgstab_data -> rt3)            = NULL;
   (bicgstab_data -> ut)             = NULL;
   (bicgstab_data -> ut1)            = NULL;
   (bicgstab_data -> ut2)            = NULL;
   (bicgstab_data -> ut3)            = NULL;
   (bicgstab_data -> xh)             = NULL;
   (bicgstab_data -> t)              = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;
 
   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLDestroy
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLDestroy( void *bicgstab_vdata )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int i, ierr = 0;
 
   if (bicgstab_data)
   {
      if ((bicgstab_data -> logging) > 0)
      {
         nalu_hypre_TFree(bicgstab_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
 
      nalu_hypre_ParKrylovMatvecDestroy(bicgstab_data -> matvec_data);
 
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> r);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> rh);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> rt);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> rt1);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> rt2);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> rt3);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> ut);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> ut1);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> ut2);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> ut3);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> xh);
      nalu_hypre_ParKrylovDestroyVector(bicgstab_data -> t);
 
      nalu_hypre_TFree(bicgstab_data, NALU_HYPRE_MEMORY_HOST);
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetup
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetup( void *bicgstab_vdata, void *A, void *b, void *x         )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data     = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            max_iter         = (bicgstab_data -> max_iter);
   int          (*precond_setup)(void*,void*,void*,void*) = (bicgstab_data -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);
   int            ierr = 0;
 
   (bicgstab_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((bicgstab_data -> r) == NULL)
      (bicgstab_data -> r) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> rh) == NULL)
      (bicgstab_data -> rh) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> rt) == NULL)
      (bicgstab_data -> rt) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> rt1) == NULL)
      (bicgstab_data -> rt1) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> rt2) == NULL)
      (bicgstab_data -> rt2) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> rt3) == NULL)
      (bicgstab_data -> rt3) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> ut) == NULL)
      (bicgstab_data -> ut) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> ut1) == NULL)
      (bicgstab_data -> ut1) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> ut2) == NULL)
      (bicgstab_data -> ut2) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> ut3) == NULL)
      (bicgstab_data -> ut3) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> xh) == NULL)
      (bicgstab_data -> xh) = nalu_hypre_ParKrylovCreateVector(b);
   if ((bicgstab_data -> t) == NULL)
      (bicgstab_data -> t) = nalu_hypre_ParKrylovCreateVector(b);
 
   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) = nalu_hypre_ParKrylovMatvecCreate(A, x);
 
   precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ((bicgstab_data -> logging) > 0)
   {
      if ((bicgstab_data -> norms) == NULL)
         (bicgstab_data -> norms) = nalu_hypre_CTAlloc(double,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
      if ((bicgstab_data -> log_file_name) == NULL)
		  (bicgstab_data -> log_file_name) = (char*)"bicgstab.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSolve
 *-------------------------------------------------------------------------*/

int nalu_hypre_BiCGSTABLSolve(void  *bicgstab_vdata, void  *A, void  *b, void  *x)
{
	nalu_hypre_BiCGSTABLData  *bicgstab_data   =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int               size         = (bicgstab_data -> size);
   int 		     max_iter     = (bicgstab_data -> max_iter);
   int 		     stop_crit    = (bicgstab_data -> stop_crit);
   double 	     accuracy     = (bicgstab_data -> tol);
   void              *matvec_data  = (bicgstab_data -> matvec_data);
   double            mat[2][2], gammanp[2], gammap[2], sigma[2], tau[2][2];

   void             *r            = (bicgstab_data -> r);
   void             *rh           = (bicgstab_data -> rh);
   void             *rt           = (bicgstab_data -> rt);
   void             *rt1          = (bicgstab_data -> rt1);
   void             *rt2          = (bicgstab_data -> rt2);
   void             *rt3          = (bicgstab_data -> rt3);
   void             *ut           = (bicgstab_data -> ut);
   void             *ut1          = (bicgstab_data -> ut1);
   void             *ut2          = (bicgstab_data -> ut2);
   void             *ut3          = (bicgstab_data -> ut3);
   void             *xh           = (bicgstab_data -> xh);
   void             *t            = (bicgstab_data -> t);

   int 	           (*precond)(void*,void*,void*,void*)   = (bicgstab_data -> precond);
   int 	            *precond_data = (int*) (bicgstab_data -> precond_data);

   /* logging variables */
   int             logging        = (bicgstab_data -> logging);
   double         *norms          = (bicgstab_data -> norms);
   char           *log_file_name  = (bicgstab_data -> log_file_name);
   
   int        ierr = 0;
   int        iter; 
   int        j; 
   int        my_id, num_procs;
   double     alpha, beta, gamma, epsilon, rho, rho1, dtmp, r_norm, b_norm;
   double     gammapp[2], darray[2], epsmac = 1.e-16, omega; 

   nalu_hypre_ParKrylovCommInfo(A,&my_id,&num_procs);
   if (logging > 0)
   {
      norms          = (bicgstab_data -> norms);
      log_file_name  = (bicgstab_data -> log_file_name);
   }

   /* initialize work arrays */
nalu_hypre_ParKrylovClearVector(x);
   nalu_hypre_ParKrylovCopyVector(b,r);

   /* compute initial residual */

   nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, x, 1.0, r);
   r_norm = sqrt(nalu_hypre_ParKrylovInnerProd(r,r));
   b_norm = sqrt(nalu_hypre_ParKrylovInnerProd(b,b));
   if (logging > 0)
   {
      norms[0] = r_norm;
      if (my_id == 0)
      {
  	 printf("BiCGSTABL : L2 norm of b = %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("BiCGSTABL : Initial L2 norm of residual = %e\n", r_norm);
      }
      
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= accuracy*|b| if |b| > 0 */
      epsilon = accuracy * b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= accuracy*|r0| if |b| = 0 */
      epsilon = accuracy * r_norm;
   };

   /* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if (stop_crit) epsilon = accuracy;

   nalu_hypre_ParKrylovCopyVector(r,rh);
   nalu_hypre_ParKrylovCopyVector(r,rt);
   nalu_hypre_ParKrylovCopyVector(x,xh);
   nalu_hypre_ParKrylovClearVector(ut);
   omega = rho = 1.0; alpha = 0.0;

   while ( iter < max_iter && r_norm > epsilon )
   {
      iter += size;

      nalu_hypre_ParKrylovCopyVector(ut,ut1);
      nalu_hypre_ParKrylovCopyVector(rt,rt1);
    
      rho = - omega * rho;    

      rho1 = nalu_hypre_ParKrylovInnerProd(rh,rt1);
      beta = alpha * rho1 / rho;
      rho = rho1;
      dtmp = -beta;
      nalu_hypre_ParKrylovScaleVector(dtmp,ut1);
      nalu_hypre_ParKrylovAxpy(1.0,rt1,ut1);
      precond(precond_data, A, ut1, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,ut2);
      gamma = nalu_hypre_ParKrylovInnerProd(rh,ut2);
      alpha = rho / gamma; dtmp = -alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,ut2,rt1);
      precond(precond_data, A, rt1, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,rt2);
      nalu_hypre_ParKrylovAxpy(alpha,ut1,xh);

      rho1 = nalu_hypre_ParKrylovInnerProd(rh,rt2);
      beta = alpha * rho1 / rho;
      rho = rho1;
      dtmp = -beta;
      nalu_hypre_ParKrylovScaleVector(dtmp,ut1);
      nalu_hypre_ParKrylovAxpy(1.0,rt1,ut1);
      nalu_hypre_ParKrylovScaleVector(dtmp,ut2);
      nalu_hypre_ParKrylovAxpy(1.0,rt2,ut2);
      precond(precond_data, A, ut2, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,ut3);
      gamma = nalu_hypre_ParKrylovInnerProd(rh,ut3);
      alpha = rho / gamma; dtmp = -alpha;
      nalu_hypre_ParKrylovAxpy(dtmp,ut2,rt1);
      nalu_hypre_ParKrylovAxpy(dtmp,ut3,rt2);
      precond(precond_data, A, rt2, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,1.0,A,t,0.0,rt3);
      nalu_hypre_ParKrylovAxpy(alpha,ut1,xh);

      mat[0][0] = 0.0;
      mat[0][1] = 0.0;
      mat[1][0] = 0.0;
      mat[1][1] = 0.0;

      darray[0] = nalu_hypre_ParKrylovInnerProd(rt2,rt2);
      darray[1] = nalu_hypre_ParKrylovInnerProd(rt1,rt2);
      sigma[0]  = darray[0];
      mat[0][0] = sigma[0];
      gammap[0] = darray[1] / sigma[0];

      dtmp = nalu_hypre_ParKrylovInnerProd(rt2,rt3);
      tau[0][1] = dtmp / sigma[0];
      mat[0][1] = tau[0][1] * sigma[0];
      dtmp = -tau[0][1];
      nalu_hypre_ParKrylovAxpy(dtmp,rt2,rt3);
      darray[0] = nalu_hypre_ParKrylovInnerProd(rt3,rt3);
      darray[1] = nalu_hypre_ParKrylovInnerProd(rt1,rt3);
      sigma[1]  = darray[0];
      mat[1][1] = sigma[1];
      gammap[1] = darray[1] / sigma[1];

      gammanp[1] = gammap[1];
      omega = gammanp[1];
      gammanp[0] = gammap[0];
      gammanp[0] = gammanp[0] - tau[0][1] * gammanp[1];
      gammapp[0] = gammanp[1];

      dtmp = gammanp[0];
      nalu_hypre_ParKrylovAxpy(dtmp,rt1,xh);
      dtmp = - gammap[1];
      nalu_hypre_ParKrylovAxpy(dtmp,rt3,rt1);
      dtmp = - gammanp[1];
      nalu_hypre_ParKrylovAxpy(dtmp,ut3,ut1);
      dtmp = - gammanp[0];
      nalu_hypre_ParKrylovAxpy(dtmp,ut2,ut1);
      dtmp = gammapp[0];
      nalu_hypre_ParKrylovAxpy(dtmp,rt2,xh);
      dtmp = - gammap[0];
      nalu_hypre_ParKrylovAxpy(dtmp,rt2,rt1);

      nalu_hypre_ParKrylovCopyVector(ut1,ut);
      nalu_hypre_ParKrylovCopyVector(rt1,rt);
      nalu_hypre_ParKrylovCopyVector(xh,x);

      precond(precond_data, A, x, t);
      nalu_hypre_ParKrylovMatvec(matvec_data,-1.0, A, t, 1.0, r);
      r_norm = nalu_hypre_ParKrylovInnerProd(r,r);
      if ( my_id == 0 && logging )
         printf(" BiCGSTABL : iter %4d - res. norm = %e \n", iter, r_norm);
   }
   precond(precond_data, A, x, t);
   nalu_hypre_ParKrylovCopyVector(t,x);

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (bicgstab_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon) ierr = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetTol
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetTol( void *bicgstab_vdata, double tol )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> tol) = tol;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetMinIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetSize( void *bicgstab_vdata, int size )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> size) = size;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetMaxIter( void *bicgstab_vdata, int max_iter )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> max_iter) = max_iter;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetStopCrit
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetStopCrit( void *bicgstab_vdata, double stop_crit )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int            ierr = 0;
 
   (bicgstab_data -> stop_crit) = stop_crit;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetPrecond( void  *bicgstab_vdata, int  (*precond)(void*,void*,void*,void*),
							   int  (*precond_setup)(void*,void*,void*,void*), void  *precond_data )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data =  (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> precond)        = precond;
   (bicgstab_data -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLSetLogging( void *bicgstab_vdata, int logging)
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   (bicgstab_data -> logging) = logging;
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLGetNumIterations
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLGetNumIterations(void *bicgstab_vdata,int  *num_iterations)
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int              ierr = 0;
 
   *num_iterations = (bicgstab_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int nalu_hypre_BiCGSTABLGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                         double *relative_residual_norm )
{
	nalu_hypre_BiCGSTABLData *bicgstab_data = (nalu_hypre_BiCGSTABLData *) bicgstab_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);
   
   return ierr;
} 

#endif

