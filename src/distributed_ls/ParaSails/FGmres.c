/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * FlexGmres - Preconditioned flexible GMRES algorithm using the
 * ParaSails preconditioner.
 *
 *****************************************************************************/

#include "math.h"
#include "Common.h"
#include "Matrix.h"
#include "ParaSails.h"
#include "_nalu_hypre_blas.h"

static NALU_HYPRE_Real InnerProd(NALU_HYPRE_Int n, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y, MPI_Comm comm)
{
    NALU_HYPRE_Real local_result, result;

    NALU_HYPRE_Int one = 1;
    local_result = nalu_hypre_ddot(&n, x, &one, y, &one);

    nalu_hypre_MPI_Allreduce(&local_result, &result, 1, nalu_hypre_MPI_REAL, nalu_hypre_MPI_SUM, comm);

    return result;
}

static void CopyVector(NALU_HYPRE_Int n, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
    NALU_HYPRE_Int one = 1;
    nalu_hypre_dcopy(&n, x, &one, y, &one);
}

static void ScaleVector(NALU_HYPRE_Int n, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x)
{
    NALU_HYPRE_Int one = 1;
    nalu_hypre_dscal(&n, &alpha, x, &one);
}

static void Axpy(NALU_HYPRE_Int n, NALU_HYPRE_Real alpha, NALU_HYPRE_Real *x, NALU_HYPRE_Real *y)
{
    NALU_HYPRE_Int one = 1;
    nalu_hypre_daxpy(&n, &alpha, x, &one, y, &one);
}

/* simulate 2-D arrays at the cost of some arithmetic */
#define V(i) (&V[(i)*n])
#define W(i) (&W[(i)*n])
#define H(i,j) (H[(j)*m1+(i)])

static void
GeneratePlaneRotation(NALU_HYPRE_Real dx, NALU_HYPRE_Real dy, NALU_HYPRE_Real *cs, NALU_HYPRE_Real *sn)
{
  if (dy == 0.0) {
    *cs = 1.0;
    *sn = 0.0;
  } else if (ABS(dy) > ABS(dx)) {
    NALU_HYPRE_Real temp = dx / dy;
    *sn = 1.0 / sqrt( 1.0 + temp*temp );
    *cs = temp * *sn;
  } else {
    NALU_HYPRE_Real temp = dy / dx;
    *cs = 1.0 / sqrt( 1.0 + temp*temp );
    *sn = temp * *cs;
  }
}

static void ApplyPlaneRotation(NALU_HYPRE_Real *dx, NALU_HYPRE_Real *dy, NALU_HYPRE_Real cs, NALU_HYPRE_Real sn)
{
  NALU_HYPRE_Real temp  =  cs * *dx + sn * *dy;
  *dy = -sn * *dx + cs * *dy;
  *dx = temp;
}

void FGMRES_ParaSails(Matrix *mat, ParaSails *ps, NALU_HYPRE_Real *b, NALU_HYPRE_Real *x,
  NALU_HYPRE_Int dim, NALU_HYPRE_Real tol, NALU_HYPRE_Int max_iter)
{
    NALU_HYPRE_Int mype;
    NALU_HYPRE_Int iter;
    NALU_HYPRE_Real rel_resid;

    NALU_HYPRE_Real *H  = nalu_hypre_TAlloc(NALU_HYPRE_Real, dim*(dim+1) , NALU_HYPRE_MEMORY_HOST);

    /* local problem size */
    NALU_HYPRE_Int n = mat->end_row - mat->beg_row + 1;

    NALU_HYPRE_Int m1 = dim+1; /* used inside H macro */
    NALU_HYPRE_Int i, j, k;
    NALU_HYPRE_Real beta, resid0;

    NALU_HYPRE_Real *s  = nalu_hypre_TAlloc(NALU_HYPRE_Real, (dim+1) , NALU_HYPRE_MEMORY_HOST);
    NALU_HYPRE_Real *cs = nalu_hypre_TAlloc(NALU_HYPRE_Real, dim , NALU_HYPRE_MEMORY_HOST);
    NALU_HYPRE_Real *sn = nalu_hypre_TAlloc(NALU_HYPRE_Real, dim , NALU_HYPRE_MEMORY_HOST);

    NALU_HYPRE_Real *V  = nalu_hypre_TAlloc(NALU_HYPRE_Real, n*(dim+1) , NALU_HYPRE_MEMORY_HOST);
    NALU_HYPRE_Real *W  = nalu_hypre_TAlloc(NALU_HYPRE_Real, n*dim , NALU_HYPRE_MEMORY_HOST);

    MPI_Comm comm = mat->comm;
    nalu_hypre_MPI_Comm_rank(comm, &mype);

    iter = 0;
    do
    {
        /* compute initial residual and its norm */
        MatrixMatvec(mat, x, V(0));                      /* V(0) = A*x        */
        Axpy(n, -1.0, b, V(0));                          /* V(0) = V(0) - b   */
        beta = sqrt(InnerProd(n, V(0), V(0), comm));     /* beta = norm(V(0)) */
        ScaleVector(n, -1.0/beta, V(0));                 /* V(0) = -V(0)/beta */

        /* save very first residual norm */
        if (iter == 0)
            resid0 = beta;

        for (i = 1; i < dim+1; i++)
            s[i] = 0.0;
        s[0] = beta;

        i = -1;
        do
        {
            i++;
            iter++;

            if (ps != NULL)
                ParaSailsApply(ps, V(i), W(i));
            else
                CopyVector(n, V(i), W(i));

            MatrixMatvec(mat, W(i), V(i+1));

            for (k = 0; k <= i; k++)
            {
                H(k, i) = InnerProd(n, V(i+1), V(k), comm);
                /* V(i+1) -= H(k, i) * V(k); */
                Axpy(n, -H(k,i), V(k), V(i+1));
            }

            H(i+1, i) = sqrt(InnerProd(n, V(i+1), V(i+1), comm));
            /* V(i+1) = V(i+1) / H(i+1, i) */
            ScaleVector(n, 1.0 / H(i+1, i), V(i+1));

            for (k = 0; k < i; k++)
                ApplyPlaneRotation(&H(k,i), &H(k+1,i), cs[k], sn[k]);

            GeneratePlaneRotation(H(i,i), H(i+1,i), &cs[i], &sn[i]);
            ApplyPlaneRotation(&H(i,i), &H(i+1,i), cs[i], sn[i]);
            ApplyPlaneRotation(&s[i], &s[i+1], cs[i], sn[i]);

            rel_resid = ABS(s[i+1]) / resid0;
#ifdef PARASAILS_CG_PRINT
            if (mype == 0 && iter % 10 == 0)
               nalu_hypre_printf("Iter (%d): rel. resid. norm: %e\n", iter, rel_resid);
#endif
            if (rel_resid <= tol)
                break;
        }
        while (i+1 < dim && iter+1 <= max_iter);

        /* solve upper triangular system in place */
        for (j = i; j >= 0; j--)
        {
            s[j] /= H(j,j);
            for (k = j-1; k >= 0; k--)
                s[k] -= H(k,j) * s[j];
        }

        /* update the solution */
        for (j = 0; j <= i; j++)
        {
            /* x = x + s[j] * W(j) */
            Axpy(n, s[j], W(j), x);
        }
    }
    while (rel_resid > tol && iter+1 <= max_iter);

    /* compute exact residual norm reduction */
    MatrixMatvec(mat, x, V(0));                         /* V(0) = A*x        */
    Axpy(n, -1.0, b, V(0));                             /* V(0) = V(0) - b   */
    beta = sqrt(InnerProd(n, V(0), V(0), comm));        /* beta = norm(V(0)) */
    rel_resid = beta / resid0;

    if (mype == 0)
        nalu_hypre_printf("Iter (%d): computed rrn    : %e\n", iter, rel_resid);

    nalu_hypre_TFree(H, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(s, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(cs, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(sn, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(V, NALU_HYPRE_MEMORY_HOST);
    nalu_hypre_TFree(W, NALU_HYPRE_MEMORY_HOST);
}

