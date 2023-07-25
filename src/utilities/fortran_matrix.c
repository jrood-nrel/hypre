/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "fortran_matrix.h"
#include "_nalu_hypre_utilities.h"

utilities_FortranMatrix*
utilities_FortranMatrixCreate(void)
{

   utilities_FortranMatrix* mtx;

   mtx = nalu_hypre_TAlloc(utilities_FortranMatrix, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( mtx != NULL );

   mtx->globalHeight = 0;
   mtx->height = 0;
   mtx->width = 0;
   mtx->value = NULL;
   mtx->ownsValues = 0;

   return mtx;
}

void
utilities_FortranMatrixAllocateData( NALU_HYPRE_BigInt  h, NALU_HYPRE_BigInt w,
                                     utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( h > 0 && w > 0 );
   nalu_hypre_assert( mtx != NULL );

   if ( mtx->value != NULL && mtx->ownsValues )
   {
      nalu_hypre_TFree( mtx->value, NALU_HYPRE_MEMORY_HOST);
   }

   mtx->value = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  h * w, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert ( mtx->value != NULL );

   mtx->globalHeight = h;
   mtx->height = h;
   mtx->width = w;
   mtx->ownsValues = 1;
}


void
utilities_FortranMatrixWrap( NALU_HYPRE_Real* v, NALU_HYPRE_BigInt gh, NALU_HYPRE_BigInt  h, NALU_HYPRE_BigInt w,
                             utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( h > 0 && w > 0 );
   nalu_hypre_assert( mtx != NULL );

   if ( mtx->value != NULL && mtx->ownsValues )
   {
      nalu_hypre_TFree( mtx->value, NALU_HYPRE_MEMORY_HOST);
   }

   mtx->value = v;
   nalu_hypre_assert ( mtx->value != NULL );

   mtx->globalHeight = gh;
   mtx->height = h;
   mtx->width = w;
   mtx->ownsValues = 0;
}


void
utilities_FortranMatrixDestroy( utilities_FortranMatrix* mtx )
{

   if ( mtx == NULL )
   {
      return;
   }

   if ( mtx->ownsValues && mtx->value != NULL )
   {
      nalu_hypre_TFree(mtx->value, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(mtx, NALU_HYPRE_MEMORY_HOST);
}

NALU_HYPRE_BigInt
utilities_FortranMatrixGlobalHeight( utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( mtx != NULL );

   return mtx->globalHeight;
}

NALU_HYPRE_BigInt
utilities_FortranMatrixHeight( utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( mtx != NULL );

   return mtx->height;
}

NALU_HYPRE_BigInt
utilities_FortranMatrixWidth( utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( mtx != NULL );

   return mtx->width;
}

NALU_HYPRE_Real*
utilities_FortranMatrixValues( utilities_FortranMatrix* mtx )
{

   nalu_hypre_assert( mtx != NULL );

   return mtx->value;
}

void
utilities_FortranMatrixClear( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, h, w, jump;
   NALU_HYPRE_Real* p;

   nalu_hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         *p = 0.0;
      }
      p += jump;
   }
}

void
utilities_FortranMatrixClearL( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, k, h, w, jump;
   NALU_HYPRE_Real* p;

   nalu_hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   if ( w > h )
   {
      w = h;
   }

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w - 1; j++ )
   {
      k = j + 1;
      p += k;
      for ( i = k; i < h; i++, p++ )
      {
         *p = 0.0;
      }
      p += jump;
   }
}


void
utilities_FortranMatrixSetToIdentity( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt j, h, w, jump;
   NALU_HYPRE_Real* p;

   nalu_hypre_assert( mtx != NULL );

   utilities_FortranMatrixClear( mtx );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight;

   for ( j = 0, p = mtx->value; j < w && j < h; j++, p += jump )
   {
      *p++ = 1.0;
   }

}

void
utilities_FortranMatrixTransposeSquare( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, g, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;
   NALU_HYPRE_Real tmp;

   nalu_hypre_assert( mtx != NULL );

   g = mtx->globalHeight;
   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( h == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      q = p;
      p++;
      q += g;
      for ( i = j + 1; i < h; i++, p++, q += g )
      {
         tmp = *p;
         *p = *q;
         *q = tmp;
      }
      p += ++jump;
   }
}

void
utilities_FortranMatrixSymmetrize( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, g, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;

   nalu_hypre_assert( mtx != NULL );

   g = mtx->globalHeight;
   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( h == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      q = p;
      p++;
      q += g;
      for ( i = j + 1; i < h; i++, p++, q += g )
      {
         *p = *q = (*p + *q) * 0.5;
      }
      p += ++jump;
   }
}

void
utilities_FortranMatrixCopy( utilities_FortranMatrix* src, NALU_HYPRE_Int t,
                             utilities_FortranMatrix* dest )
{

   NALU_HYPRE_BigInt i, j, h, w;
   NALU_HYPRE_BigInt jp, jq, jr;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;
   NALU_HYPRE_Real* r;

   nalu_hypre_assert( src != NULL && dest != NULL );

   h = dest->height;
   w = dest->width;

   jp = dest->globalHeight - h;

   if ( t == 0 )
   {
      nalu_hypre_assert( src->height == h && src->width == w );
      jq = 1;
      jr = src->globalHeight;
   }
   else
   {
      nalu_hypre_assert( src->height == w && src->width == h );
      jr = 1;
      jq = src->globalHeight;
   }

   for ( j = 0, p = dest->value, r = src->value; j < w; j++, p += jp, r += jr )
      for ( i = 0, q = r; i < h; i++, p++, q += jq )
      {
         *p = *q;
      }
}

void
utilities_FortranMatrixIndexCopy( NALU_HYPRE_Int* index,
                                  utilities_FortranMatrix* src, NALU_HYPRE_Int t,
                                  utilities_FortranMatrix* dest )
{

   NALU_HYPRE_BigInt i, j, h, w;
   NALU_HYPRE_BigInt jp, jq, jr;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;
   NALU_HYPRE_Real* r;

   nalu_hypre_assert( src != NULL && dest != NULL );

   h = dest->height;
   w = dest->width;

   jp = dest->globalHeight - h;

   if ( t == 0 )
   {
      nalu_hypre_assert( src->height == h && src->width == w );
      jq = 1;
      jr = src->globalHeight;
   }
   else
   {
      nalu_hypre_assert( src->height == w && src->width == h );
      jr = 1;
      jq = src->globalHeight;
   }

   for ( j = 0, p = dest->value; j < w; j++, p += jp )
   {
      r = src->value + (index[j] - 1) * jr;
      for ( i = 0, q = r; i < h; i++, p++, q += jq )
      {
         *p = *q;
      }
   }
}

void
utilities_FortranMatrixSetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* vec )
{

   NALU_HYPRE_BigInt j, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;

   nalu_hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( vec->height >= h );

   jump = mtx->globalHeight + 1;

   for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h;
         j++, p += jump, q++ )
   {
      *p = *q;
   }

}

void
utilities_FortranMatrixGetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* vec )
{

   NALU_HYPRE_BigInt j, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;

   nalu_hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( vec->height >= h );

   jump = mtx->globalHeight + 1;

   for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h;
         j++, p += jump, q++ )
   {
      *q = *p;
   }

}

void
utilities_FortranMatrixAdd( NALU_HYPRE_Real a,
                            utilities_FortranMatrix* mtxA,
                            utilities_FortranMatrix* mtxB,
                            utilities_FortranMatrix* mtxC )
{

   NALU_HYPRE_BigInt i, j, h, w, jA, jB, jC;
   NALU_HYPRE_Real *pA;
   NALU_HYPRE_Real *pB;
   NALU_HYPRE_Real *pC;

   nalu_hypre_assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

   h = mtxA->height;
   w = mtxA->width;

   nalu_hypre_assert( mtxB->height == h && mtxB->width == w );
   nalu_hypre_assert( mtxC->height == h && mtxC->width == w );

   jA = mtxA->globalHeight - h;
   jB = mtxB->globalHeight - h;
   jC = mtxC->globalHeight - h;

   pA = mtxA->value;
   pB = mtxB->value;
   pC = mtxC->value;

   if ( a == 0.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else if ( a == 1.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pA + *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else if ( a == -1.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pB - *pA;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pA * a + *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
}

void
utilities_FortranMatrixDMultiply( utilities_FortranMatrix* vec,
                                  utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;

   nalu_hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( vec->height == h );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0, q = vec->value; i < h; i++, p++, q++ )
      {
         *p = *p * (*q);
      }
      p += jump;
   }

}

void
utilities_FortranMatrixMultiplyD( utilities_FortranMatrix* mtx,
                                  utilities_FortranMatrix* vec )
{

   NALU_HYPRE_BigInt i, j, h, w, jump;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real* q;

   nalu_hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   nalu_hypre_assert( vec->height == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, q = vec->value, p = mtx->value; j < w; j++, q++ )
   {
      for ( i = 0; i < h; i++, p++)
      {
         *p = *p * (*q);
      }
      p += jump;
   }

}

void
utilities_FortranMatrixMultiply( utilities_FortranMatrix* mtxA, NALU_HYPRE_Int tA,
                                 utilities_FortranMatrix* mtxB, NALU_HYPRE_Int tB,
                                 utilities_FortranMatrix* mtxC )
{
   NALU_HYPRE_BigInt h, w;
   NALU_HYPRE_BigInt i, j, k, l;
   NALU_HYPRE_BigInt iA, kA;
   NALU_HYPRE_BigInt kB, jB;
   NALU_HYPRE_BigInt iC, jC;

   NALU_HYPRE_Real* pAi0;
   NALU_HYPRE_Real* pAik;
   NALU_HYPRE_Real* pB0j;
   NALU_HYPRE_Real* pBkj;
   NALU_HYPRE_Real* pC0j;
   NALU_HYPRE_Real* pCij;

   NALU_HYPRE_Real s;

   nalu_hypre_assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

   h = mtxC->height;
   w = mtxC->width;
   iC = 1;
   jC = mtxC->globalHeight;

   if ( tA == 0 )
   {
      nalu_hypre_assert( mtxA->height == h );
      l = mtxA->width;
      iA = 1;
      kA = mtxA->globalHeight;
   }
   else
   {
      l = mtxA->height;
      nalu_hypre_assert( mtxA->width == h );
      kA = 1;
      iA = mtxA->globalHeight;
   }

   if ( tB == 0 )
   {
      nalu_hypre_assert( mtxB->height == l );
      nalu_hypre_assert( mtxB->width == w );
      kB = 1;
      jB = mtxB->globalHeight;
   }
   else
   {
      nalu_hypre_assert( mtxB->width == l );
      nalu_hypre_assert( mtxB->height == w );
      jB = 1;
      kB = mtxB->globalHeight;
   }

   for ( j = 0, pB0j = mtxB->value, pC0j = mtxC->value; j < w;
         j++, pB0j += jB, pC0j += jC  )
      for ( i = 0, pCij = pC0j, pAi0 = mtxA->value; i < h;
            i++, pCij += iC, pAi0 += iA )
      {
         s = 0.0;
         for ( k = 0, pAik = pAi0, pBkj = pB0j; k < l;
               k++, pAik += kA, pBkj += kB )
         {
            s += *pAik * (*pBkj);
         }
         *pCij = s;
      }
}

NALU_HYPRE_Real
utilities_FortranMatrixFNorm( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, h, w, jump;
   NALU_HYPRE_Real* p;

   NALU_HYPRE_Real norm;

   nalu_hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   norm = 0.0;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         norm += (*p) * (*p);
      }
      p += jump;
   }

   norm = nalu_hypre_sqrt(norm);
   return norm;
}

NALU_HYPRE_Real
utilities_FortranMatrixValue( utilities_FortranMatrix* mtx,
                              NALU_HYPRE_BigInt i, NALU_HYPRE_BigInt j )
{

   NALU_HYPRE_BigInt k;

   nalu_hypre_assert( mtx != NULL );

   nalu_hypre_assert( 1 <= i && i <= mtx->height );
   nalu_hypre_assert( 1 <= j && j <= mtx->width );

   k = i - 1 + (j - 1) * mtx->globalHeight;
   return mtx->value[k];
}

NALU_HYPRE_Real*
utilities_FortranMatrixValuePtr( utilities_FortranMatrix* mtx,
                                 NALU_HYPRE_BigInt i, NALU_HYPRE_BigInt j )
{

   NALU_HYPRE_BigInt k;

   nalu_hypre_assert( mtx != NULL );

   nalu_hypre_assert( 1 <= i && i <= mtx->height );
   nalu_hypre_assert( 1 <= j && j <= mtx->width );

   k = i - 1 + (j - 1) * mtx->globalHeight;
   return mtx->value + k;
}

NALU_HYPRE_Real
utilities_FortranMatrixMaxValue( utilities_FortranMatrix* mtx )
{

   NALU_HYPRE_BigInt i, j, jump;
   NALU_HYPRE_BigInt h, w;
   NALU_HYPRE_Real* p;
   NALU_HYPRE_Real maxVal;

   nalu_hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   maxVal = mtx->value[0];

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
         if ( *p > maxVal )
         {
            maxVal = *p;
         }
      p += jump;
   }

   return maxVal;
}

void
utilities_FortranMatrixSelectBlock( utilities_FortranMatrix* mtx,
                                    NALU_HYPRE_BigInt iFrom, NALU_HYPRE_BigInt iTo,
                                    NALU_HYPRE_BigInt jFrom, NALU_HYPRE_BigInt jTo,
                                    utilities_FortranMatrix* block )
{

   if ( block->value != NULL && block->ownsValues )
   {
      nalu_hypre_TFree( block->value, NALU_HYPRE_MEMORY_HOST);
   }

   block->globalHeight = mtx->globalHeight;
   if ( iTo < iFrom || jTo < jFrom )
   {
      block->height = 0;
      block->width = 0;
      block->value = NULL;
      return;
   }
   block->height = iTo - iFrom + 1;
   block->width = jTo - jFrom + 1;
   block->value = mtx->value + iFrom - 1 + (jFrom - 1) * mtx->globalHeight;
   block->ownsValues = 0;
}

void
utilities_FortranMatrixUpperInv( utilities_FortranMatrix* u )
{

   NALU_HYPRE_BigInt i, j, k;
   NALU_HYPRE_BigInt n, jc, jd;
   NALU_HYPRE_Real v;
   NALU_HYPRE_Real* diag;    /* diag(i) = u(i,i)_original */
   NALU_HYPRE_Real* pin;     /* &u(i-1,n) */
   NALU_HYPRE_Real* pii;     /* &u(i,i) */
   NALU_HYPRE_Real* pij;     /* &u(i,j) */
   NALU_HYPRE_Real* pik;     /* &u(i,k) */
   NALU_HYPRE_Real* pkj;     /* &u(k,j) */
   NALU_HYPRE_Real* pd;      /* &diag(i) */

   n = u->height;
   nalu_hypre_assert( u->width == n );

   diag = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( diag != NULL );

   jc = u->globalHeight;
   jd = jc + 1;

   pii = u->value;
   pd = diag;
   for ( i = 0; i < n; i++, pii += jd, pd++ )
   {
      v = *pd = *pii;
      *pii = 1.0 / v;
   }

   pii -= jd;
   pin = pii - 1;
   pii -= jd;
   pd -= 2;
   for ( i = n - 1; i > 0; i--, pii -= jd, pin--, pd-- )
   {
      pij = pin;
      for ( j = n; j > i; j--, pij -= jc )
      {
         v = 0;
         pik = pii + jc;
         pkj = pij + 1;
         for ( k = i + 1; k <= j; k++, pik += jc, pkj++  )
         {
            v -= (*pik) * (*pkj);
         }
         *pij = v / (*pd);
      }
   }

   nalu_hypre_TFree( diag, NALU_HYPRE_MEMORY_HOST);

}

NALU_HYPRE_Int
utilities_FortranMatrixPrint( utilities_FortranMatrix* mtx, const char *fileName)
{

   NALU_HYPRE_BigInt i, j, h, w, jump;
   NALU_HYPRE_Real* p;
   FILE* fp;

   nalu_hypre_assert( mtx != NULL );

   if ( !(fp = fopen(fileName, "w")) )
   {
      return 1;
   }

   h = mtx->height;
   w = mtx->width;

   nalu_hypre_fprintf(fp, "%ld\n", h);
   nalu_hypre_fprintf(fp, "%ld\n", w);

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         nalu_hypre_fprintf(fp, "%.14e\n", *p);
      }
      p += jump;
   }

   fclose(fp);
   return 0;
}

