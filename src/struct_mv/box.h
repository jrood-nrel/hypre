/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the Box structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_BOX_HEADER
#define nalu_hypre_BOX_HEADER

#ifndef NALU_HYPRE_MAXDIM
#define NALU_HYPRE_MAXDIM 3
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_Index:
 *   This is used to define indices in index space, or dimension
 *   sizes of boxes.
 *
 *   The spatial dimensions x, y, and z may be specified by the
 *   integers 0, 1, and 2, respectively (see the nalu_hypre_IndexD macro below).
 *   This simplifies the code in the nalu_hypre_Box class by reducing code
 *   replication.
 *--------------------------------------------------------------------------*/

typedef NALU_HYPRE_Int  nalu_hypre_Index[NALU_HYPRE_MAXDIM];
typedef NALU_HYPRE_Int *nalu_hypre_IndexRef;

/*--------------------------------------------------------------------------
 * nalu_hypre_Box:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_Box_struct
{
   nalu_hypre_Index imin;           /* min bounding indices */
   nalu_hypre_Index imax;           /* max bounding indices */
   NALU_HYPRE_Int   ndim;           /* number of dimensions */

} nalu_hypre_Box;

/*--------------------------------------------------------------------------
 * nalu_hypre_BoxArray:
 *   An array of boxes.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_BoxArray_struct
{
   nalu_hypre_Box  *boxes;         /* Array of boxes */
   NALU_HYPRE_Int   size;          /* Size of box array */
   NALU_HYPRE_Int   alloc_size;    /* Size of currently alloced space */
   NALU_HYPRE_Int   ndim;          /* number of dimensions */

} nalu_hypre_BoxArray;

#define nalu_hypre_BoxArrayExcess 10

/*--------------------------------------------------------------------------
 * nalu_hypre_BoxArrayArray:
 *   An array of box arrays.
 *   Since size can be zero, need to store ndim separately.
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_BoxArrayArray_struct
{
   nalu_hypre_BoxArray  **box_arrays;    /* Array of pointers to box arrays */
   NALU_HYPRE_Int         size;          /* Size of box array array */
   NALU_HYPRE_Int         ndim;          /* number of dimensions */

} nalu_hypre_BoxArrayArray;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_Index
 *--------------------------------------------------------------------------*/

#define nalu_hypre_IndexD(index, d)  (index[d])

/* Avoid using these macros */
#define nalu_hypre_IndexX(index)     nalu_hypre_IndexD(index, 0)
#define nalu_hypre_IndexY(index)     nalu_hypre_IndexD(index, 1)
#define nalu_hypre_IndexZ(index)     nalu_hypre_IndexD(index, 2)

/*--------------------------------------------------------------------------
 * Member functions: nalu_hypre_Index
 *--------------------------------------------------------------------------*/

/*----- Avoid using these Index macros -----*/

#define nalu_hypre_SetIndex3(index, ix, iy, iz) \
( nalu_hypre_IndexD(index, 0) = ix,\
  nalu_hypre_IndexD(index, 1) = iy,\
  nalu_hypre_IndexD(index, 2) = iz )

#define nalu_hypre_ClearIndex(index)  nalu_hypre_SetIndex(index, 0)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_Box
 *--------------------------------------------------------------------------*/

#define nalu_hypre_BoxIMin(box)     ((box) -> imin)
#define nalu_hypre_BoxIMax(box)     ((box) -> imax)
#define nalu_hypre_BoxNDim(box)     ((box) -> ndim)

#define nalu_hypre_BoxIMinD(box, d) (nalu_hypre_IndexD(nalu_hypre_BoxIMin(box), d))
#define nalu_hypre_BoxIMaxD(box, d) (nalu_hypre_IndexD(nalu_hypre_BoxIMax(box), d))
#define nalu_hypre_BoxSizeD(box, d) \
nalu_hypre_max(0, (nalu_hypre_BoxIMaxD(box, d) - nalu_hypre_BoxIMinD(box, d) + 1))

#define nalu_hypre_IndexDInBox(index, d, box) \
( nalu_hypre_IndexD(index, d) >= nalu_hypre_BoxIMinD(box, d) && \
  nalu_hypre_IndexD(index, d) <= nalu_hypre_BoxIMaxD(box, d) )

/* The first nalu_hypre_CCBoxIndexRank is better style because it is similar to
   nalu_hypre_BoxIndexRank.  The second one sometimes avoids compiler warnings. */
#define nalu_hypre_CCBoxIndexRank(box, index) 0
#define nalu_hypre_CCBoxIndexRank_noargs() 0
#define nalu_hypre_CCBoxOffsetDistance(box, index) 0

/*----- Avoid using these Box macros -----*/

#define nalu_hypre_BoxSizeX(box)    nalu_hypre_BoxSizeD(box, 0)
#define nalu_hypre_BoxSizeY(box)    nalu_hypre_BoxSizeD(box, 1)
#define nalu_hypre_BoxSizeZ(box)    nalu_hypre_BoxSizeD(box, 2)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_BoxArray
 *--------------------------------------------------------------------------*/

#define nalu_hypre_BoxArrayBoxes(box_array)     ((box_array) -> boxes)
#define nalu_hypre_BoxArrayBox(box_array, i)    &((box_array) -> boxes[(i)])
#define nalu_hypre_BoxArraySize(box_array)      ((box_array) -> size)
#define nalu_hypre_BoxArrayAllocSize(box_array) ((box_array) -> alloc_size)
#define nalu_hypre_BoxArrayNDim(box_array)      ((box_array) -> ndim)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_BoxArrayArray
 *--------------------------------------------------------------------------*/

#define nalu_hypre_BoxArrayArrayBoxArrays(box_array_array) \
((box_array_array) -> box_arrays)
#define nalu_hypre_BoxArrayArrayBoxArray(box_array_array, i) \
((box_array_array) -> box_arrays[(i)])
#define nalu_hypre_BoxArrayArraySize(box_array_array) \
((box_array_array) -> size)
#define nalu_hypre_BoxArrayArrayNDim(box_array_array) \
((box_array_array) -> ndim)

/*--------------------------------------------------------------------------
 * Looping macros:
 *--------------------------------------------------------------------------*/

#define nalu_hypre_ForBoxI(i, box_array) \
for (i = 0; i < nalu_hypre_BoxArraySize(box_array); i++)

#define nalu_hypre_ForBoxArrayI(i, box_array_array) \
for (i = 0; i < nalu_hypre_BoxArrayArraySize(box_array_array); i++)

#define ZYPRE_BOX_PRIVATE nalu_hypre__IN,nalu_hypre__JN,nalu_hypre__I,nalu_hypre__J,nalu_hypre__d,nalu_hypre__i
#define NALU_HYPRE_BOX_PRIVATE ZYPRE_BOX_PRIVATE

#define zypre_BoxLoopDeclare() \
NALU_HYPRE_Int  nalu_hypre__tot, nalu_hypre__div, nalu_hypre__mod;\
NALU_HYPRE_Int  nalu_hypre__block, nalu_hypre__num_blocks;\
NALU_HYPRE_Int  nalu_hypre__d, nalu_hypre__ndim;\
NALU_HYPRE_Int  nalu_hypre__I, nalu_hypre__J, nalu_hypre__IN, nalu_hypre__JN;\
NALU_HYPRE_Int  nalu_hypre__i[NALU_HYPRE_MAXDIM+1], nalu_hypre__n[NALU_HYPRE_MAXDIM+1]

#define zypre_BoxLoopDeclareK(k) \
NALU_HYPRE_Int  nalu_hypre__ikstart##k, nalu_hypre__i0inc##k;\
NALU_HYPRE_Int  nalu_hypre__sk##k[NALU_HYPRE_MAXDIM], nalu_hypre__ikinc##k[NALU_HYPRE_MAXDIM+1]

#define zypre_BoxLoopInit(ndim, loop_size) \
nalu_hypre__ndim = ndim;\
nalu_hypre__n[0] = loop_size[0];\
nalu_hypre__tot = 1;\
for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
{\
   nalu_hypre__n[nalu_hypre__d] = loop_size[nalu_hypre__d];\
   nalu_hypre__tot *= nalu_hypre__n[nalu_hypre__d];\
}\
nalu_hypre__n[nalu_hypre__ndim] = 2;\
nalu_hypre__num_blocks = nalu_hypre_NumThreads();\
if (nalu_hypre__tot < nalu_hypre__num_blocks)\
{\
   nalu_hypre__num_blocks = nalu_hypre__tot;\
}\
if (nalu_hypre__num_blocks > 0)\
{\
   nalu_hypre__div = nalu_hypre__tot / nalu_hypre__num_blocks;\
   nalu_hypre__mod = nalu_hypre__tot % nalu_hypre__num_blocks;\
}

#define zypre_BoxLoopInitK(k, dboxk, startk, stridek, ik) \
nalu_hypre__sk##k[0] = stridek[0];\
nalu_hypre__ikinc##k[0] = 0;\
ik = nalu_hypre_BoxSizeD(dboxk, 0); /* temporarily use ik */\
for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
{\
   nalu_hypre__sk##k[nalu_hypre__d] = ik*stridek[nalu_hypre__d];\
   nalu_hypre__ikinc##k[nalu_hypre__d] = nalu_hypre__ikinc##k[nalu_hypre__d-1] +\
      nalu_hypre__sk##k[nalu_hypre__d] - nalu_hypre__n[nalu_hypre__d-1]*nalu_hypre__sk##k[nalu_hypre__d-1];\
   ik *= nalu_hypre_BoxSizeD(dboxk, nalu_hypre__d);\
}\
nalu_hypre__i0inc##k = nalu_hypre__sk##k[0];\
nalu_hypre__ikinc##k[nalu_hypre__ndim] = 0;\
nalu_hypre__ikstart##k = nalu_hypre_BoxIndexRank(dboxk, startk)

#define zypre_BoxLoopSet() \
nalu_hypre__IN = nalu_hypre__n[0];\
if (nalu_hypre__num_blocks > 1)/* in case user sets num_blocks to 1 */\
{\
   nalu_hypre__JN = nalu_hypre__div + ((nalu_hypre__mod > nalu_hypre__block) ? 1 : 0);\
   nalu_hypre__J = nalu_hypre__block * nalu_hypre__div + nalu_hypre_min(nalu_hypre__mod, nalu_hypre__block);\
   for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
   {\
      nalu_hypre__i[nalu_hypre__d] = nalu_hypre__J % nalu_hypre__n[nalu_hypre__d];\
      nalu_hypre__J /= nalu_hypre__n[nalu_hypre__d];\
   }\
}\
else\
{\
   nalu_hypre__JN = nalu_hypre__tot;\
   for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
   {\
      nalu_hypre__i[nalu_hypre__d] = 0;\
   }\
}\
nalu_hypre__i[nalu_hypre__ndim] = 0

#define zypre_BoxLoopSetK(k, ik) \
ik = nalu_hypre__ikstart##k;\
for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
{\
   ik += nalu_hypre__i[nalu_hypre__d]*nalu_hypre__sk##k[nalu_hypre__d];\
}

#define zypre_BoxLoopInc1() \
nalu_hypre__d = 1;\
while ((nalu_hypre__i[nalu_hypre__d]+2) > nalu_hypre__n[nalu_hypre__d])\
{\
   nalu_hypre__d++;\
}

#define zypre_BoxLoopInc2() \
nalu_hypre__i[nalu_hypre__d]++;\
while (nalu_hypre__d > 1)\
{\
   nalu_hypre__d--;\
   nalu_hypre__i[nalu_hypre__d] = 0;\
}

/* This returns the loop index (of type nalu_hypre_Index) for the current iteration,
 * where the numbering starts at 0.  It works even when threading is turned on,
 * as long as 'index' is declared to be private. */
#define zypre_BoxLoopGetIndex(index) \
index[0] = nalu_hypre__I;\
for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
{\
   index[nalu_hypre__d] = nalu_hypre__i[nalu_hypre__d];\
}

/* Use this before the For macros below to force only one block */
#define zypre_BoxLoopSetOneBlock() nalu_hypre__num_blocks = 1

/* Use this to get the block iteration inside a BoxLoop */
#define zypre_BoxLoopBlock() nalu_hypre__block

#define zypre_BasicBoxLoopInitK(k, stridek) \
nalu_hypre__sk##k[0] = stridek[0];\
nalu_hypre__ikinc##k[0] = 0;\
for (nalu_hypre__d = 1; nalu_hypre__d < nalu_hypre__ndim; nalu_hypre__d++)\
{\
   nalu_hypre__sk##k[nalu_hypre__d] = stridek[nalu_hypre__d];\
   nalu_hypre__ikinc##k[nalu_hypre__d] = nalu_hypre__ikinc##k[nalu_hypre__d-1] +\
      nalu_hypre__sk##k[nalu_hypre__d] - nalu_hypre__n[nalu_hypre__d-1]*nalu_hypre__sk##k[nalu_hypre__d-1];\
}\
nalu_hypre__i0inc##k = nalu_hypre__sk##k[0];\
nalu_hypre__ikinc##k[nalu_hypre__ndim] = 0;\
nalu_hypre__ikstart##k = 0

/*--------------------------------------------------------------------------
 * NOTES - Keep these for reference here and elsewhere in the code
 *--------------------------------------------------------------------------*/

#if 0

#define nalu_hypre_BoxLoop2Begin(loop_size,
dbox1, start1, stride1, i1,
       dbox2, start2, stride2, i2)
{
   /* init nalu_hypre__i1start */
   NALU_HYPRE_Int  nalu_hypre__i1start = nalu_hypre_BoxIndexRank(dbox1, start1);
   NALU_HYPRE_Int  nalu_hypre__i2start = nalu_hypre_BoxIndexRank(dbox2, start2);
   /* declare and set nalu_hypre__s1 */
   nalu_hypre_BoxLoopDeclareS(dbox1, stride1, nalu_hypre__sx1, nalu_hypre__sy1, nalu_hypre__sz1);
   nalu_hypre_BoxLoopDeclareS(dbox2, stride2, nalu_hypre__sx2, nalu_hypre__sy2, nalu_hypre__sz2);
   /* declare and set nalu_hypre__n, nalu_hypre__m, nalu_hypre__dir, nalu_hypre__max,
    *                 nalu_hypre__div, nalu_hypre__mod, nalu_hypre__block, nalu_hypre__num_blocks */
   nalu_hypre_BoxLoopDeclareN(loop_size);

#define nalu_hypre_BoxLoop2For(i, j, k, i1, i2)
   for (nalu_hypre__block = 0; nalu_hypre__block < nalu_hypre__num_blocks; nalu_hypre__block++)
   {
      /* set i and nalu_hypre__n */
      nalu_hypre_BoxLoopSet(i, j, k);
      /* set i1 */
      i1 = nalu_hypre__i1start + i * nalu_hypre__sx1 + j * nalu_hypre__sy1 + k * nalu_hypre__sz1;
      i2 = nalu_hypre__i2start + i * nalu_hypre__sx2 + j * nalu_hypre__sy2 + k * nalu_hypre__sz2;
      for (k = 0; k < nalu_hypre__nz; k++)
      {
         for (j = 0; j < nalu_hypre__ny; j++)
         {
            for (i = 0; i < nalu_hypre__nx; i++)
            {

#define nalu_hypre_BoxLoop2End(i1, i2)
               i1 += nalu_hypre__sx1;
               i2 += nalu_hypre__sx2;
            }
            i1 += nalu_hypre__sy1 - nalu_hypre__nx * nalu_hypre__sx1;
            i2 += nalu_hypre__sy2 - nalu_hypre__nx * nalu_hypre__sx2;
         }
         i1 += nalu_hypre__sz1 - nalu_hypre__ny * nalu_hypre__sy1;
         i2 += nalu_hypre__sz2 - nalu_hypre__ny * nalu_hypre__sy2;
      }
   }
}

/*----------------------------------------
 * Idea 2: Simple version of Idea 3 below
 *----------------------------------------*/

N = 1;
for (d = 0; d < ndim; d++)
{
N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (I = 0; I < N; I++)
{
/* loop body */

for (d = 0; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from nalu_hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

/*----------------------------------------
 * Idea 3: Approach used in the box loops
 *----------------------------------------*/

N = 1;
for (d = 1; d < ndim; d++)
{
N *= n[d];
   i[d] = 0;
   n[d] -= 2; /* this produces a simpler comparison below */
}
i[ndim] = 0;
n[ndim] = 0;
for (J = 0; J < N; J++)
{
for (I = 0; I < n[0]; I++)
   {
      /* loop body */

      i1 += s1[0];
      i2 += s2[0];
   }
   for (d = 1; i[d] > n[d]; d++)
   {
      i[d] = 0;
   }
   i[d]++;
   i1 += s1[d]; /* NOTE: These are different from nalu_hypre__sx1, etc. above */
   i2 += s2[d]; /* The lengths of i, n, and s must be (ndim+1) */
}

#endif
#endif /* #ifndef nalu_hypre_BOX_HEADER */

