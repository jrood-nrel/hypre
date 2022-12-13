/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * RowPatt - Pattern of a row, and functions to manipulate the pattern of
 * a row, particularly merging a pattern with a set of nonzero indices.
 *
 * Implementation and Notes: a full-length array is used to mark nonzeros
 * in the pattern.  Indices must not equal -1, which is the "empty" marker
 * used in the full length array.  It is expected that RowPatt will only be
 * presented with local indices, otherwise the full length array may be very
 * large.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "Common.h"
#include "RowPatt.h"

/*--------------------------------------------------------------------------
 * resize - local function for automatically increasing the size of RowPatt
 *--------------------------------------------------------------------------*/

static void resize(RowPatt *p, NALU_HYPRE_Int newlen)
{
    NALU_HYPRE_Int oldlen, i;

#ifdef PARASAILS_DEBUG
    hypre_printf("RowPatt resize %d\n", newlen);
#endif

    oldlen = p->maxlen;
    p->maxlen = newlen;

    p->ind  = hypre_TReAlloc(p->ind,NALU_HYPRE_Int,   p->maxlen , NALU_HYPRE_MEMORY_HOST);
    p->mark = hypre_TReAlloc(p->mark,NALU_HYPRE_Int,  p->maxlen , NALU_HYPRE_MEMORY_HOST);

    /* initialize the new portion of the mark array */
    for (i=oldlen; i<p->maxlen; i++)
	p->mark[i] = -1;
}

/*--------------------------------------------------------------------------
 * RowPattCreate - Return (a pointer to) a pattern of a row with a maximum
 * of "maxlen" nonzeros.
 *--------------------------------------------------------------------------*/

RowPatt *RowPattCreate(NALU_HYPRE_Int maxlen)
{
    NALU_HYPRE_Int i;
    RowPatt *p = hypre_TAlloc(RowPatt, 1, NALU_HYPRE_MEMORY_HOST);

    p->maxlen   = maxlen;
    p->len      = 0;
    p->prev_len = 0;
    p->ind      = hypre_TAlloc(NALU_HYPRE_Int, maxlen , NALU_HYPRE_MEMORY_HOST);
    p->mark     = hypre_TAlloc(NALU_HYPRE_Int, maxlen , NALU_HYPRE_MEMORY_HOST);
    p->buffer   = NULL;
    p->buflen   = 0;

    for (i=0; i<maxlen; i++)
        p->mark[i] = -1;

    return p;
}

/*--------------------------------------------------------------------------
 * RowPattDestroy - Destroy a row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattDestroy(RowPatt *p)
{
    hypre_TFree(p->ind,NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(p->mark,NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(p->buffer,NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(p,NALU_HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * RowPattReset - Empty the pattern of row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattReset(RowPatt *p)
{
    NALU_HYPRE_Int i;

    for (i=0; i<p->len; i++)
        p->mark[p->ind[i]] = -1;

    p->len      = 0;
    p->prev_len = 0;
}

/*--------------------------------------------------------------------------
 * RowPattMerge - Merge the "len" nonzeros in array "ind" with pattern "p".
 *--------------------------------------------------------------------------*/

void RowPattMerge(RowPatt *p, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind)
{
    NALU_HYPRE_Int i;

    for (i=0; i<len; i++)
    {
	if (ind[i] >= p->maxlen)
	    resize(p, ind[i]*2);

	if (p->mark[ind[i]] == -1)
	{
	    hypre_assert(p->len < p->maxlen);

	    p->mark[ind[i]] = p->len;
            p->ind[p->len] = ind[i];
            p->len++;
	}
    }
}

/*--------------------------------------------------------------------------
 * RowPattMergeExt - Merge the external nonzeros in the array "ind" of
 * length "len" with the pattern "p".  The external indices are those
 * that are less than "beg" or greater than "end".
 *--------------------------------------------------------------------------*/

void RowPattMergeExt(RowPatt *p, NALU_HYPRE_Int len, NALU_HYPRE_Int *ind, NALU_HYPRE_Int num_loc)
{
    NALU_HYPRE_Int i;

    for (i=0; i<len; i++)
    {
        if (ind[i] < num_loc)
	    continue;

	if (ind[i] >= p->maxlen)
	    resize(p, ind[i]*2);

	if (p->mark[ind[i]] == -1)
	{
	    hypre_assert(p->len < p->maxlen);

	    p->mark[ind[i]] = p->len;
            p->ind[p->len] = ind[i];
            p->len++;
	}
    }
}

/*--------------------------------------------------------------------------
 * RowPattGet - Return the pattern of "p".  The length and pointer to the
 * pattern indices are returned through the parameters "lenp" and "indp".
 * A copy of the indices is returned; this copy is destroyed on the next
 * call to RowPattGet or RowPattPrevLevel.
 *--------------------------------------------------------------------------*/

void RowPattGet(RowPatt *p, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp)
{
    NALU_HYPRE_Int len;

    len = p->len;

    if (len > p->buflen)
    {
	hypre_TFree(p->buffer,NALU_HYPRE_MEMORY_HOST);
	p->buflen = len + 100;
	p->buffer = hypre_TAlloc(NALU_HYPRE_Int, p->buflen , NALU_HYPRE_MEMORY_HOST);
    }

    hypre_TMemcpy(p->buffer,  p->ind, NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

    *lenp = len;
    *indp = p->buffer;
}

/*--------------------------------------------------------------------------
 * RowPattPrevLevel - Return the new indices added to the pattern of "p"
 * since the last call to RowPattPrevLevel (or all the indices if never
 * called).  The length and pointer to the pattern indices are returned
 * through the parameters "lenp" and "indp".
 * A copy of the indices is returned; this copy is destroyed on the next
 * call to RowPattGet or RowPattPrevLevel.
 *--------------------------------------------------------------------------*/

void RowPattPrevLevel(RowPatt *p, NALU_HYPRE_Int *lenp, NALU_HYPRE_Int **indp)
{
    NALU_HYPRE_Int len;

    len = p->len - p->prev_len;

    if (len > p->buflen)
    {
	hypre_TFree(p->buffer,NALU_HYPRE_MEMORY_HOST);
	p->buflen = len + 100;
	p->buffer = hypre_TAlloc(NALU_HYPRE_Int, p->buflen , NALU_HYPRE_MEMORY_HOST);
    }

    hypre_TMemcpy(p->buffer,  &p->ind[p->prev_len], NALU_HYPRE_Int, len, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

    *lenp = len;
    *indp = p->buffer;

    p->prev_len = p->len;
}
