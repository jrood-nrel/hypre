/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdio.h>
/* Convert - conversion routines from triangular formats */
/* assumes the matrix has a diagonal */

#define MM_MAX_LINE_LENGTH 1000

NALU_HYPRE_Int convert(FILE *infile, FILE *outfile)
{
    char line[MM_MAX_LINE_LENGTH];
    NALU_HYPRE_Int num_items_read, ret;
    NALU_HYPRE_Int M, N, nz, nnz;
    hypre_longint offset;
    NALU_HYPRE_Int *counts, *pointers;
    NALU_HYPRE_Int row, col;
    NALU_HYPRE_Real value;
    NALU_HYPRE_Int *ind;
    NALU_HYPRE_Real *val;
    NALU_HYPRE_Int i, j;

    /* skip the comment section */
    do
    {
        if (fgets(line, MM_MAX_LINE_LENGTH, infile) == NULL)
            return -1;
    }
    while (line[0] == '%');

    hypre_sscanf(line, "%d %d %d", &M, &N, &nz);

    hypre_printf("%d %d %d\n", M, N, nz);
    nnz = 2*nz - M;

    /* save this position in the file */
    offset = ftell(infile);

    /* allocate space for row counts */
    counts   = hypre_CTAlloc(NALU_HYPRE_Int, M+1, NALU_HYPRE_MEMORY_HOST);
    pointers = hypre_TAlloc(NALU_HYPRE_Int, (M+1) , NALU_HYPRE_MEMORY_HOST);

    /* read the entire matrix */
    ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        counts[row]++;
        if (row != col) /* do not count the diagonal twice */
           counts[col]++;

        ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* allocate space for whole matrix */
    ind = hypre_TAlloc(NALU_HYPRE_Int, nnz , NALU_HYPRE_MEMORY_HOST);
    val = hypre_TAlloc(NALU_HYPRE_Real, nnz , NALU_HYPRE_MEMORY_HOST);

    /* set pointer to beginning of each row */
    pointers[1] = 0;
    for (i=2; i<=M; i++)
        pointers[i] = pointers[i-1] + counts[i-1];

    /* traverse matrix again, putting in the values */
    fseek(infile, offset, SEEK_SET);
    ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    while (ret != EOF)
    {
        val[pointers[row]] = value;
        ind[pointers[row]++] = col;

        if (row != col)
        {
           val[pointers[col]] = value;
           ind[pointers[col]++] = row;
        }

        ret = hypre_fscanf(infile, "%d %d %lf\n", &row, &col, &value);
    }

    /* print out the matrix to the output file */
    hypre_fprintf(outfile, "%d %d %d\n", M, M, nnz);
    for (i=1; i<=M; i++)
        for (j=0; j<counts[i]; j++)
            hypre_fprintf(outfile, "%d %d %.15e\n", i, *ind++, *val++);

    hypre_TFree(counts, NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(pointers, NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(ind, NALU_HYPRE_MEMORY_HOST);
    hypre_TFree(val, NALU_HYPRE_MEMORY_HOST);

    return 0;
}

main(NALU_HYPRE_Int argc, char *argv[])
{
    NALU_HYPRE_Int ret;
    FILE *infile  = fopen(argv[1], "r");
    FILE *outfile = fopen(argv[2], "w");

    ret = convert(infile, outfile);
    if (ret)
       hypre_printf("Conversion failed\n");

    fclose(infile);
    fclose(outfile);
}
