/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE_sstruct_ls.h"
#include "NALU_HYPRE_krylov.h"
#include "_nalu_hypre_sstruct_mv.h"
#include "_nalu_hypre_sstruct_ls.h"

#define DEBUG 0

/*     include fortran headers                     */
#ifdef NALU_HYPRE_FORTRAN
#include "fortran.h"
#include "nalu_hypre_sstruct_fortran_test.h"
#endif
/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct.in.default";

typedef NALU_HYPRE_Int Index[3];
typedef NALU_HYPRE_Int ProblemIndex[9];

typedef struct
{
   /* for GridSetExtents */
   NALU_HYPRE_Int                    nboxes;
   ProblemIndex          *ilowers;
   ProblemIndex          *iuppers;
   NALU_HYPRE_Int                   *boxsizes;
   NALU_HYPRE_Int                    max_boxsize;

   /* for GridSetVariables */
   NALU_HYPRE_Int                    nvars;
#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj *vartypes;
#else
   NALU_HYPRE_SStructVariable *vartypes;
#endif

   /* for GridAddVariables */
   NALU_HYPRE_Int                    add_nvars;
   ProblemIndex          *add_indexes;
#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj *add_vartypes;
#else
   NALU_HYPRE_SStructVariable *add_vartypes;
#endif

   /* for GridSetNeighborBox */
   NALU_HYPRE_Int                    glue_nboxes;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   NALU_HYPRE_Int                   *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_index_maps;

   /* for GraphSetStencil */
   NALU_HYPRE_Int                   *stencil_num;

   /* for GraphAddEntries */
   NALU_HYPRE_Int                    graph_nentries;
   ProblemIndex          *graph_ilowers;
   ProblemIndex          *graph_iuppers;
   Index                 *graph_strides;
   NALU_HYPRE_Int                   *graph_vars;
   NALU_HYPRE_Int                   *graph_to_parts;
   ProblemIndex          *graph_to_ilowers;
   ProblemIndex          *graph_to_iuppers;
   Index                 *graph_to_strides;
   NALU_HYPRE_Int                   *graph_to_vars;
   Index                 *graph_index_maps;
   Index                 *graph_index_signs;
   NALU_HYPRE_Int                   *graph_entries;
   NALU_HYPRE_Real            *graph_values;
   NALU_HYPRE_Int                   *graph_boxsizes;

   NALU_HYPRE_Int                    matrix_nentries;
   ProblemIndex          *matrix_ilowers;
   ProblemIndex          *matrix_iuppers;
   Index                 *matrix_strides;
   NALU_HYPRE_Int                   *matrix_vars;
   NALU_HYPRE_Int                   *matrix_entries;
   NALU_HYPRE_Real            *matrix_values;

   Index                  periodic;

} ProblemPartData;

typedef struct
{
   NALU_HYPRE_Int              ndim;
   NALU_HYPRE_Int              nparts;
   ProblemPartData *pdata;
   NALU_HYPRE_Int              max_boxsize;

   NALU_HYPRE_Int              nstencils;
   NALU_HYPRE_Int             *stencil_sizes;
   Index          **stencil_offsets;
   NALU_HYPRE_Int            **stencil_vars;
   NALU_HYPRE_Real     **stencil_values;

   NALU_HYPRE_Int              symmetric_nentries;
   NALU_HYPRE_Int             *symmetric_parts;
   NALU_HYPRE_Int             *symmetric_vars;
   NALU_HYPRE_Int             *symmetric_to_vars;
   NALU_HYPRE_Int             *symmetric_booleans;

   NALU_HYPRE_Int              ns_symmetric;

   Index            rfactor;

   NALU_HYPRE_Int              npools;
   NALU_HYPRE_Int             *pools;   /* array of size nparts */

} ProblemData;

/*--------------------------------------------------------------------------
 * Read routines
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
SScanIntArray( char  *sdata_ptr,
               char **sdata_ptr_ptr,
               NALU_HYPRE_Int    size,
               NALU_HYPRE_Int   *array )
{
   NALU_HYPRE_Int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = strtol(sdata_ptr, &sdata_ptr, 10);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

NALU_HYPRE_Int
SScanProblemIndex( char          *sdata_ptr,
                   char         **sdata_ptr_ptr,
                   NALU_HYPRE_Int            ndim,
                   ProblemIndex   index )
{
   NALU_HYPRE_Int  i;
   char sign[3];

   /* initialize index array */
   for (i = 0; i < 9; i++)
   {
      index[i]   = 0;
   }

   sdata_ptr += strspn(sdata_ptr, " \t\n(");
   switch (ndim)
   {
      case 1:
         nalu_hypre_sscanf(sdata_ptr, "%d%c",
                      &index[0], &sign[0]);
         break;

      case 2:
         nalu_hypre_sscanf(sdata_ptr, "%d%c%d%c",
                      &index[0], &sign[0], &index[1], &sign[1]);
         break;

      case 3:
         nalu_hypre_sscanf(sdata_ptr, "%d%c%d%c%d%c",
                      &index[0], &sign[0], &index[1], &sign[1], &index[2], &sign[2]);
         break;
   }
   sdata_ptr += strcspn(sdata_ptr, ":)");
   if ( *sdata_ptr == ':' )
   {
      /* read in optional shift */
      sdata_ptr += 1;
      switch (ndim)
      {
         case 1:
            nalu_hypre_sscanf(sdata_ptr, "%d", &index[6]);
            break;

         case 2:
            nalu_hypre_sscanf(sdata_ptr, "%d%d", &index[6], &index[7]);
            break;

         case 3:
            nalu_hypre_sscanf(sdata_ptr, "%d%d%d", &index[6], &index[7], &index[8]);
            break;
      }
      /* pre-shift the index */
      for (i = 0; i < ndim; i++)
      {
         index[i] += index[i + 6];
      }
   }
   sdata_ptr += strcspn(sdata_ptr, ")") + 1;

   for (i = 0; i < ndim; i++)
   {
      if (sign[i] == '+')
      {
         index[i + 3] = 1;
      }
   }

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

NALU_HYPRE_Int
ReadData( char         *filename,
          ProblemData  *data_ptr )
{
   ProblemData        data;
   ProblemPartData    pdata;

   NALU_HYPRE_Int                myid;
   FILE              *file;

   char              *sdata = NULL;
   char              *sdata_line;
   char              *sdata_ptr;
   NALU_HYPRE_Int                sdata_size;
   NALU_HYPRE_Int                size;
   NALU_HYPRE_Int                memchunk = 10000;
   NALU_HYPRE_Int                maxline  = 250;

   char               key[250];

   NALU_HYPRE_Int                part, var, entry, s, i, il, iu;

   /*-----------------------------------------------------------
    * Read data file from process 0, then broadcast
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);

   if (myid == 0)
   {
      if ((file = fopen(filename, "r")) == NULL)
      {
         nalu_hypre_printf("Error: can't open input file %s\n", filename);
         exit(1);
      }

      /* allocate initial space, and read first input line */
      sdata_size = 0;
      sdata = nalu_hypre_TAlloc(char,  memchunk, NALU_HYPRE_MEMORY_HOST);
      sdata_line = fgets(sdata, maxline, file);

      s = 0;
      while (sdata_line != NULL)
      {
         sdata_size += strlen(sdata_line) + 1;

         /* allocate more space, if necessary */
         if ((sdata_size + maxline) > s)
         {
            sdata = nalu_hypre_TReAlloc(sdata,  char,  (sdata_size + memchunk), NALU_HYPRE_MEMORY_HOST);
            s = sdata_size + memchunk;
         }

         /* read the next input line */
         sdata_line = fgets((sdata + sdata_size), maxline, file);
      }
   }

   /* broadcast the data size */
   nalu_hypre_MPI_Bcast(&sdata_size, 1, NALU_HYPRE_MPI_INT, 0, nalu_hypre_MPI_COMM_WORLD);

   /* broadcast the data */
   sdata = nalu_hypre_TReAlloc(sdata,  char,  sdata_size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Bcast(sdata, sdata_size, nalu_hypre_MPI_CHAR, 0, nalu_hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Parse the data and fill ProblemData structure
    *-----------------------------------------------------------*/

   data.max_boxsize = 0;
   data.symmetric_nentries = 0;
   data.symmetric_parts    = NULL;
   data.symmetric_vars     = NULL;
   data.symmetric_to_vars  = NULL;
   data.symmetric_booleans = NULL;
   data.ns_symmetric = 0;

   sdata_line = sdata;
   while (sdata_line < (sdata + sdata_size))
   {
      sdata_ptr = sdata_line;

      if ( ( nalu_hypre_sscanf(sdata_ptr, "%s", key) > 0 ) && ( sdata_ptr[0] != '#' ) )
      {
         sdata_ptr += strcspn(sdata_ptr, " \t\n");

         if ( strcmp(key, "GridCreate:") == 0 )
         {
            data.ndim = strtol(sdata_ptr, &sdata_ptr, 10);
            data.nparts = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pdata = nalu_hypre_CTAlloc(ProblemPartData,  data.nparts, NALU_HYPRE_MEMORY_HOST);
         }
         else if ( strcmp(key, "GridSetExtents:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.nboxes % 10) == 0)
            {
               size = pdata.nboxes + 10;
               pdata.ilowers =
                  nalu_hypre_TReAlloc(pdata.ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.iuppers =
                  nalu_hypre_TReAlloc(pdata.iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.boxsizes =
                  nalu_hypre_TReAlloc(pdata.boxsizes,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.ilowers[pdata.nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.iuppers[pdata.nboxes]);
            /* check use of +- in GridSetExtents */
            il = 1;
            iu = 1;
            for (i = 0; i < data.ndim; i++)
            {
               il *= pdata.ilowers[pdata.nboxes][i + 3];
               iu *= pdata.iuppers[pdata.nboxes][i + 3];
            }
            if ( (il != 0) || (iu != 1) )
            {
               nalu_hypre_printf("Error: Invalid use of `+-' in GridSetExtents\n");
               exit(1);
            }
            pdata.boxsizes[pdata.nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[pdata.nboxes] *=
                  (pdata.iuppers[pdata.nboxes][i] -
                   pdata.ilowers[pdata.nboxes][i] + 2);
            }
            pdata.max_boxsize =
               nalu_hypre_max(pdata.max_boxsize, pdata.boxsizes[pdata.nboxes]);
            pdata.nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetVariables:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            pdata.nvars = strtol(sdata_ptr, &sdata_ptr, 10);
#ifdef NALU_HYPRE_FORTRAN
            pdata.vartypes =  nalu_hypre_CTAlloc(nalu_hypre_F90_Obj,  pdata.nvars, NALU_HYPRE_MEMORY_HOST);
#else
            pdata.vartypes = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  pdata.nvars, NALU_HYPRE_MEMORY_HOST);
#endif
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          pdata.nvars, (NALU_HYPRE_Int *) pdata.vartypes);
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridAddVariables:") == 0 )
         {
            /* TODO */
            nalu_hypre_printf("GridAddVariables not yet implemented!\n");
            exit(1);
         }
         else if ( strcmp(key, "GridSetNeighborBox:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.glue_nboxes % 10) == 0)
            {
               size = pdata.glue_nboxes + 10;
               pdata.glue_ilowers =
                  nalu_hypre_TReAlloc(pdata.glue_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_iuppers =
                  nalu_hypre_TReAlloc(pdata.glue_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_parts =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_parts,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_ilowers =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_iuppers =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_index_maps =
                  nalu_hypre_TReAlloc(pdata.glue_index_maps,  Index,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_iuppers[pdata.glue_nboxes]);
            pdata.glue_nbor_parts[pdata.glue_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_iuppers[pdata.glue_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_maps[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_maps[pdata.glue_nboxes][i] = i;
            }
            pdata.glue_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetPeriodic:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, pdata.periodic);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.periodic[i] = 0;
            }
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "StencilCreate:") == 0 )
         {
            data.nstencils = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_sizes   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
            data.stencil_offsets = nalu_hypre_CTAlloc(Index *,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
            data.stencil_vars    = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
            data.stencil_values  = nalu_hypre_CTAlloc(NALU_HYPRE_Real *,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.nstencils, data.stencil_sizes);
            for (s = 0; s < data.nstencils; s++)
            {
               data.stencil_offsets[s] =
                  nalu_hypre_CTAlloc(Index,  data.stencil_sizes[s], NALU_HYPRE_MEMORY_HOST);
               data.stencil_vars[s] =
                  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.stencil_sizes[s], NALU_HYPRE_MEMORY_HOST);
               data.stencil_values[s] =
                  nalu_hypre_CTAlloc(NALU_HYPRE_Real,  data.stencil_sizes[s], NALU_HYPRE_MEMORY_HOST);
            }
         }
         else if ( strcmp(key, "StencilSetEntry:") == 0 )
         {
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            entry = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.ndim, data.stencil_offsets[s][entry]);
            for (i = data.ndim; i < 3; i++)
            {
               data.stencil_offsets[s][entry][i] = 0;
            }
            data.stencil_vars[s][entry] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_values[s][entry] = (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "GraphSetStencil:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            var = strtol(sdata_ptr, &sdata_ptr, 10);
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if (pdata.stencil_num == NULL)
            {
               pdata.stencil_num = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  pdata.nvars, NALU_HYPRE_MEMORY_HOST);
            }
            pdata.stencil_num[var] = s;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GraphAddEntries:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.graph_nentries % 10) == 0)
            {
               size = pdata.graph_nentries + 10;
               pdata.graph_ilowers =
                  nalu_hypre_TReAlloc(pdata.graph_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_iuppers =
                  nalu_hypre_TReAlloc(pdata.graph_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_strides =
                  nalu_hypre_TReAlloc(pdata.graph_strides,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_vars =
                  nalu_hypre_TReAlloc(pdata.graph_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_to_parts =
                  nalu_hypre_TReAlloc(pdata.graph_to_parts,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_to_ilowers =
                  nalu_hypre_TReAlloc(pdata.graph_to_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_to_iuppers =
                  nalu_hypre_TReAlloc(pdata.graph_to_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_to_strides =
                  nalu_hypre_TReAlloc(pdata.graph_to_strides,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_to_vars =
                  nalu_hypre_TReAlloc(pdata.graph_to_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_index_maps =
                  nalu_hypre_TReAlloc(pdata.graph_index_maps,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_index_signs =
                  nalu_hypre_TReAlloc(pdata.graph_index_signs,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_entries =
                  nalu_hypre_TReAlloc(pdata.graph_entries,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_values =
                  nalu_hypre_TReAlloc(pdata.graph_values,  NALU_HYPRE_Real,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.graph_boxsizes =
                  nalu_hypre_TReAlloc(pdata.graph_boxsizes,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_ilowers[pdata.graph_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_iuppers[pdata.graph_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_strides[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_strides[pdata.graph_nentries][i] = 1;
            }
            pdata.graph_vars[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_to_parts[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_ilowers[pdata.graph_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_iuppers[pdata.graph_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_to_strides[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_to_strides[pdata.graph_nentries][i] = 1;
            }
            pdata.graph_to_vars[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_index_maps[pdata.graph_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_index_maps[pdata.graph_nentries][i] = i;
            }
            for (i = 0; i < 3; i++)
            {
               pdata.graph_index_signs[pdata.graph_nentries][i] = 1;
               if ( pdata.graph_to_iuppers[pdata.graph_nentries][i] <
                    pdata.graph_to_ilowers[pdata.graph_nentries][i] )
               {
                  pdata.graph_index_signs[pdata.graph_nentries][i] = -1;
               }
            }
            pdata.graph_entries[pdata.graph_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_values[pdata.graph_nentries] =
               (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.graph_boxsizes[pdata.graph_nentries] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[pdata.graph_nentries] *=
                  (pdata.graph_iuppers[pdata.graph_nentries][i] -
                   pdata.graph_ilowers[pdata.graph_nentries][i] + 1);
            }
            pdata.graph_nentries++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            if ((data.symmetric_nentries % 10) == 0)
            {
               size = data.symmetric_nentries + 10;
               data.symmetric_parts =
                  nalu_hypre_TReAlloc(data.symmetric_parts,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_vars =
                  nalu_hypre_TReAlloc(data.symmetric_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_to_vars =
                  nalu_hypre_TReAlloc(data.symmetric_to_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_booleans =
                  nalu_hypre_TReAlloc(data.symmetric_booleans,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            data.symmetric_parts[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_vars[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_to_vars[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_booleans[data.symmetric_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_nentries++;
         }
         else if ( strcmp(key, "MatrixSetNSSymmetric:") == 0 )
         {
            data.ns_symmetric = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "MatrixSetValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matrix_nentries % 10) == 0)
            {
               size = pdata.matrix_nentries + 10;
               pdata.matrix_ilowers =
                  nalu_hypre_TReAlloc(pdata.matrix_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matrix_iuppers =
                  nalu_hypre_TReAlloc(pdata.matrix_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matrix_strides =
                  nalu_hypre_TReAlloc(pdata.matrix_strides,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matrix_vars =
                  nalu_hypre_TReAlloc(pdata.matrix_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matrix_entries =
                  nalu_hypre_TReAlloc(pdata.matrix_entries,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matrix_values =
                  nalu_hypre_TReAlloc(pdata.matrix_values,  NALU_HYPRE_Real,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matrix_ilowers[pdata.matrix_nentries]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matrix_iuppers[pdata.matrix_nentries]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.matrix_strides[pdata.matrix_nentries]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.matrix_strides[pdata.matrix_nentries][i] = 1;
            }
            pdata.matrix_vars[pdata.matrix_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matrix_entries[pdata.matrix_nentries] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matrix_values[pdata.matrix_nentries] =
               (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.matrix_nentries++;
            data.pdata[part] = pdata;
         }

         else if ( strcmp(key, "rfactor:") == 0 )
         {
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, data.rfactor);
            for (i = data.ndim; i < 3; i++)
            {
               data.rfactor[i] = 1;
            }
         }

         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.nparts, NALU_HYPRE_MEMORY_HOST);
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools[part] = i;
         }
      }

      sdata_line += strlen(sdata_line) + 1;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         nalu_hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   nalu_hypre_TFree(sdata, NALU_HYPRE_MEMORY_HOST);

   *data_ptr = data;
   return 0;
}

/*--------------------------------------------------------------------------
 * Distribute routines
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
MapProblemIndex( ProblemIndex index,
                 Index        m )
{
   /* un-shift the index */
   index[0] -= index[6];
   index[1] -= index[7];
   index[2] -= index[8];
   /* map the index */
   index[0] = m[0] * index[0] + (m[0] - 1) * index[3];
   index[1] = m[1] * index[1] + (m[1] - 1) * index[4];
   index[2] = m[2] * index[2] + (m[2] - 1) * index[5];
   /* pre-shift the new mapped index */
   index[0] += index[6];
   index[1] += index[7];
   index[2] += index[8];

   return 0;
}

NALU_HYPRE_Int
IntersectBoxes( ProblemIndex ilower1,
                ProblemIndex iupper1,
                ProblemIndex ilower2,
                ProblemIndex iupper2,
                ProblemIndex int_ilower,
                ProblemIndex int_iupper )
{
   NALU_HYPRE_Int d, size;

   size = 1;
   for (d = 0; d < 3; d++)
   {
      int_ilower[d] = nalu_hypre_max(ilower1[d], ilower2[d]);
      int_iupper[d] = nalu_hypre_min(iupper1[d], iupper2[d]);
      size *= nalu_hypre_max(0, (int_iupper[d] - int_ilower[d] + 1));
   }

   return size;
}

NALU_HYPRE_Int
DistributeData( ProblemData   global_data,
                Index        *refine,
                Index        *distribute,
                Index        *block,
                NALU_HYPRE_Int           num_procs,
                NALU_HYPRE_Int           myid,
                ProblemData  *data_ptr )
{
   ProblemData      data = global_data;
   ProblemPartData  pdata;
   NALU_HYPRE_Int             *pool_procs;
   NALU_HYPRE_Int              np, pid;
   NALU_HYPRE_Int              pool, part, box, entry, p, q, r, i, d, dmap, sign, size;
   Index            m, mmap, n;
   ProblemIndex     int_ilower, int_iupper;

   /* determine first process number in each pool */
   pool_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  (data.npools + 1), NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < data.nparts; part++)
   {
      pool = data.pools[part] + 1;
      np = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pool_procs[pool] = nalu_hypre_max(pool_procs[pool], np);

   }
   pool_procs[0] = 0;
   for (pool = 1; pool < (data.npools + 1); pool++)
   {
      pool_procs[pool] = pool_procs[pool - 1] + pool_procs[pool];
   }

   /* check number of processes */
   if (pool_procs[data.npools] != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processes or process topology \n");
      exit(1);
   }

   /* modify part data */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      pool  = data.pools[part];
      np  = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pid = myid - pool_procs[pool];

      if ( (pid < 0) || (pid >= np) )
      {
         /* none of this part data lives on this process */
         pdata.nboxes = 0;
         pdata.glue_nboxes = 0;
         pdata.graph_nentries = 0;
         pdata.matrix_nentries = 0;
      }
      else
      {
         /* refine boxes */
         m[0] = refine[part][0];
         m[1] = refine[part][1];
         m[2] = refine[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
            }

            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);
            }
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);
            }
         }

         /* refine and distribute boxes */
         m[0] = distribute[part][0];
         m[1] = distribute[part][1];
         m[2] = distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            p = pid % m[0];
            q = ((pid - p) / m[0]) % m[1];
            r = (pid - p - q * m[0]) / (m[0] * m[1]);

            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               pdata.ilowers[box][0] = pdata.ilowers[box][0] + p * n[0];
               pdata.ilowers[box][1] = pdata.ilowers[box][1] + q * n[1];
               pdata.ilowers[box][2] = pdata.ilowers[box][2] + r * n[2];
               pdata.iuppers[box][0] = pdata.iuppers[box][0] + p * n[0];
               pdata.iuppers[box][1] = pdata.iuppers[box][1] + q * n[1];
               pdata.iuppers[box][2] = pdata.iuppers[box][2] + r * n[2];
            }

            i = 0;
            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  size = IntersectBoxes(pdata.graph_ilowers[entry],
                                        pdata.graph_iuppers[entry],
                                        pdata.ilowers[box],
                                        pdata.iuppers[box],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[entry][d];
                        sign = pdata.graph_index_signs[entry][d];
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign *
                           (int_iupper[d] - pdata.graph_ilowers[entry][d]);
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[entry][dmap] + sign *
                           (int_ilower[d] - pdata.graph_ilowers[entry][d]);
                        pdata.graph_ilowers[i][d] = int_ilower[d];
                        pdata.graph_iuppers[i][d] = int_iupper[d];
                        pdata.graph_strides[i][d] =
                           pdata.graph_strides[entry][d];
                        pdata.graph_to_strides[i][d] =
                           pdata.graph_to_strides[entry][d];
                        pdata.graph_index_maps[i][d]  = dmap;
                        pdata.graph_index_signs[i][d] = sign;
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.graph_ilowers[i][d] =
                           pdata.graph_ilowers[entry][d];
                        pdata.graph_iuppers[i][d] =
                           pdata.graph_iuppers[entry][d];
                        pdata.graph_to_ilowers[i][d] =
                           pdata.graph_to_ilowers[entry][d];
                        pdata.graph_to_iuppers[i][d] =
                           pdata.graph_to_iuppers[entry][d];
                     }
                     pdata.graph_vars[i]     = pdata.graph_vars[entry];
                     pdata.graph_to_parts[i] = pdata.graph_to_parts[entry];
                     pdata.graph_to_vars[i]  = pdata.graph_to_vars[entry];
                     pdata.graph_entries[i]  = pdata.graph_entries[entry];
                     pdata.graph_values[i]   = pdata.graph_values[entry];
                     i++;
                     break;
                  }
               }
            }
            pdata.graph_nentries = i;

            i = 0;
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  size = IntersectBoxes(pdata.matrix_ilowers[entry],
                                        pdata.matrix_iuppers[entry],
                                        pdata.ilowers[box],
                                        pdata.iuppers[box],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matrix_ilowers[i][d] = int_ilower[d];
                        pdata.matrix_iuppers[i][d] = int_iupper[d];
                        pdata.matrix_strides[i][d] =
                           pdata.matrix_strides[entry][d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matrix_ilowers[i][d] =
                           pdata.matrix_ilowers[entry][d];
                        pdata.matrix_iuppers[i][d] =
                           pdata.matrix_iuppers[entry][d];
                     }
                     pdata.matrix_vars[i]    = pdata.matrix_vars[entry];
                     pdata.matrix_entries[i]  = pdata.matrix_entries[entry];
                     pdata.matrix_values[i]   = pdata.matrix_values[entry];
                     i++;
                     break;
                  }
               }
            }
            pdata.matrix_nentries = i;
         }

         /* refine and block boxes */
         m[0] = block[part][0];
         m[1] = block[part][1];
         m[2] = block[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            pdata.ilowers = nalu_hypre_TReAlloc(pdata.ilowers,  ProblemIndex,
                                           m[0] * m[1] * m[2] * pdata.nboxes, NALU_HYPRE_MEMORY_HOST);
            pdata.iuppers = nalu_hypre_TReAlloc(pdata.iuppers,  ProblemIndex,
                                           m[0] * m[1] * m[2] * pdata.nboxes, NALU_HYPRE_MEMORY_HOST);
            pdata.boxsizes = nalu_hypre_TReAlloc(pdata.boxsizes,  NALU_HYPRE_Int,
                                            m[0] * m[1] * m[2] * pdata.nboxes, NALU_HYPRE_MEMORY_HOST);
            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);

               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               i = box;
               for (r = 0; r < m[2]; r++)
               {
                  for (q = 0; q < m[1]; q++)
                  {
                     for (p = 0; p < m[0]; p++)
                     {
                        pdata.ilowers[i][0] = pdata.ilowers[box][0] + p * n[0];
                        pdata.ilowers[i][1] = pdata.ilowers[box][1] + q * n[1];
                        pdata.ilowers[i][2] = pdata.ilowers[box][2] + r * n[2];
                        pdata.iuppers[i][0] = pdata.iuppers[box][0] + p * n[0];
                        pdata.iuppers[i][1] = pdata.iuppers[box][1] + q * n[1];
                        pdata.iuppers[i][2] = pdata.iuppers[box][2] + r * n[2];
                        for (d = 3; d < 9; d++)
                        {
                           pdata.ilowers[i][d] = pdata.ilowers[box][d];
                           pdata.iuppers[i][d] = pdata.iuppers[box][d];
                        }
                        i += pdata.nboxes;
                     }
                  }
               }
            }
            pdata.nboxes *= m[0] * m[1] * m[2];

            for (entry = 0; entry < pdata.graph_nentries; entry++)
            {
               MapProblemIndex(pdata.graph_ilowers[entry], m);
               MapProblemIndex(pdata.graph_iuppers[entry], m);
               mmap[0] = m[pdata.graph_index_maps[entry][0]];
               mmap[1] = m[pdata.graph_index_maps[entry][1]];
               mmap[2] = m[pdata.graph_index_maps[entry][2]];
               MapProblemIndex(pdata.graph_to_ilowers[entry], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[entry], mmap);
            }
            for (entry = 0; entry < pdata.matrix_nentries; entry++)
            {
               MapProblemIndex(pdata.matrix_ilowers[entry], m);
               MapProblemIndex(pdata.matrix_iuppers[entry], m);
            }
         }

         /* map remaining ilowers & iuppers */
         m[0] = refine[part][0] * block[part][0] * distribute[part][0];
         m[1] = refine[part][1] * block[part][1] * distribute[part][1];
         m[2] = refine[part][2] * block[part][2] * distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.glue_nboxes; box++)
            {
               MapProblemIndex(pdata.glue_ilowers[box], m);
               MapProblemIndex(pdata.glue_iuppers[box], m);
               mmap[0] = m[pdata.glue_index_maps[box][0]];
               mmap[1] = m[pdata.glue_index_maps[box][1]];
               mmap[2] = m[pdata.glue_index_maps[box][2]];
               MapProblemIndex(pdata.glue_nbor_ilowers[box], mmap);
               MapProblemIndex(pdata.glue_nbor_iuppers[box], mmap);
            }
         }

         /* compute box sizes, etc. */
         pdata.max_boxsize = 0;
         for (box = 0; box < pdata.nboxes; box++)
         {
            pdata.boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[box] *=
                  (pdata.iuppers[box][i] - pdata.ilowers[box][i] + 2);
            }
            pdata.max_boxsize =
               nalu_hypre_max(pdata.max_boxsize, pdata.boxsizes[box]);
         }
         for (box = 0; box < pdata.graph_nentries; box++)
         {
            pdata.graph_boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[box] *=
                  (pdata.graph_iuppers[box][i] -
                   pdata.graph_ilowers[box][i] + 1);
            }
         }
      }

      if (pdata.nboxes == 0)
      {
         nalu_hypre_TFree(pdata.ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.boxsizes, NALU_HYPRE_MEMORY_HOST);
         pdata.max_boxsize = 0;
      }

      if (pdata.glue_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.glue_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_maps, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nentries == 0)
      {
         nalu_hypre_TFree(pdata.graph_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_index_maps, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_index_signs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_values, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_boxsizes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matrix_nentries == 0)
      {
         nalu_hypre_TFree(pdata.matrix_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_values, NALU_HYPRE_MEMORY_HOST);
      }

      data.pdata[part] = pdata;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         nalu_hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   nalu_hypre_TFree(pool_procs, NALU_HYPRE_MEMORY_HOST);

   *data_ptr = data;
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy data
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
DestroyData( ProblemData   data )
{
   ProblemPartData  pdata;
   NALU_HYPRE_Int              part, s;

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      if (pdata.nboxes > 0)
      {
         nalu_hypre_TFree(pdata.ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.boxsizes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         nalu_hypre_TFree(pdata.vartypes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.add_nvars > 0)
      {
         nalu_hypre_TFree(pdata.add_indexes, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.add_vartypes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.glue_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.glue_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_maps, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         nalu_hypre_TFree(pdata.stencil_num, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nentries > 0)
      {
         nalu_hypre_TFree(pdata.graph_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_to_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_index_maps, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_index_signs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_values, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.graph_boxsizes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matrix_nentries > 0)
      {
         nalu_hypre_TFree(pdata.matrix_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matrix_values, NALU_HYPRE_MEMORY_HOST);
      }

   }
   nalu_hypre_TFree(data.pdata, NALU_HYPRE_MEMORY_HOST);

   for (s = 0; s < data.nstencils; s++)
   {
      nalu_hypre_TFree(data.stencil_offsets[s], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.stencil_vars[s], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.stencil_values[s], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(data.stencil_sizes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(data.stencil_offsets, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(data.stencil_vars, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(data.stencil_values, NALU_HYPRE_MEMORY_HOST);

   if (data.symmetric_nentries > 0)
   {
      nalu_hypre_TFree(data.symmetric_parts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_vars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_to_vars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_booleans, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(data.pools, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
GetVariableBox( Index  cell_ilower,
                Index  cell_iupper,
                NALU_HYPRE_Int    int_vartype,
                Index  var_ilower,
                Index  var_iupper )
{
   NALU_HYPRE_Int ierr = 0;
#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj  vartype = (nalu_hypre_F90_Obj) int_vartype;
#else
   NALU_HYPRE_SStructVariable  vartype = (NALU_HYPRE_SStructVariable) int_vartype;
#endif

   var_ilower[0] = cell_ilower[0];
   var_ilower[1] = cell_ilower[1];
   var_ilower[2] = cell_ilower[2];
   var_iupper[0] = cell_iupper[0];
   var_iupper[1] = cell_iupper[1];
   var_iupper[2] = cell_iupper[2];

   switch (vartype)
   {
      case NALU_HYPRE_SSTRUCT_VARIABLE_CELL:
         var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 0;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_NODE:
         var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 1;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_XFACE:
         var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 0;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_YFACE:
         var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 0;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZFACE:
         var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 1;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_XEDGE:
         var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 1;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_YEDGE:
         var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 1;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 0;
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_UNDEFINED:
         break;
   }

   return ierr;
}
/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
PrintUsage( char *progname,
            NALU_HYPRE_Int   myid )
{
   if ( myid == 0 )
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Usage: %s [<options>]\n", progname);
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -in <filename> : input file (default is `%s')\n",
                   infile_default);
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
      nalu_hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
      nalu_hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
      nalu_hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
      nalu_hypre_printf("  -solver <ID>        : solver ID (default = 39)\n");
      nalu_hypre_printf("  -print             : print out the system\n");
      nalu_hypre_printf("  -v <n_pre> <n_post>: SysPFMG and Struct- # of pre and post relax\n");
      nalu_hypre_printf("  -sym <s>           : Struct- symmetric storage (1) or not (0)\n");

      nalu_hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
main( NALU_HYPRE_Int   argc,
      char *argv[] )
{
   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   NALU_HYPRE_Int                   nparts;
   NALU_HYPRE_Int                  *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   NALU_HYPRE_Int                   solver_id;
   NALU_HYPRE_Int                   print_system;

#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj   grid;
   nalu_hypre_F90_Obj  *stencils;
   nalu_hypre_F90_Obj   graph;
   nalu_hypre_F90_Obj   A;
   nalu_hypre_F90_Obj   T, parA;
   nalu_hypre_F90_Obj   b;
   nalu_hypre_F90_Obj   x;
   nalu_hypre_F90_Obj   parb, parx;
   nalu_hypre_F90_Obj   solver;

   nalu_hypre_F90_Obj   cell_grid;
#else
   NALU_HYPRE_SStructGrid     grid;
   NALU_HYPRE_SStructStencil *stencils;
   NALU_HYPRE_SStructGraph    graph;
   NALU_HYPRE_SStructMatrix   A;
   NALU_HYPRE_ParCSRMatrix    T, parA;
   NALU_HYPRE_SStructVector   b;
   NALU_HYPRE_SStructVector   x;
   NALU_HYPRE_ParVector       parb, parx;
   NALU_HYPRE_SStructSolver   solver;

   NALU_HYPRE_StructGrid      cell_grid;
#endif

   nalu_hypre_Box            *bounding_box;
   NALU_HYPRE_Real            h;

   NALU_HYPRE_Int                 **bdryRanks, *bdryRanksCnt;

   Index                 ilower, iupper;
   Index                 index, to_index;
   NALU_HYPRE_Real           *values;

   NALU_HYPRE_Int                   num_iterations;
   NALU_HYPRE_Real            final_res_norm;

   NALU_HYPRE_Int                   num_procs, myid;
   NALU_HYPRE_Int                   time_index;

   NALU_HYPRE_Int                   n_pre, n_post;

   NALU_HYPRE_Int                   arg_index, part, box, var, entry, s, i, j, k;

#ifdef NALU_HYPRE_FORTRAN
   nalu_hypre_F90_Obj  long_temp_COMM;
   NALU_HYPRE_Int       temp_COMM;
   NALU_HYPRE_Int zero = 0;
   NALU_HYPRE_Int one = 1;
   NALU_HYPRE_Int twenty = 20;
   NALU_HYPRE_Int for_NALU_HYPRE_PARCSR = 5555;

   NALU_HYPRE_Real ftol = 1.e-8;
#endif

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   if (argc > 1)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
   }

   ReadData(infile, &global_data);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nparts = global_data.nparts;

   parts      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   refine     = nalu_hypre_TAlloc(Index,  nparts, NALU_HYPRE_MEMORY_HOST);
   distribute = nalu_hypre_TAlloc(Index,  nparts, NALU_HYPRE_MEMORY_HOST);
   block      = nalu_hypre_TAlloc(Index,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      parts[part] = part;
      for (j = 0; j < 3; j++)
      {
         refine[part][j]     = 1;
         distribute[part][j] = 1;
         block[part][j]      = 1;
      }
   }

   print_system = 0;

   n_pre  = 1;
   n_post = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-pt") == 0 )
      {
         arg_index++;
         nparts = 0;
         while ( strncmp(argv[arg_index], "-", 1) != 0 )
         {
            parts[nparts++] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               refine[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               distribute[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               block[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
         break;
      }
      else
      {
         break;
      }
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, refine, distribute, block,
                  num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

#ifdef NALU_HYPRE_FORTRAN
   temp_COMM = (NALU_HYPRE_Int) nalu_hypre_MPI_COMM_WORLD;
   long_temp_COMM = (nalu_hypre_F90_Obj) nalu_hypre_MPI_COMM_WORLD;
#endif

   time_index = nalu_hypre_InitializeTiming("SStruct Interface");
   nalu_hypre_BeginTiming(time_index);

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGridCreate(&temp_COMM, &data.ndim, &data.nparts, &grid);
#else
   NALU_HYPRE_SStructGridCreate(nalu_hypre_MPI_COMM_WORLD, data.ndim, data.nparts, &grid);
#endif
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (box = 0; box < pdata.nboxes; box++)
      {
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructGridSetExtents(&grid, &part,
                                     pdata.ilowers[box], pdata.iuppers[box]);
#else
         NALU_HYPRE_SStructGridSetExtents(grid, part,
                                     pdata.ilowers[box], pdata.iuppers[box]);
#endif
      }

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridSetVariables(&grid, &part, &pdata.nvars, pdata.vartypes);
#else
      NALU_HYPRE_SStructGridSetVariables(grid, part, pdata.nvars, pdata.vartypes);
#endif

      /* GridAddVariabes */

      /* GridSetNeighborBox */
      for (box = 0; box < pdata.glue_nboxes; box++)
      {
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructGridSetNeighborBox(&grid, &part,
                                         pdata.glue_ilowers[box],
                                         pdata.glue_iuppers[box],
                                         &pdata.glue_nbor_parts[box],
                                         pdata.glue_nbor_ilowers[box],
                                         pdata.glue_nbor_iuppers[box],
                                         pdata.glue_index_maps[box]);
#else
         NALU_HYPRE_SStructGridSetNeighborBox(grid, part,
                                         pdata.glue_ilowers[box],
                                         pdata.glue_iuppers[box],
                                         pdata.glue_nbor_parts[box],
                                         pdata.glue_nbor_ilowers[box],
                                         pdata.glue_nbor_iuppers[box],
                                         pdata.glue_index_maps[box]);
#endif
      }

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructGridSetPeriodic(&grid, &part, pdata.periodic);
#else
      NALU_HYPRE_SStructGridSetPeriodic(grid, part, pdata.periodic);
#endif
   }

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGridAssemble(&grid);
#else
   NALU_HYPRE_SStructGridAssemble(grid);
#endif

   /*-----------------------------------------------------------
    * Set up the stencils
    *-----------------------------------------------------------*/

#ifdef NALU_HYPRE_FORTRAN
   stencils = nalu_hypre_CTAlloc(nalu_hypre_F90_Obj,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
#else
   stencils = nalu_hypre_CTAlloc(NALU_HYPRE_SStructStencil,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
#endif
   for (s = 0; s < data.nstencils; s++)
   {
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructStencilCreate(&data.ndim, &data.stencil_sizes[s],
                                 &stencils[s]);
#else
      NALU_HYPRE_SStructStencilCreate(data.ndim, data.stencil_sizes[s],
                                 &stencils[s]);
#endif
      for (i = 0; i < data.stencil_sizes[s]; i++)
      {
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructStencilSetEntry(&stencils[s], &i,
                                      data.stencil_offsets[s][i],
                                      &data.stencil_vars[s][i]);
#else
         NALU_HYPRE_SStructStencilSetEntry(stencils[s], i,
                                      data.stencil_offsets[s][i],
                                      data.stencil_vars[s][i]);
#endif
      }
   }

   /*-----------------------------------------------------------
    * Set up the graph
    *-----------------------------------------------------------*/

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGraphCreate(&temp_COMM, &grid, &graph);
   NALU_HYPRE_SStructGraphSetObjectType(&graph, &for_NALU_HYPRE_PARCSR);
#else
   NALU_HYPRE_SStructGraphCreate(nalu_hypre_MPI_COMM_WORLD, grid, &graph);
   NALU_HYPRE_SStructGraphSetObjectType(graph, NALU_HYPRE_PARCSR);
#endif

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencils */
      for (var = 0; var < pdata.nvars; var++)
      {
#ifdef NALU_HYPRE_FORTRAN
         NALU_HYPRE_SStructGraphSetStencil(&graph, &part, &var,
                                      &stencils[pdata.stencil_num[var]]);
#else
         NALU_HYPRE_SStructGraphSetStencil(graph, part, var,
                                      stencils[pdata.stencil_num[var]]);
#endif
      }

      /* add entries */
      for (entry = 0; entry < pdata.graph_nentries; entry++)
      {
         for (index[2] = pdata.graph_ilowers[entry][2];
              index[2] <= pdata.graph_iuppers[entry][2];
              index[2] += pdata.graph_strides[entry][2])
         {
            for (index[1] = pdata.graph_ilowers[entry][1];
                 index[1] <= pdata.graph_iuppers[entry][1];
                 index[1] += pdata.graph_strides[entry][1])
            {
               for (index[0] = pdata.graph_ilowers[entry][0];
                    index[0] <= pdata.graph_iuppers[entry][0];
                    index[0] += pdata.graph_strides[entry][0])
               {
                  for (i = 0; i < 3; i++)
                  {
                     j = pdata.graph_index_maps[entry][i];
                     k = index[i] - pdata.graph_ilowers[entry][i];
                     k /= pdata.graph_strides[entry][i];
                     k *= pdata.graph_index_signs[entry][i];
                     to_index[j] = pdata.graph_to_ilowers[entry][j] +
                                   k * pdata.graph_to_strides[entry][j];
                  }
#ifdef NALU_HYPRE_FORTRAN
                  NALU_HYPRE_SStructGraphAddEntries(&graph, &part, index,
                                               &pdata.graph_vars[entry],
                                               &pdata.graph_to_parts[entry],
                                               to_index,
                                               &pdata.graph_to_vars[entry]);
#else
                  NALU_HYPRE_SStructGraphAddEntries(graph, part, index,
                                               pdata.graph_vars[entry],
                                               pdata.graph_to_parts[entry],
                                               to_index,
                                               pdata.graph_to_vars[entry]);
#endif
               }
            }
         }
      }
   }

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGraphAssemble(&graph);
#else
   NALU_HYPRE_SStructGraphAssemble(graph);
#endif

   /*-----------------------------------------------------------
    * Set up the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_TAlloc(NALU_HYPRE_Real,  data.max_boxsize, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructMatrixCreate(&temp_COMM, &graph, &A);
#else
   NALU_HYPRE_SStructMatrixCreate(nalu_hypre_MPI_COMM_WORLD, graph, &A);
#endif

   /* TODO NALU_HYPRE_SStructMatrixSetSymmetric(A, 1); */
   for (entry = 0; entry < data.symmetric_nentries; entry++)
   {
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixSetSymmetric(&A,
                                      &data.symmetric_parts[entry],
                                      &data.symmetric_vars[entry],
                                      &data.symmetric_to_vars[entry],
                                      &data.symmetric_booleans[entry]);
#else
      NALU_HYPRE_SStructMatrixSetSymmetric(A,
                                      data.symmetric_parts[entry],
                                      data.symmetric_vars[entry],
                                      data.symmetric_to_vars[entry],
                                      data.symmetric_booleans[entry]);
#endif
   }

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructMatrixSetNSSymmetric(&A, &data.ns_symmetric);

   NALU_HYPRE_SStructMatrixSetObjectType(&A, &for_NALU_HYPRE_PARCSR);
   NALU_HYPRE_SStructMatrixInitialize(&A);
#else
   NALU_HYPRE_SStructMatrixSetNSSymmetric(A, data.ns_symmetric);

   NALU_HYPRE_SStructMatrixSetObjectType(A, NALU_HYPRE_PARCSR);
   NALU_HYPRE_SStructMatrixInitialize(A);
#endif

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      cell_grid =  nalu_hypre_SStructPGridCellSGrid(nalu_hypre_SStructGridPGrid(grid, part));
      bounding_box = nalu_hypre_StructGridBoundingBox(cell_grid);

      h = (NALU_HYPRE_Real) (nalu_hypre_BoxIMax(bounding_box)[0] - nalu_hypre_BoxIMin(bounding_box)[0]);
      for (i = 1; i < data.ndim; i++)
      {
         if ((nalu_hypre_BoxIMax(bounding_box)[i] - nalu_hypre_BoxIMin(bounding_box)[i]) > h)
         {
            h = (NALU_HYPRE_Real) (nalu_hypre_BoxIMax(bounding_box)[i] - nalu_hypre_BoxIMin(bounding_box)[i]);
         }
      }
      h = 1.0 / h;

      /* set stencil values */
      for (var = 0; var < pdata.nvars; var++)
      {
         s = pdata.stencil_num[var];
         for (i = 0; i < data.stencil_sizes[s]; i++)
         {
            for (j = 0; j < pdata.max_boxsize; j++)
            {
               values[j] = h * data.stencil_values[s][i];
            }
            if (i < 9)
            {
               for (j = 0; j < pdata.max_boxsize; j++)
               {
                  values[j] += data.stencil_values[s + data.ndim][i] / h;
               }
            }

            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
#ifdef NALU_HYPRE_FORTRAN
               NALU_HYPRE_SStructMatrixSetBoxValues(&A, &part, &ilower[0], &iupper[0],
                                               &var, &one, &i, &values[0]);
#else
               NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                               var, 1, &i, values);
#endif
            }
         }
      }

      /* set non-stencil entries */
      for (entry = 0; entry < pdata.graph_nentries; entry++)
      {
         for (index[2] = pdata.graph_ilowers[entry][2];
              index[2] <= pdata.graph_iuppers[entry][2];
              index[2] += pdata.graph_strides[entry][2])
         {
            for (index[1] = pdata.graph_ilowers[entry][1];
                 index[1] <= pdata.graph_iuppers[entry][1];
                 index[1] += pdata.graph_strides[entry][1])
            {
               for (index[0] = pdata.graph_ilowers[entry][0];
                    index[0] <= pdata.graph_iuppers[entry][0];
                    index[0] += pdata.graph_strides[entry][0])
               {
#ifdef NALU_HYPRE_FORTRAN
                  NALU_HYPRE_SStructMatrixSetValues(&A, &part, index,
                                               &pdata.graph_vars[entry],
                                               &one, &pdata.graph_entries[entry],
                                               &pdata.graph_values[entry]);
#else
                  NALU_HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.graph_vars[entry],
                                               1, &pdata.graph_entries[entry],
                                               &pdata.graph_values[entry]);
#endif
               }
            }
         }
      }
   }

   /* reset matrix values:
    *   NOTE THAT THE matrix_ilowers & matrix_iuppers MUST BE IN TERMS OF THE
    *   CHOOSEN VAR_TYPE INDICES, UNLIKE THE EXTENTS OF THE GRID< WHICH ARE
    *   IN TERMS OF THE CELL VARTYPE INDICES.
    */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (entry = 0; entry < pdata.matrix_nentries; entry++)
      {
         for (index[2] = pdata.matrix_ilowers[entry][2];
              index[2] <= pdata.matrix_iuppers[entry][2];
              index[2] += pdata.matrix_strides[entry][2])
         {
            for (index[1] = pdata.matrix_ilowers[entry][1];
                 index[1] <= pdata.matrix_iuppers[entry][1];
                 index[1] += pdata.matrix_strides[entry][1])
            {
               for (index[0] = pdata.matrix_ilowers[entry][0];
                    index[0] <= pdata.matrix_iuppers[entry][0];
                    index[0] += pdata.matrix_strides[entry][0])
               {
#ifdef NALU_HYPRE_FORTRAN
                  NALU_HYPRE_SStructMatrixSetValues(&A, &part, index,
                                               &pdata.matrix_vars[entry],
                                               &one, &pdata.matrix_entries[entry],
                                               &pdata.matrix_values[entry]);
#else
                  NALU_HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.matrix_vars[entry],
                                               1, &pdata.matrix_entries[entry],
                                               &pdata.matrix_values[entry]);
#endif
               }
            }
         }
      }
   }

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructMatrixAssemble(&A);
   NALU_HYPRE_MaxwellGrad(&grid, &T);
#else
   NALU_HYPRE_SStructMatrixAssemble(A);
   NALU_HYPRE_MaxwellGrad(grid, &T);
#endif

   /* eliminate the physical boundary points */
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructMatrixGetObject(&A, &parA);
   NALU_HYPRE_SStructMaxwellPhysBdy(&grid, &one, &data.rfactor[0],
                               &bdryRanks[0][0], &bdryRanksCnt[0]);

   NALU_HYPRE_ParCSRMatrixEliminateRowsCols(&parA, &bdryRanksCnt[0], &bdryRanks[0]);

   NALU_HYPRE_ParCSRMatrixEliminateRowsCols(&parA, &bdryRanksCnt[0], &bdryRanks[0]);
#else
   NALU_HYPRE_SStructMatrixGetObject(A, (void **) &parA);
   NALU_HYPRE_SStructMaxwellPhysBdy(&grid, 1, data.rfactor,
                               &bdryRanks, &bdryRanksCnt);

   NALU_HYPRE_ParCSRMatrixEliminateRowsCols(parA, bdryRanksCnt[0], bdryRanks[0]);

   NALU_HYPRE_ParCSRMatrixEliminateRowsCols(parA, bdryRanksCnt[0], bdryRanks[0]);
#endif

   {
      nalu_hypre_MaxwellOffProcRow **OffProcRows;
      nalu_hypre_SStructSharedDOF_ParcsrMatRowsComm(&grid,
                                               (nalu_hypre_ParCSRMatrix *) parA,
                                               &i,
                                               &OffProcRows);
      for (j = 0; j < i; j++)
      {
         nalu_hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[j]);
      }
      nalu_hypre_TFree(OffProcRows, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructVectorCreate(&temp_COMM, &grid, &b);
   NALU_HYPRE_SStructVectorSetObjectType(&b, &for_NALU_HYPRE_PARCSR);

   NALU_HYPRE_SStructVectorInitialize(&b);
#else
   NALU_HYPRE_SStructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, &b);
   NALU_HYPRE_SStructVectorSetObjectType(b, NALU_HYPRE_PARCSR);

   NALU_HYPRE_SStructVectorInitialize(b);
#endif

   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = nalu_hypre_sin((NALU_HYPRE_Real)(j + 1));
      values[j] = (NALU_HYPRE_Real) nalu_hypre_Rand();
      values[j] = (NALU_HYPRE_Real) j;
   }
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (var = 0; var < pdata.nvars; var++)
      {
         for (box = 0; box < pdata.nboxes; box++)
         {
            GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                           pdata.vartypes[var], ilower, iupper);
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&b, &part, &ilower[0], &iupper[0],
                                            &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
                                            var, values);
#endif
         }
      }
   }
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructVectorAssemble(&b);
#else
   NALU_HYPRE_SStructVectorAssemble(b);
#endif

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructVectorCreate(&temp_COMM, &grid, &x);
   NALU_HYPRE_SStructVectorSetObjectType(&x, &for_NALU_HYPRE_PARCSR);

   NALU_HYPRE_SStructVectorInitialize(&x);
#else
   NALU_HYPRE_SStructVectorCreate(nalu_hypre_MPI_COMM_WORLD, grid, x);
   NALU_HYPRE_SStructVectorSetObjectType(x, NALU_HYPRE_PARCSR);

   NALU_HYPRE_SStructVectorInitialize(x);
#endif

   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = 0.0;
   }
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (var = 0; var < pdata.nvars; var++)
      {
         for (box = 0; box < pdata.nboxes; box++)
         {
            GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                           pdata.vartypes[var], ilower, iupper);
#ifdef NALU_HYPRE_FORTRAN
            NALU_HYPRE_SStructVectorSetBoxValues(&x, &part, &ilower[0], &iupper[0],
                                            &var, &values[0]);
#else
            NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                            var, values);
#endif
         }
      }
   }
#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructVectorAssemble(&x);

   NALU_HYPRE_SStructVectorGetObject(&x, &parx);
   NALU_HYPRE_SStructVectorGetObject(&b, &parb);
#else
   NALU_HYPRE_SStructVectorAssemble(x);

   NALU_HYPRE_SStructVectorGetObject(x, (void **) &parx);
   NALU_HYPRE_SStructVectorGetObject(b, (void **) &parb);
#endif

   nalu_hypre_ParVectorZeroBCValues((nalu_hypre_ParVector *) parx, bdryRanks[0],
                               bdryRanksCnt[0]);
   nalu_hypre_ParVectorZeroBCValues((nalu_hypre_ParVector *) parb, bdryRanks[0],
                               bdryRanksCnt[0]);

   nalu_hypre_TFree(bdryRanks[0], NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(bdryRanks, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(bdryRanksCnt, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("SStruct Interface", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMatrixPrint("sstruct.out.A",  &A, &zero);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.b",  &b, &zero);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x0", &x, &zero);
#else
      NALU_HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
#endif
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   if (solver_id == 1)
   {
      time_index = nalu_hypre_InitializeTiming("Maxwell Setup");
      nalu_hypre_BeginTiming(time_index);

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMaxwellCreate(&long_temp_COMM, &solver);
      NALU_HYPRE_SStructMaxwellSetMaxIter(&solver, &twenty);
      NALU_HYPRE_SStructMaxwellSetTol(&solver, &ftol);
      NALU_HYPRE_SStructMaxwellSetRelChange(&solver, &zero);
      NALU_HYPRE_SStructMaxwellSetNumPreRelax(&solver, &one);
      NALU_HYPRE_SStructMaxwellSetNumPostRelax(&solver, &one);
      NALU_HYPRE_SStructMaxwellSetRfactors(&solver, &data.rfactor[0]);
      NALU_HYPRE_SStructMaxwellSetGrad(&solver, &T);
      /*NALU_HYPRE_SStructMaxwellSetConstantCoef(solver, 1);*/
      NALU_HYPRE_SStructMaxwellSetPrintLevel(&solver, &one);
      NALU_HYPRE_SStructMaxwellSetLogging(&solver, &one);
      NALU_HYPRE_SStructMaxwellSetup(&solver, &A, &b, &x);
#else
      NALU_HYPRE_SStructMaxwellCreate(nalu_hypre_MPI_COMM_WORLD, &solver);
      NALU_HYPRE_SStructMaxwellSetMaxIter(solver, 20);
      NALU_HYPRE_SStructMaxwellSetTol(solver, 1.0e-8);
      NALU_HYPRE_SStructMaxwellSetRelChange(solver, 0);
      NALU_HYPRE_SStructMaxwellSetNumPreRelax(solver, 1);
      NALU_HYPRE_SStructMaxwellSetNumPostRelax(solver, 1);
      NALU_HYPRE_SStructMaxwellSetRfactors(solver, data.rfactor);
      NALU_HYPRE_SStructMaxwellSetGrad(solver, T);
      /*NALU_HYPRE_SStructMaxwellSetConstantCoef(solver, 1);*/
      NALU_HYPRE_SStructMaxwellSetPrintLevel(solver, 1);
      NALU_HYPRE_SStructMaxwellSetLogging(solver, 1);
      NALU_HYPRE_SStructMaxwellSetup(solver, A, b, x);
#endif

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("Maxwell Solve");
      nalu_hypre_BeginTiming(time_index);

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMaxwellSolve(&solver, &A, &b, &x);
#else
      NALU_HYPRE_SStructMaxwellSolve(solver, A, b, x);
#endif

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructMaxwellGetNumIterations(&solver, &num_iterations);
      NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(&solver, &final_res_norm);
      NALU_HYPRE_SStructMaxwellDestroy(&solver);
#else
      NALU_HYPRE_SStructMaxwellGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_SStructMaxwellGetFinalRelativeResidualNorm(
         solver, &final_res_norm);
      NALU_HYPRE_SStructMaxwellDestroy(solver);
#endif
   }

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructVectorGather(&x);
#else
   NALU_HYPRE_SStructVectorGather(x);
#endif

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x", &x, &zero);
#else
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
#endif
   }

   if (myid == 0)
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Iterations = %d\n", num_iterations);
      nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      nalu_hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGridDestroy(&grid);
#else
   NALU_HYPRE_SStructGridDestroy(grid);
#endif

   for (s = 0; s < data.nstencils; s++)
   {
#ifdef NALU_HYPRE_FORTRAN
      NALU_HYPRE_SStructStencilDestroy(&stencils[s]);
#else
      NALU_HYPRE_SStructStencilDestroy(stencils[s]);
#endif
   }
   nalu_hypre_TFree(stencils, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_FORTRAN
   NALU_HYPRE_SStructGraphDestroy(&graph);
   NALU_HYPRE_SStructMatrixDestroy(&A);
   NALU_HYPRE_ParCSRMatrixDestroy(&T);
   NALU_HYPRE_SStructVectorDestroy(&b);
   NALU_HYPRE_SStructVectorDestroy(&x);
#else
   NALU_HYPRE_SStructGraphDestroy(graph);
   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_ParCSRMatrixDestroy(T);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);
#endif


   DestroyData(data);

   nalu_hypre_TFree(parts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(refine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(distribute, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(block, NALU_HYPRE_MEMORY_HOST);

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

   return (0);
}
