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
#include "NALU_HYPRE_struct_ls.h"
#include "NALU_HYPRE_krylov.h"
#include "_nalu_hypre_sstruct_mv.h"
//#include "_nalu_hypre_struct_mv.hpp"

/* begin lobpcg */

#include <time.h>

#include "NALU_HYPRE_lobpcg.h"

#define NO_SOLVER -9198

/* end lobpcg */

#define DEBUG 0

#define SECOND_TIME 0

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct.in.default";

typedef NALU_HYPRE_Int Index[3];

/*------------------------------------------------------------
 * ProblemIndex:
 *
 * The index has extra information stored in entries 3-8 that
 * determine how the index gets "mapped" to finer index spaces.
 *
 * NOTE: For implementation convenience, the index is "pre-shifted"
 * according to the values in entries 6,7,8.  The following discussion
 * describes how "un-shifted" indexes are mapped, because that is a
 * more natural way to think about this mapping problem, and because
 * that is the convention used in the input file for this code.  The
 * reason that pre-shifting is convenient is because it makes the true
 * value of the index on the unrefined index space readily available
 * in entries 0-2, hence, all operations on that unrefined space are
 * straightforward.  Also, the only time that the extra mapping
 * information is needed is when an index is mapped to a new refined
 * index space, allowing us to isolate the mapping details to the
 * routine MapProblemIndex.  The only other effected routine is
 * SScanProblemIndex, which takes the user input and pre-shifts it.
 *
 * - Entries 3,4,5 have values of either 0 or 1 that indicate
 *   whether to map an index "to the left" or "to the right".
 *   Here is a 1D diagram:
 *
 *    --  |     *     |    unrefined index space
 *   |
 *    --> | * | . | * |    refined index space (factor = 3)
 *          0       1
 *
 *   The '*' index on the unrefined index space gets mapped to one of
 *   the '*' indexes on the refined space based on the value (0 or 1)
 *   of the relevent entry (3,4, or 5).  The actual mapping formula is
 *   as follows (with refinement factor, r):
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3]
 *
 * - Entries 6,7,8 contain "shift" information.  The shift is
 *   simply added to the mapped index just described.  So, the
 *   complete mapping formula is as follows:
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3] + index[i+6]
 *
 *------------------------------------------------------------*/

typedef NALU_HYPRE_Int ProblemIndex[9];

typedef struct
{
   /* for GridSetExtents */
   NALU_HYPRE_Int              nboxes;
   ProblemIndex          *ilowers;
   ProblemIndex          *iuppers;
   NALU_HYPRE_Int             *boxsizes;
   NALU_HYPRE_Int              max_boxsize;

   /* for GridSetVariables */
   NALU_HYPRE_Int              nvars;
   NALU_HYPRE_SStructVariable *vartypes;

   /* for GridAddVariables */
   NALU_HYPRE_Int              add_nvars;
   ProblemIndex          *add_indexes;
   NALU_HYPRE_SStructVariable *add_vartypes;

   /* for GridSetNeighborPart and GridSetSharedPart */
   NALU_HYPRE_Int              glue_nboxes;
   NALU_HYPRE_Int             *glue_shared;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   Index                 *glue_offsets;
   NALU_HYPRE_Int             *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_nbor_offsets;
   Index                 *glue_index_maps;
   Index                 *glue_index_dirs;
   NALU_HYPRE_Int             *glue_primaries;

   /* for GraphSetStencil */
   NALU_HYPRE_Int             *stencil_num;

   /* for GraphAddEntries */
   NALU_HYPRE_Int              graph_nboxes;
   ProblemIndex          *graph_ilowers;
   ProblemIndex          *graph_iuppers;
   Index                 *graph_strides;
   NALU_HYPRE_Int             *graph_vars;
   NALU_HYPRE_Int             *graph_to_parts;
   ProblemIndex          *graph_to_ilowers;
   ProblemIndex          *graph_to_iuppers;
   Index                 *graph_to_strides;
   NALU_HYPRE_Int             *graph_to_vars;
   Index                 *graph_index_maps;
   Index                 *graph_index_signs;
   NALU_HYPRE_Int             *graph_entries;
   NALU_HYPRE_Int              graph_values_size;
   NALU_HYPRE_Real            *graph_values;
   NALU_HYPRE_Real            *d_graph_values;
   NALU_HYPRE_Int             *graph_boxsizes;

   /* MatrixSetValues */
   NALU_HYPRE_Int              matset_nboxes;
   ProblemIndex          *matset_ilowers;
   ProblemIndex          *matset_iuppers;
   Index                 *matset_strides;
   NALU_HYPRE_Int             *matset_vars;
   NALU_HYPRE_Int             *matset_entries;
   NALU_HYPRE_Real            *matset_values;

   /* MatrixAddToValues */
   NALU_HYPRE_Int              matadd_nboxes;
   ProblemIndex          *matadd_ilowers;
   ProblemIndex          *matadd_iuppers;
   NALU_HYPRE_Int             *matadd_vars;
   NALU_HYPRE_Int             *matadd_nentries;
   NALU_HYPRE_Int            **matadd_entries;
   NALU_HYPRE_Real           **matadd_values;

   /* FEMMatrixAddToValues */
   NALU_HYPRE_Int              fem_matadd_nboxes;
   ProblemIndex          *fem_matadd_ilowers;
   ProblemIndex          *fem_matadd_iuppers;
   NALU_HYPRE_Int             *fem_matadd_nrows;
   NALU_HYPRE_Int            **fem_matadd_rows;
   NALU_HYPRE_Int             *fem_matadd_ncols;
   NALU_HYPRE_Int            **fem_matadd_cols;
   NALU_HYPRE_Real           **fem_matadd_values;

   /* RhsAddToValues */
   NALU_HYPRE_Int              rhsadd_nboxes;
   ProblemIndex          *rhsadd_ilowers;
   ProblemIndex          *rhsadd_iuppers;
   NALU_HYPRE_Int             *rhsadd_vars;
   NALU_HYPRE_Real            *rhsadd_values;

   /* FEMRhsAddToValues */
   NALU_HYPRE_Int              fem_rhsadd_nboxes;
   ProblemIndex          *fem_rhsadd_ilowers;
   ProblemIndex          *fem_rhsadd_iuppers;
   NALU_HYPRE_Real           **fem_rhsadd_values;

   Index                  periodic;

} ProblemPartData;

typedef struct
{
   NALU_HYPRE_Int        ndim;
   NALU_HYPRE_Int        nparts;
   ProblemPartData *pdata;
   NALU_HYPRE_Int        max_boxsize;

   NALU_HYPRE_MemoryLocation memory_location;

   /* for GridSetNumGhost */
   NALU_HYPRE_Int       *numghost;

   NALU_HYPRE_Int        nstencils;
   NALU_HYPRE_Int       *stencil_sizes;
   Index          **stencil_offsets;
   NALU_HYPRE_Int      **stencil_vars;
   NALU_HYPRE_Real     **stencil_values;

   NALU_HYPRE_Int        rhs_true;
   NALU_HYPRE_Real       rhs_value;

   NALU_HYPRE_Int        fem_nvars;
   Index           *fem_offsets;
   NALU_HYPRE_Int       *fem_vars;
   NALU_HYPRE_Real     **fem_values_full;
   NALU_HYPRE_Int      **fem_ivalues_full;
   NALU_HYPRE_Int       *fem_ordering; /* same info as vars/offsets */
   NALU_HYPRE_Int        fem_nsparse;  /* number of nonzeros in values_full */
   NALU_HYPRE_Int       *fem_sparsity; /* nonzeros in values_full */
   NALU_HYPRE_Real      *fem_values;   /* nonzero values in values_full */
   NALU_HYPRE_Real      *d_fem_values;

   NALU_HYPRE_Int        fem_rhs_true;
   NALU_HYPRE_Real      *fem_rhs_values;
   NALU_HYPRE_Real      *d_fem_rhs_values;

   NALU_HYPRE_Int        symmetric_num;
   NALU_HYPRE_Int       *symmetric_parts;
   NALU_HYPRE_Int       *symmetric_vars;
   NALU_HYPRE_Int       *symmetric_to_vars;
   NALU_HYPRE_Int       *symmetric_booleans;

   NALU_HYPRE_Int        ns_symmetric;

   NALU_HYPRE_Int        npools;
   NALU_HYPRE_Int       *pools;   /* array of size nparts */
   NALU_HYPRE_Int        ndists;  /* number of (pool) distributions */
   NALU_HYPRE_Int       *dist_npools;
   NALU_HYPRE_Int      **dist_pools;

} ProblemData;

/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
GetVariableBox( Index  cell_ilower,
                Index  cell_iupper,
                NALU_HYPRE_Int    vartype,
                Index  var_ilower,
                Index  var_iupper )
{
   NALU_HYPRE_Int ierr = 0;

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
SScanDblArray( char   *sdata_ptr,
               char  **sdata_ptr_ptr,
               NALU_HYPRE_Int     size,
               NALU_HYPRE_Real *array )
{
   NALU_HYPRE_Int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

NALU_HYPRE_Int
SScanProblemIndex( char          *sdata_ptr,
                   char         **sdata_ptr_ptr,
                   NALU_HYPRE_Int      ndim,
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

   NALU_HYPRE_Int          myid;
   FILE              *file;

   char              *sdata = NULL;
   char              *sdata_line;
   char              *sdata_ptr;
   NALU_HYPRE_Int          sdata_size;
   NALU_HYPRE_Int          size;
   NALU_HYPRE_Int          memchunk = 10000;
   NALU_HYPRE_Int          maxline  = 250;

   char               key[250];
   NALU_HYPRE_Int          part, var, s, entry, i, j, k, il, iu;

   NALU_HYPRE_MemoryLocation memory_location = data_ptr -> memory_location;

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

   data.memory_location = memory_location;
   data.max_boxsize = 0;
   data.numghost = NULL;
   data.nstencils = 0;
   data.rhs_true = 0;
   data.fem_nvars = 0;
   data.fem_nsparse = 0;
   data.fem_rhs_true = 0;
   data.symmetric_num = 0;
   data.symmetric_parts    = NULL;
   data.symmetric_vars     = NULL;
   data.symmetric_to_vars  = NULL;
   data.symmetric_booleans = NULL;
   data.ns_symmetric = 0;
   data.ndists = 0;
   data.dist_npools = NULL;
   data.dist_pools  = NULL;

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
         else if ( strcmp(key, "GridSetNumGhost:") == 0 )
         {
            // # GridSetNumGhost: numghost[2*ndim]
            // GridSetNumGhost: [3 3 3 3]
            data.numghost = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2 * data.ndim, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, 2 * data.ndim, data.numghost);
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
            pdata.vartypes = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  pdata.nvars, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, pdata.nvars, pdata.vartypes);
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridAddVariables:") == 0 )
         {
            /* TODO */
            nalu_hypre_printf("GridAddVariables not yet implemented!\n");
            exit(1);
         }
         else if ( strcmp(key, "GridSetNeighborPart:") == 0 ||
                   strcmp(key, "GridSetSharedPart:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.glue_nboxes % 10) == 0)
            {
               size = pdata.glue_nboxes + 10;
               pdata.glue_shared =
                  nalu_hypre_TReAlloc(pdata.glue_shared,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_ilowers =
                  nalu_hypre_TReAlloc(pdata.glue_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_iuppers =
                  nalu_hypre_TReAlloc(pdata.glue_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_offsets =
                  nalu_hypre_TReAlloc(pdata.glue_offsets,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_parts =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_parts,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_ilowers =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_iuppers =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_nbor_offsets =
                  nalu_hypre_TReAlloc(pdata.glue_nbor_offsets,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_index_maps =
                  nalu_hypre_TReAlloc(pdata.glue_index_maps,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_index_dirs =
                  nalu_hypre_TReAlloc(pdata.glue_index_dirs,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.glue_primaries =
                  nalu_hypre_TReAlloc(pdata.glue_primaries,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            pdata.glue_shared[pdata.glue_nboxes] = 0;
            if ( strcmp(key, "GridSetSharedPart:") == 0 )
            {
               pdata.glue_shared[pdata.glue_nboxes] = 1;
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_iuppers[pdata.glue_nboxes]);
            if (pdata.glue_shared[pdata.glue_nboxes])
            {
               SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                             pdata.glue_offsets[pdata.glue_nboxes]);
            }
            pdata.glue_nbor_parts[pdata.glue_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_iuppers[pdata.glue_nboxes]);
            if (pdata.glue_shared[pdata.glue_nboxes])
            {
               SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                             pdata.glue_nbor_offsets[pdata.glue_nboxes]);
            }
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_maps[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_maps[pdata.glue_nboxes][i] = i;
            }
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_dirs[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_dirs[pdata.glue_nboxes][i] = 1;
            }
            sdata_ptr += strcspn(sdata_ptr, ":\t\n");
            if ( *sdata_ptr == ':' )
            {
               /* read in optional primary indicator */
               sdata_ptr += 1;
               pdata.glue_primaries[pdata.glue_nboxes] =
                  strtol(sdata_ptr, &sdata_ptr, 10);
            }
            else
            {
               pdata.glue_primaries[pdata.glue_nboxes] = -1;
               sdata_ptr -= 1;
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
            if (data.fem_nvars > 0)
            {
               nalu_hypre_printf("Stencil and FEMStencil cannot be used together\n");
               exit(1);
            }
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
         else if ( strcmp(key, "RhsSet:") == 0 )
         {
            if (data.rhs_true == 0)
            {
               data.rhs_true = 1;
            }
            data.rhs_value = (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "FEMStencilCreate:") == 0 )
         {
            if (data.nstencils > 0)
            {
               nalu_hypre_printf("Stencil and FEMStencil cannot be used together\n");
               exit(1);
            }
            data.fem_nvars = strtol(sdata_ptr, &sdata_ptr, 10);
            data.fem_offsets = nalu_hypre_CTAlloc(Index,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
            data.fem_vars = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
            data.fem_values_full = nalu_hypre_CTAlloc(NALU_HYPRE_Real *,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < data.fem_nvars; i++)
            {
               data.fem_values_full[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
            }
         }
         else if ( strcmp(key, "FEMStencilSetRow:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.ndim, data.fem_offsets[i]);
            for (k = data.ndim; k < 3; k++)
            {
               data.fem_offsets[i][k] = 0;
            }
            data.fem_vars[i] = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanDblArray(sdata_ptr, &sdata_ptr,
                          data.fem_nvars, data.fem_values_full[i]);
         }
         else if ( strcmp(key, "FEMRhsSet:") == 0 )
         {
            if (data.fem_rhs_true == 0)
            {
               data.fem_rhs_true = 1;
               data.fem_rhs_values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
               data.d_fem_rhs_values = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data.fem_nvars, memory_location);
            }
            SScanDblArray(sdata_ptr, &sdata_ptr,
                          data.fem_nvars, data.fem_rhs_values);
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
            if ((pdata.graph_nboxes % 10) == 0)
            {
               size = pdata.graph_nboxes + 10;
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
               pdata.d_graph_values =
                  nalu_hypre_TReAlloc_v2(pdata.d_graph_values, NALU_HYPRE_Real, pdata.graph_values_size,
                                    NALU_HYPRE_Real, size, memory_location);
               pdata.graph_values_size = size;
               pdata.graph_boxsizes =
                  nalu_hypre_TReAlloc(pdata.graph_boxsizes,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_to_parts[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_to_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_to_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_to_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_index_maps[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_index_maps[pdata.graph_nboxes][i] = i;
            }
            for (i = 0; i < 3; i++)
            {
               pdata.graph_index_signs[pdata.graph_nboxes][i] = 1;
               if ( pdata.graph_to_iuppers[pdata.graph_nboxes][i] <
                    pdata.graph_to_ilowers[pdata.graph_nboxes][i] )
               {
                  pdata.graph_index_signs[pdata.graph_nboxes][i] = -1;
               }
            }
            pdata.graph_entries[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_values[pdata.graph_nboxes] =
               (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.graph_boxsizes[pdata.graph_nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[pdata.graph_nboxes] *=
                  (pdata.graph_iuppers[pdata.graph_nboxes][i] -
                   pdata.graph_ilowers[pdata.graph_nboxes][i] + 1);
            }
            pdata.graph_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            if ((data.symmetric_num % 10) == 0)
            {
               size = data.symmetric_num + 10;
               data.symmetric_parts =
                  nalu_hypre_TReAlloc(data.symmetric_parts,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_vars =
                  nalu_hypre_TReAlloc(data.symmetric_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_to_vars =
                  nalu_hypre_TReAlloc(data.symmetric_to_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               data.symmetric_booleans =
                  nalu_hypre_TReAlloc(data.symmetric_booleans,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
            }
            data.symmetric_parts[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_to_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_booleans[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_num++;
         }
         else if ( strcmp(key, "MatrixSetNSSymmetric:") == 0 )
         {
            data.ns_symmetric = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "MatrixSetValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matset_nboxes % 10) == 0)
            {
               size = pdata.matset_nboxes + 10;
               pdata.matset_ilowers =
                  nalu_hypre_TReAlloc(pdata.matset_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matset_iuppers =
                  nalu_hypre_TReAlloc(pdata.matset_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matset_strides =
                  nalu_hypre_TReAlloc(pdata.matset_strides,  Index,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matset_vars =
                  nalu_hypre_TReAlloc(pdata.matset_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matset_entries =
                  nalu_hypre_TReAlloc(pdata.matset_entries,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matset_values =
                  nalu_hypre_TReAlloc(pdata.matset_values,  NALU_HYPRE_Real,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_ilowers[pdata.matset_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_iuppers[pdata.matset_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.matset_strides[pdata.matset_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.matset_strides[pdata.matset_nboxes][i] = 1;
            }
            pdata.matset_vars[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_entries[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_values[pdata.matset_nboxes] =
               (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.matset_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matadd_nboxes % 10) == 0)
            {
               size = pdata.matadd_nboxes + 10;
               pdata.matadd_ilowers =
                  nalu_hypre_TReAlloc(pdata.matadd_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matadd_iuppers =
                  nalu_hypre_TReAlloc(pdata.matadd_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matadd_vars =
                  nalu_hypre_TReAlloc(pdata.matadd_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matadd_nentries =
                  nalu_hypre_TReAlloc(pdata.matadd_nentries,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matadd_entries =
                  nalu_hypre_TReAlloc(pdata.matadd_entries,  NALU_HYPRE_Int *,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.matadd_values =
                  nalu_hypre_TReAlloc(pdata.matadd_values,  NALU_HYPRE_Real *,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_ilowers[pdata.matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matadd_iuppers[pdata.matadd_nboxes]);
            pdata.matadd_vars[pdata.matadd_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matadd_nentries[pdata.matadd_nboxes] = i;
            pdata.matadd_entries[pdata.matadd_nboxes] =
               nalu_hypre_TAlloc(NALU_HYPRE_Int,  i, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (NALU_HYPRE_Int*) pdata.matadd_entries[pdata.matadd_nboxes]);
            pdata.matadd_values[pdata.matadd_nboxes] =
               nalu_hypre_TAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, i,
                          (NALU_HYPRE_Real *) pdata.matadd_values[pdata.matadd_nboxes]);
            pdata.matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMMatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_matadd_nboxes % 10) == 0)
            {
               size = pdata.fem_matadd_nboxes + 10;
               pdata.fem_matadd_ilowers =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_iuppers =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_nrows =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_nrows,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_rows =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_rows,  NALU_HYPRE_Int *,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_ncols =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_ncols,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_cols =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_cols,  NALU_HYPRE_Int *,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_matadd_values =
                  nalu_hypre_TReAlloc(pdata.fem_matadd_values,  NALU_HYPRE_Real *,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_ilowers[pdata.fem_matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_matadd_iuppers[pdata.fem_matadd_nboxes]);
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_nrows[pdata.fem_matadd_nboxes] = i;
            pdata.fem_matadd_rows[pdata.fem_matadd_nboxes] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  i, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
                          (NALU_HYPRE_Int*) pdata.fem_matadd_rows[pdata.fem_matadd_nboxes]);
            j = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.fem_matadd_ncols[pdata.fem_matadd_nboxes] = j;
            pdata.fem_matadd_cols[pdata.fem_matadd_nboxes] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  j, NALU_HYPRE_MEMORY_HOST);
            SScanIntArray(sdata_ptr, &sdata_ptr, j,
                          (NALU_HYPRE_Int*) pdata.fem_matadd_cols[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_values[pdata.fem_matadd_nboxes] =
               nalu_hypre_TAlloc(NALU_HYPRE_Real,  i * j, NALU_HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, i * j,
                          (NALU_HYPRE_Real *) pdata.fem_matadd_values[pdata.fem_matadd_nboxes]);
            pdata.fem_matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "RhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.rhsadd_nboxes % 10) == 0)
            {
               size = pdata.rhsadd_nboxes + 10;
               pdata.rhsadd_ilowers =
                  nalu_hypre_TReAlloc(pdata.rhsadd_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.rhsadd_iuppers =
                  nalu_hypre_TReAlloc(pdata.rhsadd_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.rhsadd_vars =
                  nalu_hypre_TReAlloc(pdata.rhsadd_vars,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.rhsadd_values =
                  nalu_hypre_TReAlloc(pdata.rhsadd_values,  NALU_HYPRE_Real,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_ilowers[pdata.rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.rhsadd_iuppers[pdata.rhsadd_nboxes]);
            pdata.rhsadd_vars[pdata.rhsadd_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.rhsadd_values[pdata.rhsadd_nboxes] =
               (NALU_HYPRE_Real)strtod(sdata_ptr, &sdata_ptr);
            pdata.rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "FEMRhsAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.fem_rhsadd_nboxes % 10) == 0)
            {
               size = pdata.fem_rhsadd_nboxes + 10;
               pdata.fem_rhsadd_ilowers =
                  nalu_hypre_TReAlloc(pdata.fem_rhsadd_ilowers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_rhsadd_iuppers =
                  nalu_hypre_TReAlloc(pdata.fem_rhsadd_iuppers,  ProblemIndex,  size, NALU_HYPRE_MEMORY_HOST);
               pdata.fem_rhsadd_values =
                  nalu_hypre_TReAlloc(pdata.fem_rhsadd_values,  NALU_HYPRE_Real *,  size, NALU_HYPRE_MEMORY_HOST);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_ilowers[pdata.fem_rhsadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.fem_rhsadd_iuppers[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes] =
               nalu_hypre_TAlloc(NALU_HYPRE_Real,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
            SScanDblArray(sdata_ptr, &sdata_ptr, data.fem_nvars,
                          (NALU_HYPRE_Real *) pdata.fem_rhsadd_values[pdata.fem_rhsadd_nboxes]);
            pdata.fem_rhsadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.ndists++;
            data.dist_npools = nalu_hypre_TReAlloc(data.dist_npools,  NALU_HYPRE_Int,  data.ndists, NALU_HYPRE_MEMORY_HOST);
            data.dist_pools = nalu_hypre_TReAlloc(data.dist_pools,  NALU_HYPRE_Int *,  data.ndists, NALU_HYPRE_MEMORY_HOST);
            data.dist_npools[data.ndists - 1] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists - 1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.nparts, NALU_HYPRE_MEMORY_HOST);
#if 0
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.nparts, NALU_HYPRE_MEMORY_HOST);
#endif
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists - 1][part] = i;
         }
         else if ( strcmp(key, "GridSetNeighborBox:") == 0 )
         {
            nalu_hypre_printf("Error: No longer supporting SetNeighborBox\n");
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

   /* build additional FEM information */
   if (data.fem_nvars > 0)
   {
      NALU_HYPRE_Int d;

      data.fem_ivalues_full = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
      data.fem_ordering = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (1 + data.ndim) * data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
      data.fem_sparsity = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 2 * data.fem_nvars * data.fem_nvars,
                                        NALU_HYPRE_MEMORY_HOST);
      data.fem_values   = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data.fem_nvars * data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
      data.d_fem_values = nalu_hypre_TAlloc(NALU_HYPRE_Real, data.fem_nvars * data.fem_nvars, memory_location);

      for (i = 0; i < data.fem_nvars; i++)
      {
         data.fem_ivalues_full[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  data.fem_nvars, NALU_HYPRE_MEMORY_HOST);
         k = (1 + data.ndim) * i;
         data.fem_ordering[k] = data.fem_vars[i];
         for (d = 0; d < data.ndim; d++)
         {
            data.fem_ordering[k + 1 + d] = data.fem_offsets[i][d];
         }
         for (j = 0; j < data.fem_nvars; j++)
         {
            if (data.fem_values_full[i][j] != 0.0)
            {
               k = 2 * data.fem_nsparse;
               data.fem_sparsity[k]   = i;
               data.fem_sparsity[k + 1] = j;
               data.fem_values[data.fem_nsparse] = data.fem_values_full[i][j];
               data.fem_ivalues_full[i][j] = data.fem_nsparse;
               data.fem_nsparse ++;
            }
         }
      }
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
                NALU_HYPRE_Int     pooldist,
                Index        *refine,
                Index        *distribute,
                Index        *block,
                NALU_HYPRE_Int     num_procs,
                NALU_HYPRE_Int     myid,
                ProblemData  *data_ptr )
{
   NALU_HYPRE_MemoryLocation memory_location = global_data.memory_location;
   ProblemData      data = global_data;
   ProblemPartData  pdata;
   NALU_HYPRE_Int       *pool_procs;
   NALU_HYPRE_Int        np, pid;
   NALU_HYPRE_Int        pool, part, box, b, p, q, r, i, d;
   NALU_HYPRE_Int        dmap, sign, size;
   NALU_HYPRE_Int       *iptr;
   NALU_HYPRE_Real      *dptr;
   Index            m, mmap, n;
   ProblemIndex     ilower, iupper, int_ilower, int_iupper;

   /* set default pool distribution */
   data.npools = data.dist_npools[pooldist];
   data.pools  = data.dist_pools[pooldist];

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
      nalu_hypre_printf("%d,  %d \n", pool_procs[data.npools], num_procs);
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
#if 1 /* set this to 0 to make all of the SetSharedPart calls */
         pdata.glue_nboxes = 0;
#endif
         pdata.graph_nboxes = 0;
         pdata.matset_nboxes = 0;
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.matadd_entries[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.matadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         pdata.matadd_nboxes = 0;
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.fem_matadd_rows[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.fem_matadd_cols[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.fem_matadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         pdata.fem_matadd_nboxes = 0;
         pdata.rhsadd_nboxes = 0;
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.fem_rhsadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         pdata.fem_rhsadd_nboxes = 0;
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

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);
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
            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.graph_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.graph_ilowers[box],
                                        pdata.graph_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[box][d];
                        sign = pdata.graph_index_signs[box][d];
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_ilower[d] - pdata.graph_ilowers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_iuppers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_iupper[d] - pdata.graph_iuppers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_ilowers[i][d] = int_ilower[d];
                        pdata.graph_iuppers[i][d] = int_iupper[d];
                        pdata.graph_strides[i][d] =
                           pdata.graph_strides[box][d];
                        pdata.graph_to_strides[i][d] =
                           pdata.graph_to_strides[box][d];
                        pdata.graph_index_maps[i][d]  = dmap;
                        pdata.graph_index_signs[i][d] = sign;
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.graph_ilowers[i][d] =
                           pdata.graph_ilowers[box][d];
                        pdata.graph_iuppers[i][d] =
                           pdata.graph_iuppers[box][d];
                        pdata.graph_to_ilowers[i][d] =
                           pdata.graph_to_ilowers[box][d];
                        pdata.graph_to_iuppers[i][d] =
                           pdata.graph_to_iuppers[box][d];
                     }
                     pdata.graph_vars[i]     = pdata.graph_vars[box];
                     pdata.graph_to_parts[i] = pdata.graph_to_parts[box];
                     pdata.graph_to_vars[i]  = pdata.graph_to_vars[box];
                     pdata.graph_entries[i]  = pdata.graph_entries[box];
                     pdata.graph_values[i]   = pdata.graph_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.graph_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matset_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matset_ilowers[box],
                                        pdata.matset_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matset_ilowers[i][d] = int_ilower[d];
                        pdata.matset_iuppers[i][d] = int_iupper[d];
                        pdata.matset_strides[i][d] =
                           pdata.matset_strides[box][d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matset_ilowers[i][d] =
                           pdata.matset_ilowers[box][d];
                        pdata.matset_iuppers[i][d] =
                           pdata.matset_iuppers[box][d];
                     }
                     pdata.matset_vars[i]     = pdata.matset_vars[box];
                     pdata.matset_entries[i]  = pdata.matset_entries[box];
                     pdata.matset_values[i]   = pdata.matset_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.matset_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matadd_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matadd_ilowers[box],
                                        pdata.matadd_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matadd_ilowers[i][d] = int_ilower[d];
                        pdata.matadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matadd_ilowers[i][d] =
                           pdata.matadd_ilowers[box][d];
                        pdata.matadd_iuppers[i][d] =
                           pdata.matadd_iuppers[box][d];
                     }
                     pdata.matadd_vars[i]     = pdata.matadd_vars[box];
                     pdata.matadd_nentries[i] = pdata.matadd_nentries[box];
                     iptr = pdata.matadd_entries[i];
                     pdata.matadd_entries[i] = pdata.matadd_entries[box];
                     pdata.matadd_entries[box] = iptr;
                     dptr = pdata.matadd_values[i];
                     pdata.matadd_values[i] = pdata.matadd_values[box];
                     pdata.matadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.matadd_nboxes; box++)
            {
               nalu_hypre_TFree(pdata.matadd_entries[box], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(pdata.matadd_values[box], NALU_HYPRE_MEMORY_HOST);
            }
            pdata.matadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* fe is cell-based, so no need to convert box extents */
                  size = IntersectBoxes(pdata.fem_matadd_ilowers[box],
                                        pdata.fem_matadd_iuppers[box],
                                        pdata.ilowers[b], pdata.iuppers[b],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.fem_matadd_ilowers[i][d] = int_ilower[d];
                        pdata.fem_matadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.fem_matadd_ilowers[i][d] =
                           pdata.fem_matadd_ilowers[box][d];
                        pdata.fem_matadd_iuppers[i][d] =
                           pdata.fem_matadd_iuppers[box][d];
                     }
                     pdata.fem_matadd_nrows[i]  = pdata.fem_matadd_nrows[box];
                     iptr = pdata.fem_matadd_rows[box];
                     iptr = pdata.fem_matadd_rows[i];
                     pdata.fem_matadd_rows[i] = pdata.fem_matadd_rows[box];
                     pdata.fem_matadd_rows[box] = iptr;
                     pdata.fem_matadd_ncols[i]  = pdata.fem_matadd_ncols[box];
                     iptr = pdata.fem_matadd_cols[i];
                     pdata.fem_matadd_cols[i] = pdata.fem_matadd_cols[box];
                     pdata.fem_matadd_cols[box] = iptr;
                     dptr = pdata.fem_matadd_values[i];
                     pdata.fem_matadd_values[i] = pdata.fem_matadd_values[box];
                     pdata.fem_matadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.fem_matadd_nboxes; box++)
            {
               nalu_hypre_TFree(pdata.fem_matadd_rows[box], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(pdata.fem_matadd_cols[box], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(pdata.fem_matadd_values[box], NALU_HYPRE_MEMORY_HOST);
            }
            pdata.fem_matadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.rhsadd_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.rhsadd_ilowers[box],
                                        pdata.rhsadd_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.rhsadd_ilowers[i][d] = int_ilower[d];
                        pdata.rhsadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.rhsadd_ilowers[i][d] =
                           pdata.rhsadd_ilowers[box][d];
                        pdata.rhsadd_iuppers[i][d] =
                           pdata.rhsadd_iuppers[box][d];
                     }
                     pdata.rhsadd_vars[i]   = pdata.rhsadd_vars[box];
                     pdata.rhsadd_values[i] = pdata.rhsadd_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.rhsadd_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* fe is cell-based, so no need to convert box extents */
                  size = IntersectBoxes(pdata.fem_rhsadd_ilowers[box],
                                        pdata.fem_rhsadd_iuppers[box],
                                        pdata.ilowers[b], pdata.iuppers[b],
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.fem_rhsadd_ilowers[i][d] = int_ilower[d];
                        pdata.fem_rhsadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.fem_rhsadd_ilowers[i][d] =
                           pdata.fem_rhsadd_ilowers[box][d];
                        pdata.fem_rhsadd_iuppers[i][d] =
                           pdata.fem_rhsadd_iuppers[box][d];
                     }
                     dptr = pdata.fem_rhsadd_values[i];
                     pdata.fem_rhsadd_values[i] = pdata.fem_rhsadd_values[box];
                     pdata.fem_rhsadd_values[box] = dptr;
                     i++;
                     break;
                  }
               }
            }
            for (box = i; box < pdata.fem_rhsadd_nboxes; box++)
            {
               nalu_hypre_TFree(pdata.fem_rhsadd_values[box], NALU_HYPRE_MEMORY_HOST);
            }
            pdata.fem_rhsadd_nboxes = i;
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

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_matadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_matadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.rhsadd_iuppers[box], m);
            }
            for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
            {
               MapProblemIndex(pdata.fem_rhsadd_ilowers[box], m);
               MapProblemIndex(pdata.fem_rhsadd_iuppers[box], m);
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
         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            pdata.graph_boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[box] *=
                  (pdata.graph_iuppers[box][i] -
                   pdata.graph_ilowers[box][i] + 1);
            }
         }
         for (box = 0; box < pdata.matset_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.matset_iuppers[box][i] -
                        pdata.matset_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = nalu_hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.matadd_iuppers[box][i] -
                        pdata.matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = nalu_hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.fem_matadd_iuppers[box][i] -
                        pdata.fem_matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = nalu_hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.rhsadd_iuppers[box][i] -
                        pdata.rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = nalu_hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size *= (pdata.fem_rhsadd_iuppers[box][i] -
                        pdata.fem_rhsadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = nalu_hypre_max(pdata.max_boxsize, size);
         }

         /* refine periodicity */
         pdata.periodic[0] *= refine[part][0] * block[part][0] * distribute[part][0];
         pdata.periodic[1] *= refine[part][1] * block[part][1] * distribute[part][1];
         pdata.periodic[2] *= refine[part][2] * block[part][2] * distribute[part][2];
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
         nalu_hypre_TFree(pdata.glue_shared, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_maps, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_dirs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_primaries, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nboxes == 0)
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
         pdata.graph_values_size = 0;
         nalu_hypre_TFree(pdata.graph_values, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.d_graph_values, memory_location);
         nalu_hypre_TFree(pdata.graph_boxsizes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matset_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.matset_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matadd_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.matadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_nentries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_matadd_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.fem_matadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_nrows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_ncols, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_rows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_cols, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.rhsadd_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.rhsadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_rhsadd_nboxes == 0)
      {
         nalu_hypre_TFree(pdata.fem_rhsadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_rhsadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_rhsadd_values, NALU_HYPRE_MEMORY_HOST);
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
   NALU_HYPRE_MemoryLocation memory_location = data.memory_location;
   ProblemPartData  pdata;
   NALU_HYPRE_Int        part, box, s, i;

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
         nalu_hypre_TFree(pdata.glue_shared, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_parts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_nbor_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_maps, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_index_dirs, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.glue_primaries, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.nvars > 0)
      {
         nalu_hypre_TFree(pdata.stencil_num, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.graph_nboxes > 0)
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
         pdata.graph_values_size = 0;
         nalu_hypre_TFree(pdata.graph_values, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.d_graph_values, memory_location);
         nalu_hypre_TFree(pdata.graph_boxsizes, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matset_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.matset_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_strides, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matset_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.matadd_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.matadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_nentries, NALU_HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.matadd_entries[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.matadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(pdata.matadd_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.matadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_matadd_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.fem_matadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_nrows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_ncols, NALU_HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.fem_matadd_rows[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.fem_matadd_cols[box], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pdata.fem_matadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(pdata.fem_matadd_rows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_cols, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_matadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.rhsadd_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.rhsadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_vars, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.rhsadd_values, NALU_HYPRE_MEMORY_HOST);
      }

      if (pdata.fem_rhsadd_nboxes > 0)
      {
         nalu_hypre_TFree(pdata.fem_rhsadd_ilowers, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(pdata.fem_rhsadd_iuppers, NALU_HYPRE_MEMORY_HOST);
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            nalu_hypre_TFree(pdata.fem_rhsadd_values[box], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(pdata.fem_rhsadd_values, NALU_HYPRE_MEMORY_HOST);
      }
   }
   nalu_hypre_TFree(data.pdata, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(data.numghost, NALU_HYPRE_MEMORY_HOST);

   if (data.nstencils > 0)
   {
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
   }

   if (data.fem_nvars > 0)
   {
      for (s = 0; s < data.fem_nvars; s++)
      {
         nalu_hypre_TFree(data.fem_values_full[s], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(data.fem_ivalues_full[s], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(data.fem_offsets, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_vars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_values_full, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_ivalues_full, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_ordering, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_sparsity, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.fem_values, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.d_fem_values, memory_location);
   }

   if (data.fem_rhs_true > 0)
   {
      nalu_hypre_TFree(data.fem_rhs_values, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.d_fem_rhs_values, memory_location);
   }

   if (data.symmetric_num > 0)
   {
      nalu_hypre_TFree(data.symmetric_parts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_vars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_to_vars, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data.symmetric_booleans, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < data.ndists; i++)
   {
      nalu_hypre_TFree(data.dist_pools[i], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(data.dist_pools, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(data.dist_npools, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
SetCosineVector(NALU_HYPRE_Real  scale,
                Index       ilower,
                Index       iupper,
                NALU_HYPRE_Real *values)
{
   NALU_HYPRE_Int  i, j, k;
   NALU_HYPRE_Int  count = 0;

   for (k = ilower[2]; k <= iupper[2]; k++)
   {
      for (j = ilower[1]; j <= iupper[1]; j++)
      {
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[count] = scale * nalu_hypre_cos((i + j + k) / 10.0);
            count++;
         }
      }
   }

   return (0);
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
      nalu_hypre_printf("Usage: %s [-in <filename>] [<options>]\n", progname);
      nalu_hypre_printf("       %s -help | -version | -vernum \n", progname);
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -in <filename> : input file (default is `%s')\n",
                   infile_default);
      nalu_hypre_printf("  -fromfile <filename> : read SStructMatrix from file\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pt <pt1> <pt2> ... : set part(s) for subsequent options\n");
      nalu_hypre_printf("  -pooldist <p>       : pool distribution to use\n");
      nalu_hypre_printf("  -r <rx> <ry> <rz>   : refine part(s)\n");
      nalu_hypre_printf("  -P <Px> <Py> <Pz>   : refine and distribute part(s)\n");
      nalu_hypre_printf("  -b <bx> <by> <bz>   : refine and block part(s)\n");
      nalu_hypre_printf("  -solver <ID>        : solver ID (default = 39)\n");
      nalu_hypre_printf("                         0 - SMG split solver\n");
      nalu_hypre_printf("                         1 - PFMG split solver\n");
      nalu_hypre_printf("                         3 - SysPFMG\n");
      nalu_hypre_printf("                         8 - 1-step Jacobi split solver\n");
      nalu_hypre_printf("                        10 - PCG with SMG split precond\n");
      nalu_hypre_printf("                        11 - PCG with PFMG split precond\n");
      nalu_hypre_printf("                        13 - PCG with SysPFMG precond\n");
      nalu_hypre_printf("                        18 - PCG with diagonal scaling\n");
      nalu_hypre_printf("                        19 - PCG\n");
      nalu_hypre_printf("                        20 - PCG with BoomerAMG precond\n");
      nalu_hypre_printf("                        21 - PCG with EUCLID precond\n");
      nalu_hypre_printf("                        22 - PCG with ParaSails precond\n");
      nalu_hypre_printf("                        28 - PCG with diagonal scaling\n");
      nalu_hypre_printf("                        30 - GMRES with SMG split precond\n");
      nalu_hypre_printf("                        31 - GMRES with PFMG split precond\n");
      nalu_hypre_printf("                        38 - GMRES with diagonal scaling\n");
      nalu_hypre_printf("                        39 - GMRES\n");
      nalu_hypre_printf("                        40 - GMRES with BoomerAMG precond\n");
      nalu_hypre_printf("                        41 - GMRES with EUCLID precond\n");
      nalu_hypre_printf("                        42 - GMRES with ParaSails precond\n");
      nalu_hypre_printf("                        50 - BiCGSTAB with SMG split precond\n");
      nalu_hypre_printf("                        51 - BiCGSTAB with PFMG split precond\n");
      nalu_hypre_printf("                        58 - BiCGSTAB with diagonal scaling\n");
      nalu_hypre_printf("                        59 - BiCGSTAB\n");
      nalu_hypre_printf("                        60 - BiCGSTAB with BoomerAMG precond\n");
      nalu_hypre_printf("                        61 - BiCGSTAB with EUCLID precond\n");
      nalu_hypre_printf("                        62 - BiCGSTAB with ParaSails precond\n");
      nalu_hypre_printf("                        70 - Flexible GMRES with SMG split precond\n");
      nalu_hypre_printf("                        71 - Flexible GMRES with PFMG split precond\n");
      nalu_hypre_printf("                        78 - Flexible GMRES with diagonal scaling\n");
      nalu_hypre_printf("                        80 - Flexible GMRES with BoomerAMG precond\n");
      nalu_hypre_printf("                        90 - LGMRES with BoomerAMG precond\n");
      nalu_hypre_printf("                        120- ParCSRHybrid with DSCG/BoomerAMG precond\n");
      nalu_hypre_printf("                        150- AMS solver\n");
      nalu_hypre_printf("                        200- Struct SMG\n");
      nalu_hypre_printf("                        201- Struct PFMG\n");
      nalu_hypre_printf("                        202- Struct SparseMSG\n");
      nalu_hypre_printf("                        203- Struct PFMG constant coefficients\n");
      nalu_hypre_printf("                        204- Struct PFMG constant coefficients variable diagonal\n");
      nalu_hypre_printf("                        205- Struct Cyclic Reduction\n");
      nalu_hypre_printf("                        208- Struct Jacobi\n");
      nalu_hypre_printf("                        210- Struct CG with SMG precond\n");
      nalu_hypre_printf("                        211- Struct CG with PFMG precond\n");
      nalu_hypre_printf("                        212- Struct CG with SparseMSG precond\n");
      nalu_hypre_printf("                        217- Struct CG with 2-step Jacobi\n");
      nalu_hypre_printf("                        218- Struct CG with diagonal scaling\n");
      nalu_hypre_printf("                        219- Struct CG\n");
      nalu_hypre_printf("                        220- Struct Hybrid with SMG precond\n");
      nalu_hypre_printf("                        221- Struct Hybrid with PFMG precond\n");
      nalu_hypre_printf("                        222- Struct Hybrid with SparseMSG precond\n");
      nalu_hypre_printf("                        230- Struct GMRES with SMG precond\n");
      nalu_hypre_printf("                        231- Struct GMRES with PFMG precond\n");
      nalu_hypre_printf("                        232- Struct GMRES with SparseMSG precond\n");
      nalu_hypre_printf("                        237- Struct GMRES with 2-step Jacobi\n");
      nalu_hypre_printf("                        238- Struct GMRES with diagonal scaling\n");
      nalu_hypre_printf("                        239- Struct GMRES\n");
      nalu_hypre_printf("                        240- Struct BiCGSTAB with SMG precond\n");
      nalu_hypre_printf("                        241- Struct BiCGSTAB with PFMG precond\n");
      nalu_hypre_printf("                        242- Struct BiCGSTAB with SparseMSG precond\n");
      nalu_hypre_printf("                        247- Struct BiCGSTAB with 2-step Jacobi\n");
      nalu_hypre_printf("                        248- Struct BiCGSTAB with diagonal scaling\n");
      nalu_hypre_printf("                        249- Struct BiCGSTAB\n");
      nalu_hypre_printf("  -print             : print out the system\n");
      nalu_hypre_printf("  -rhsfromcosine     : solution is cosine function (default)\n");
      nalu_hypre_printf("  -rhsone            : rhs is vector with unit components\n");
      nalu_hypre_printf("  -tol <val>         : convergence tolerance (default 1e-6)\n");
      nalu_hypre_printf("  -solver_type <ID>  : Solver type for Hybrid\n");
      nalu_hypre_printf("                        1 - PCG (default)\n");
      nalu_hypre_printf("                        2 - GMRES\n");
      nalu_hypre_printf("                        3 - BiCGSTAB (only ParCSRHybrid)\n");
      nalu_hypre_printf("  -recompute <bool>  : Recompute residual in PCG?\n");
      nalu_hypre_printf("  -v <n_pre> <n_post>: SysPFMG and Struct- # of pre and post relax\n");
      nalu_hypre_printf("  -skip <s>          : SysPFMG and Struct- skip relaxation (0 or 1)\n");
      nalu_hypre_printf("  -rap <r>           : Struct- coarse grid operator type\n");
      nalu_hypre_printf("                        0 - Galerkin (default)\n");
      nalu_hypre_printf("                        1 - non-Galerkin ParFlow operators\n");
      nalu_hypre_printf("                        2 - Galerkin, general operators\n");
      nalu_hypre_printf("  -relax <r>         : Struct- relaxation type\n");
      nalu_hypre_printf("                        0 - Jacobi\n");
      nalu_hypre_printf("                        1 - Weighted Jacobi (default)\n");
      nalu_hypre_printf("                        2 - R/B Gauss-Seidel\n");
      nalu_hypre_printf("                        3 - R/B Gauss-Seidel (nonsymmetric)\n");
      nalu_hypre_printf("  -w <jacobi_weight> : jacobi weight\n");
      nalu_hypre_printf("  -jump <num>        : Struct- num levels to jump in SparseMSG\n");
      nalu_hypre_printf("  -cf <cf>           : Struct- convergence factor for Hybrid\n");
      nalu_hypre_printf("  -crtdim <tdim>     : Struct- cyclic reduction tdim\n");
      nalu_hypre_printf("  -cri <ix> <iy> <iz>: Struct- cyclic reduction base_index\n");
      nalu_hypre_printf("  -crs <sx> <sy> <sz>: Struct- cyclic reduction base_stride\n");
      nalu_hypre_printf("  -old_default: sets old BoomerAMG defaults, possibly better for 2D problems\n");

      /* begin lobpcg */

      nalu_hypre_printf("\nLOBPCG options:\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -lobpcg            : run LOBPCG instead of PCG\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -solver none       : no HYPRE preconditioner is used\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -itr <val>         : maximal number of LOBPCG iterations (default 100);\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -vrand <val>       : compute <val> eigenpairs using random initial vectors (default 1)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -seed <val>        : use <val> as the seed for the pseudo-random number generator\n");
      nalu_hypre_printf("                       (default seed is based on the time of the run)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -orthchk           : check eigenvectors for orthonormality\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -verb <val>        : verbosity level\n");
      nalu_hypre_printf("  -verb 0            : no print\n");
      nalu_hypre_printf("  -verb 1            : print initial eigenvalues and residuals, iteration number, number of\n");
      nalu_hypre_printf("                       non-convergent eigenpairs and final eigenvalues and residuals (default)\n");
      nalu_hypre_printf("  -verb 2            : print eigenvalues and residuals on each iteration\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pcgitr <val>      : maximal number of inner PCG iterations for preconditioning (default 1);\n");
      nalu_hypre_printf("                       if <val> = 0 then the preconditioner is applied directly\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -pcgtol <val>      : residual tolerance for inner iterations (default 0.01)\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -vout <val>        : file output level\n");
      nalu_hypre_printf("  -vout 0            : no files created (default)\n");
      nalu_hypre_printf("  -vout 1            : write eigenvalues to values.txt and residuals to residuals.txt\n");
      nalu_hypre_printf("  -vout 2            : in addition to the above, write the eigenvalues history (the matrix whose\n");
      nalu_hypre_printf("                       i-th column contains eigenvalues at (i+1)-th iteration) to val_hist.txt and\n");
      nalu_hypre_printf("                       residuals history to res_hist.txt\n");
      nalu_hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 10, 11, 13, and 18\n");
      nalu_hypre_printf("\ndefault solver is 10\n");

      /* end lobpcg */

      nalu_hypre_printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   MPI_Comm              comm = nalu_hypre_MPI_COMM_WORLD;

   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   NALU_HYPRE_Int             nparts;
   NALU_HYPRE_Int             pooldist;
   NALU_HYPRE_Int            *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   NALU_HYPRE_Int             solver_id, object_type;
   NALU_HYPRE_Int             print_system;
   NALU_HYPRE_Int             cosine;
   NALU_HYPRE_Real            scale;
   NALU_HYPRE_Int             read_fromfile_flag = 0;
   NALU_HYPRE_Int             read_fromfile_index[3] = {-1, -1, -1};

   NALU_HYPRE_SStructGrid     grid = NULL;
   NALU_HYPRE_SStructGrid     G_grid = NULL;
   NALU_HYPRE_SStructStencil *stencils = NULL;
   NALU_HYPRE_SStructStencil *G_stencils = NULL;
   NALU_HYPRE_SStructGraph    graph = NULL;
   NALU_HYPRE_SStructGraph    G_graph = NULL;
   NALU_HYPRE_SStructMatrix   A = NULL;
   NALU_HYPRE_SStructMatrix   G = NULL;
   NALU_HYPRE_SStructVector   b = NULL;
   NALU_HYPRE_SStructVector   x = NULL;
   NALU_HYPRE_SStructSolver   solver;
   NALU_HYPRE_SStructSolver   precond;

   NALU_HYPRE_ParCSRMatrix    par_A;
   NALU_HYPRE_ParVector       par_b;
   NALU_HYPRE_ParVector       par_x;
   NALU_HYPRE_Solver          par_solver;
   NALU_HYPRE_Solver          par_precond;

   NALU_HYPRE_StructMatrix    sA;
   NALU_HYPRE_StructVector    sb;
   NALU_HYPRE_StructVector    sx;
   NALU_HYPRE_StructSolver    struct_solver;
   NALU_HYPRE_StructSolver    struct_precond;

   Index                 ilower, iupper;
   Index                 index, to_index;

   NALU_HYPRE_Int             values_size;
   NALU_HYPRE_Real           *values = NULL;
   NALU_HYPRE_Real           *d_values = NULL;

   NALU_HYPRE_Int             num_iterations;
   NALU_HYPRE_Real            final_res_norm;

   NALU_HYPRE_Int             num_procs, myid;
   NALU_HYPRE_Int             time_index;

   NALU_HYPRE_Int             n_pre, n_post;
   NALU_HYPRE_Int             skip;
   NALU_HYPRE_Int             rap;
   NALU_HYPRE_Int             relax;
   NALU_HYPRE_Real            jacobi_weight;
   NALU_HYPRE_Int             usr_jacobi_weight;
   NALU_HYPRE_Int             jump;
   NALU_HYPRE_Int             solver_type;
   NALU_HYPRE_Int             recompute_res;

   NALU_HYPRE_Real            cf_tol;

   NALU_HYPRE_Int             cycred_tdim;
   Index                 cycred_index, cycred_stride;

   NALU_HYPRE_Int             arg_index, part, var, box, s, entry, i, j, k, size;
   NALU_HYPRE_Int             row, col;
   NALU_HYPRE_Int             gradient_matrix;
   NALU_HYPRE_Int             old_default;

   /* begin lobpcg */

   NALU_HYPRE_SStructSolver   lobpcg_solver;

   NALU_HYPRE_Int lobpcgFlag = 0;
   NALU_HYPRE_Int lobpcgSeed = 0;
   NALU_HYPRE_Int blockSize = 1;
   NALU_HYPRE_Int verbosity = 1;
   NALU_HYPRE_Int iterations;
   NALU_HYPRE_Int maxIterations = 100;
   NALU_HYPRE_Int checkOrtho = 0;
   NALU_HYPRE_Int printLevel = 0;
   NALU_HYPRE_Int pcgIterations = 0;
   NALU_HYPRE_Int pcgMode = 0;
   NALU_HYPRE_Real tol = 1e-6;
   NALU_HYPRE_Real pcgTol = 1e-2;
   NALU_HYPRE_Real nonOrthF;

   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constrains = NULL;
   NALU_HYPRE_Real* eigenvalues = NULL;

   NALU_HYPRE_Real* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   mv_InterfaceInterpreter* interpreter;
   NALU_HYPRE_MatvecFunctions matvec_fn;

   /* end lobpcg */

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   NALU_HYPRE_Int print_mem_tracker = 0;
   char mem_tracker_name[NALU_HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_Int spgemm_use_vendor = 0;
#endif

#if defined(NALU_HYPRE_TEST_USING_HOST)
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_HOST;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_HOST;
#else
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_DEVICE;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
#endif

   global_data.memory_location = memory_location;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Initialize() and should not be changed after
    *-----------------------------------------------------------------*/
   nalu_hypre_bind_device(myid, num_procs, comm);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   NALU_HYPRE_Initialize();

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   skip  = 0;
   rap   = 0;
   relax = 1;
   usr_jacobi_weight = 0;
   jump  = 0;
   gradient_matrix = 0;
   object_type = NALU_HYPRE_SSTRUCT;
   solver_type = 1;
   recompute_res = 0;   /* What should be the default here? */
   cf_tol = 0.90;
   pooldist = 0;
   cycred_tdim = 0;
   for (i = 0; i < 3; i++)
   {
      cycred_index[i]  = 0;
      cycred_stride[i] = 1;
   }

   solver_id = 39;
   print_system = 0;
   cosine = 1;
   skip = 0;
   n_pre  = 1;
   n_post = 1;

   old_default = 0;

   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/
   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
      else if (strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 1;
         read_fromfile_index[0] = arg_index++;
      }
      else if (strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 2;
         read_fromfile_index[1] = arg_index++;
      }
      else if (strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         read_fromfile_flag += 4;
         read_fromfile_index[2] = arg_index++;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-version") == 0 )
      {
         char *version_string;
         NALU_HYPRE_Version(&version_string);
         nalu_hypre_printf("%s\n", version_string);
         nalu_hypre_TFree(version_string, NALU_HYPRE_MEMORY_HOST);
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-vernum") == 0 )
      {
         NALU_HYPRE_Int major, minor, patch, single;
         NALU_HYPRE_VersionNumber(&major, &minor, &patch, &single);
         nalu_hypre_printf("HYPRE Version %d.%d.%d\n", major, minor, patch);
         nalu_hypre_printf("HYPRE Single = %d\n", single);
         exit(1);
      }
      else
      {
         break;
      }
   }

   /*-----------------------------------------------------------
    * Are we reading matrices/vectors directly from file?
    *-----------------------------------------------------------*/

   if (read_fromfile_index[0] == -1 &&
       read_fromfile_index[1] == -1 &&
       read_fromfile_index[2] == -1)
   {
      ReadData(infile, &global_data);

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

      if (global_data.rhs_true || global_data.fem_rhs_true)
      {
         cosine = 0;
      }
   }
   else
   {
      if (read_fromfile_flag < 7)
      {
         if (!myid)
         {
            nalu_hypre_printf("Error: Must read A, b, and x from file! \n");
         }
         exit(1);
      }
   }

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
      else if ( strcmp(argv[arg_index], "-pooldist") == 0 )
      {
         arg_index++;
         pooldist = atoi(argv[arg_index++]);
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

         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            solver_id = NO_SOLVER;
            arg_index++;
         }
         else /* end lobpcg */
         {
            solver_id = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromcosine") == 0 )
      {
         arg_index++;
         cosine = 1;
      }
      else if ( strcmp(argv[arg_index], "-rhsone") == 0 )
      {
         arg_index++;
         cosine = 0;
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 )
      {
         arg_index++;
         n_pre = atoi(argv[arg_index++]);
         n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-skip") == 0 )
      {
         arg_index++;
         skip = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-relax") == 0 )
      {
         arg_index++;
         relax = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         jacobi_weight = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         usr_jacobi_weight = 1; /* flag user weight */
      }
      else if ( strcmp(argv[arg_index], "-jump") == 0 )
      {
         arg_index++;
         jump = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crtdim") == 0 )
      {
         arg_index++;
         cycred_tdim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cri") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_index[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-crs") == 0 )
      {
         arg_index++;
         for (i = 0; i < 3; i++)
         {
            cycred_stride[i] = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-old_default") == 0 )
      {
         /* uses old BoomerAMG defaults */
         arg_index++;
         old_default = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: max inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations tolerance */
         arg_index++;
         pcgTol = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;
         /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         printLevel = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         arg_index++;
         memory_location = NALU_HYPRE_MEMORY_HOST;
      }
      else if ( strcmp(argv[arg_index], "-memory_device") == 0 )
      {
         arg_index++;
         memory_location = NALU_HYPRE_MEMORY_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         arg_index++;
         default_exec_policy = NALU_HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         arg_index++;
         default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
      }
#if defined(NALU_HYPRE_USING_GPU)
      else if ( strcmp(argv[arg_index], "-mm_vendor") == 0 )
      {
         arg_index++;
         spgemm_use_vendor = atoi(argv[arg_index++]);
      }
#endif
#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
      else if ( strcmp(argv[arg_index], "-print_mem_tracker") == 0 )
      {
         arg_index++;
         print_mem_tracker = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mem_tracker_filename") == 0 )
      {
         arg_index++;
         snprintf(mem_tracker_name, NALU_HYPRE_MAX_FILE_NAME_LEN, "%s", argv[arg_index++]);
      }
#endif
      else
      {
         arg_index++;
      }
   }

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   nalu_hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0]) { nalu_hypre_MemoryTrackerSetFileName(mem_tracker_name); }
#endif

   /* default memory location */
   NALU_HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   NALU_HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_SetSpGemmUseVendor(spgemm_use_vendor);
#endif

   if ( solver_id == 39 && lobpcgFlag )
   {
      solver_id = 10;
   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Print driver parameters TODO
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
#if defined(NALU_HYPRE_DEVELOP_STRING) && defined(NALU_HYPRE_DEVELOP_BRANCH)
      nalu_hypre_printf("\nUsing NALU_HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                   NALU_HYPRE_DEVELOP_STRING, NALU_HYPRE_DEVELOP_BRANCH);

#elif defined(NALU_HYPRE_DEVELOP_STRING) && !defined(NALU_HYPRE_DEVELOP_BRANCH)
      nalu_hypre_printf("\nUsing NALU_HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                   NALU_HYPRE_DEVELOP_STRING, NALU_HYPRE_BRANCH_NAME);

#elif defined(NALU_HYPRE_RELEASE_VERSION)
      nalu_hypre_printf("\nUsing NALU_HYPRE_RELEASE_VERSION: %s\n\n",
                   NALU_HYPRE_RELEASE_VERSION);
#endif
   }

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Barrier(comm);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   time_index = nalu_hypre_InitializeTiming("SStruct Interface");
   nalu_hypre_BeginTiming(time_index);

   if (read_fromfile_flag & 0x1)
   {
      if (!myid)
      {
         nalu_hypre_printf("Reading SStructMatrix A from file: %s\n", argv[read_fromfile_index[0]]);
      }

      NALU_HYPRE_SStructMatrixRead(comm, argv[read_fromfile_index[0]], &A);
   }
   else
   {
      /*-----------------------------------------------------------
       * Distribute data
       *-----------------------------------------------------------*/

      DistributeData(global_data, pooldist, refine, distribute, block,
                     num_procs, myid, &data);

      /*-----------------------------------------------------------
       * Check a few things
       *-----------------------------------------------------------*/
      if (solver_id >= 200)
      {
         pdata = data.pdata[0];
         if (nparts > 1)
         {
            if (!myid)
            {
               nalu_hypre_printf("Warning: Invalid number of parts for Struct Solver. Part 0 taken.\n");
            }
         }

         if (pdata.nvars > 1)
         {
            if (!myid)
            {
               nalu_hypre_printf("Error: Invalid number of nvars for Struct Solver \n");
            }
            exit(1);
         }
      }

      NALU_HYPRE_SStructGridCreate(comm, data.ndim, data.nparts, &grid);
      if (data.numghost != NULL)
      {
         NALU_HYPRE_SStructGridSetNumGhost(grid, data.numghost);
      }
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.nboxes; box++)
         {
            NALU_HYPRE_SStructGridSetExtents(grid, part,
                                        pdata.ilowers[box], pdata.iuppers[box]);
         }

         NALU_HYPRE_SStructGridSetVariables(grid, part, pdata.nvars, pdata.vartypes);

         /* GridAddVariabes */

         if (data.fem_nvars > 0)
         {
            NALU_HYPRE_SStructGridSetFEMOrdering(grid, part, data.fem_ordering);
         }

         /* GridSetNeighborPart and GridSetSharedPart */
         for (box = 0; box < pdata.glue_nboxes; box++)
         {
            if (pdata.glue_shared[box])
            {
               NALU_HYPRE_SStructGridSetSharedPart(grid, part,
                                              pdata.glue_ilowers[box],
                                              pdata.glue_iuppers[box],
                                              pdata.glue_offsets[box],
                                              pdata.glue_nbor_parts[box],
                                              pdata.glue_nbor_ilowers[box],
                                              pdata.glue_nbor_iuppers[box],
                                              pdata.glue_nbor_offsets[box],
                                              pdata.glue_index_maps[box],
                                              pdata.glue_index_dirs[box]);
            }
            else
            {
               NALU_HYPRE_SStructGridSetNeighborPart(grid, part,
                                                pdata.glue_ilowers[box],
                                                pdata.glue_iuppers[box],
                                                pdata.glue_nbor_parts[box],
                                                pdata.glue_nbor_ilowers[box],
                                                pdata.glue_nbor_iuppers[box],
                                                pdata.glue_index_maps[box],
                                                pdata.glue_index_dirs[box]);
            }
         }

         NALU_HYPRE_SStructGridSetPeriodic(grid, part, pdata.periodic);
      }
      NALU_HYPRE_SStructGridAssemble(grid);

      /*-----------------------------------------------------------
       * Set up the stencils
       *-----------------------------------------------------------*/

      stencils = nalu_hypre_CTAlloc(NALU_HYPRE_SStructStencil,  data.nstencils, NALU_HYPRE_MEMORY_HOST);
      for (s = 0; s < data.nstencils; s++)
      {
         NALU_HYPRE_SStructStencilCreate(data.ndim, data.stencil_sizes[s],
                                    &stencils[s]);
         for (entry = 0; entry < data.stencil_sizes[s]; entry++)
         {
            NALU_HYPRE_SStructStencilSetEntry(stencils[s], entry,
                                         data.stencil_offsets[s][entry],
                                         data.stencil_vars[s][entry]);
         }
      }

      /*-----------------------------------------------------------
       * Set object type
       *-----------------------------------------------------------*/
      /* determine if we build a gradient matrix */
      if (solver_id == 150)
      {
         gradient_matrix = 1;
         /* for now, change solver 150 to solver 28 */
         solver_id = 28;
      }

      if ( ((solver_id >= 20) && (solver_id < 30)) ||
           ((solver_id >= 40) && (solver_id < 50)) ||
           ((solver_id >= 60) && (solver_id < 70)) ||
           ((solver_id >= 80) && (solver_id < 90)) ||
           ((solver_id >= 90) && (solver_id < 100)) ||
           (solver_id == 120) )
      {
         object_type = NALU_HYPRE_PARCSR;
      }

      if (solver_id >= 200)
      {
         object_type = NALU_HYPRE_STRUCT;
      }

      /*-----------------------------------------------------------
       * Set up the graph
       *-----------------------------------------------------------*/

      NALU_HYPRE_SStructGraphCreate(comm, grid, &graph);

      /* NALU_HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != NALU_HYPRE_SSTRUCT )
      {
         NALU_HYPRE_SStructGraphSetObjectType(graph, object_type);
      }

      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];

         if (data.nstencils > 0)
         {
            /* set stencils */
            for (var = 0; var < pdata.nvars; var++)
            {
               NALU_HYPRE_SStructGraphSetStencil(graph, part, var,
                                            stencils[pdata.stencil_num[var]]);
            }
         }
         else if (data.fem_nvars > 0)
         {
            /* indicate FEM approach */
            NALU_HYPRE_SStructGraphSetFEM(graph, part);

            /* set sparsity */
            NALU_HYPRE_SStructGraphSetFEMSparsity(graph, part,
                                             data.fem_nsparse, data.fem_sparsity);
         }

         /* add entries */
         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            for (index[2] = pdata.graph_ilowers[box][2];
                 index[2] <= pdata.graph_iuppers[box][2];
                 index[2] += pdata.graph_strides[box][2])
            {
               for (index[1] = pdata.graph_ilowers[box][1];
                    index[1] <= pdata.graph_iuppers[box][1];
                    index[1] += pdata.graph_strides[box][1])
               {
                  for (index[0] = pdata.graph_ilowers[box][0];
                       index[0] <= pdata.graph_iuppers[box][0];
                       index[0] += pdata.graph_strides[box][0])
                  {
                     for (i = 0; i < 3; i++)
                     {
                        j = pdata.graph_index_maps[box][i];
                        k = index[i] - pdata.graph_ilowers[box][i];
                        k /= pdata.graph_strides[box][i];
                        k *= pdata.graph_index_signs[box][i];
#if 0 /* the following does not work with some Intel compilers with -O2 */
                        to_index[j] = pdata.graph_to_ilowers[box][j] +
                                      k * pdata.graph_to_strides[box][j];
#else
                        to_index[j] = pdata.graph_to_ilowers[box][j];
                        to_index[j] += k * pdata.graph_to_strides[box][j];
#endif
                     }
                     NALU_HYPRE_SStructGraphAddEntries(graph, part, index,
                                                  pdata.graph_vars[box],
                                                  pdata.graph_to_parts[box],
                                                  to_index,
                                                  pdata.graph_to_vars[box]);
                  }
               }
            }
         }
      }

      NALU_HYPRE_SStructGraphAssemble(graph);

      /*-----------------------------------------------------------
       * Set up the matrix
       *-----------------------------------------------------------*/

      values_size = nalu_hypre_max(data.max_boxsize, data.max_boxsize * data.fem_nsparse);

      values   = nalu_hypre_TAlloc(NALU_HYPRE_Real, values_size, NALU_HYPRE_MEMORY_HOST);
      d_values = nalu_hypre_TAlloc(NALU_HYPRE_Real, values_size, memory_location);

      NALU_HYPRE_SStructMatrixCreate(comm, graph, &A);

      /* TODO NALU_HYPRE_SStructMatrixSetSymmetric(A, 1); */
      for (i = 0; i < data.symmetric_num; i++)
      {
         NALU_HYPRE_SStructMatrixSetSymmetric(A, data.symmetric_parts[i],
                                         data.symmetric_vars[i],
                                         data.symmetric_to_vars[i],
                                         data.symmetric_booleans[i]);
      }
      NALU_HYPRE_SStructMatrixSetNSSymmetric(A, data.ns_symmetric);

      /* NALU_HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != NALU_HYPRE_SSTRUCT )
      {
         NALU_HYPRE_SStructMatrixSetObjectType(A, object_type);
      }

      NALU_HYPRE_SStructMatrixInitialize(A);

      if (data.nstencils > 0)
      {
         /* StencilSetEntry: set stencil values */
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               s = pdata.stencil_num[var];
               for (i = 0; i < data.stencil_sizes[s]; i++)
               {
                  for (j = 0; j < pdata.max_boxsize; j++)
                  {
                     values[j] = data.stencil_values[s][i];
                  }

                  nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                                memory_location, NALU_HYPRE_MEMORY_HOST);

                  for (box = 0; box < pdata.nboxes; box++)
                  {
                     GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                    pdata.vartypes[var], ilower, iupper);

                     NALU_HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                                     var, 1, &i, d_values);
                  }
               }
            }
         }
      }
      else if (data.fem_nvars > 0)
      {
         /* FEMStencilSetRow: add to stencil values */
#if 0    // Use AddFEMValues
         nalu_hypre_TMemcpy(data.d_fem_values, data.fem_values, NALU_HYPRE_Real,
                       data.fem_nsparse, memory_location, NALU_HYPRE_MEMORY_HOST);

         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               for (index[2] = pdata.ilowers[box][2];
                    index[2] <= pdata.iuppers[box][2]; index[2]++)
               {
                  for (index[1] = pdata.ilowers[box][1];
                       index[1] <= pdata.iuppers[box][1]; index[1]++)
                  {
                     for (index[0] = pdata.ilowers[box][0];
                          index[0] <= pdata.iuppers[box][0]; index[0]++)
                     {
                        NALU_HYPRE_SStructMatrixAddFEMValues(A, part, index,
                                                        data.d_fem_values);
                     }
                  }
               }
            }
         }
#else    // Use AddFEMBoxValues
         /* TODO: There is probably a smarter way to do this copy */
         for (i = 0; i < data.max_boxsize; i++)
         {
            j = i * data.fem_nsparse;
            nalu_hypre_TMemcpy(&d_values[j], data.fem_values, NALU_HYPRE_Real,
                          data.fem_nsparse, memory_location, NALU_HYPRE_MEMORY_HOST);
         }
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               NALU_HYPRE_SStructMatrixAddFEMBoxValues(
                  A, part, pdata.ilowers[box], pdata.iuppers[box], d_values);
            }
         }
#endif
      }

      /* GraphAddEntries: set non-stencil entries */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];

         nalu_hypre_TMemcpy(pdata.d_graph_values, pdata.graph_values,
                       NALU_HYPRE_Real, pdata.graph_values_size,
                       memory_location, NALU_HYPRE_MEMORY_HOST);

         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            /*
             * RDF NOTE: Add a separate interface routine for setting non-stencil
             * entries.  It would be more efficient to set boundary values a box
             * at a time, but AMR may require striding, and some codes may already
             * have a natural values array to pass in, but can't because it uses
             * ghost values.
             *
             * Example new interface routine:
             *   SetNSBoxValues(matrix, part, ilower, iupper, stride, entry
             *                  values_ilower, values_iupper, values);
             */

            /* since we have already tested SetBoxValues above, use SetValues here */
#if 0
            for (j = 0; j < pdata.graph_boxsizes[box]; j++)
            {
               values[j] = pdata.graph_values[box];
            }
            NALU_HYPRE_SStructMatrixSetBoxValues(A, part,
                                            pdata.graph_ilowers[box],
                                            pdata.graph_iuppers[box],
                                            pdata.graph_vars[box],
                                            1, &pdata.graph_entries[box],
                                            values);
#else
            for (index[2] = pdata.graph_ilowers[box][2];
                 index[2] <= pdata.graph_iuppers[box][2];
                 index[2] += pdata.graph_strides[box][2])
            {
               for (index[1] = pdata.graph_ilowers[box][1];
                    index[1] <= pdata.graph_iuppers[box][1];
                    index[1] += pdata.graph_strides[box][1])
               {
                  for (index[0] = pdata.graph_ilowers[box][0];
                       index[0] <= pdata.graph_iuppers[box][0];
                       index[0] += pdata.graph_strides[box][0])
                  {
                     NALU_HYPRE_SStructMatrixSetValues(A, part, index,
                                                  pdata.graph_vars[box],
                                                  1, &pdata.graph_entries[box],
                                                  &pdata.d_graph_values[box]);
                  }
               }
            }
#endif
         }
      }

      /* MatrixSetValues: reset some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.matset_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.matset_iuppers[box][j] -
                        pdata.matset_ilowers[box][j] + 1);
            }
            for (j = 0; j < size; j++)
            {
               values[j] = pdata.matset_values[box];
            }

            nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                          memory_location, NALU_HYPRE_MEMORY_HOST);

            NALU_HYPRE_SStructMatrixSetBoxValues(A, part,
                                            pdata.matset_ilowers[box],
                                            pdata.matset_iuppers[box],
                                            pdata.matset_vars[box],
                                            1, &pdata.matset_entries[box],
                                            d_values);
         }
      }

      /* MatrixAddToValues: add to some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.matadd_iuppers[box][j] -
                        pdata.matadd_ilowers[box][j] + 1);
            }

            for (entry = 0; entry < pdata.matadd_nentries[box]; entry++)
            {
               for (j = 0; j < size; j++)
               {
                  values[j] = pdata.matadd_values[box][entry];
               }

               nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                             memory_location, NALU_HYPRE_MEMORY_HOST);

               NALU_HYPRE_SStructMatrixAddToBoxValues(A, part,
                                                 pdata.matadd_ilowers[box],
                                                 pdata.matadd_iuppers[box],
                                                 pdata.matadd_vars[box],
                                                 1, &pdata.matadd_entries[box][entry],
                                                 d_values);
            }
         }
      }

      /* FEMMatrixAddToValues: add to some matrix values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.fem_matadd_nboxes; box++)
         {
            for (i = 0; i < data.fem_nsparse; i++)
            {
               values[i] = 0.0;
            }
            s = 0;
            for (i = 0; i < pdata.fem_matadd_nrows[box]; i++)
            {
               row = pdata.fem_matadd_rows[box][i];
               for (j = 0; j < pdata.fem_matadd_ncols[box]; j++)
               {
                  col = pdata.fem_matadd_cols[box][j];
                  values[data.fem_ivalues_full[row][col]] =
                     pdata.fem_matadd_values[box][s];
                  s++;
               }
            }

            nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                          memory_location, NALU_HYPRE_MEMORY_HOST);

            for (index[2] = pdata.fem_matadd_ilowers[box][2];
                 index[2] <= pdata.fem_matadd_iuppers[box][2]; index[2]++)
            {
               for (index[1] = pdata.fem_matadd_ilowers[box][1];
                    index[1] <= pdata.fem_matadd_iuppers[box][1]; index[1]++)
               {
                  for (index[0] = pdata.fem_matadd_ilowers[box][0];
                       index[0] <= pdata.fem_matadd_iuppers[box][0]; index[0]++)
                  {
                     NALU_HYPRE_SStructMatrixAddFEMValues(A, part, index, d_values);
                  }
               }
            }
         }
      }

      NALU_HYPRE_SStructMatrixAssemble(A);
   }

   /*-----------------------------------------------------------
    * Set up the RHS vector
    *-----------------------------------------------------------*/

   if (read_fromfile_flag & 0x2)
   {
      if (!myid)
      {
         nalu_hypre_printf("Reading SStructVector b from file: %s\n", argv[read_fromfile_index[1]]);
      }
      cosine = 0;

      NALU_HYPRE_SStructVectorRead(comm, argv[read_fromfile_index[1]], &b);
   }
   else
   {
      NALU_HYPRE_SStructVectorCreate(comm, grid, &b);

      /* NALU_HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != NALU_HYPRE_SSTRUCT )
      {
         NALU_HYPRE_SStructVectorSetObjectType(b, object_type);
      }

      NALU_HYPRE_SStructVectorInitialize(b);

      /* Initialize the rhs values */
      if (data.rhs_true)
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = data.rhs_value;
         }
      }
      else if (data.fem_rhs_true)
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = 0.0;
         }
      }
      else /* rhs=1 is the default */
      {
         for (j = 0; j < data.max_boxsize; j++)
         {
            values[j] = 1.0;
         }
      }

      nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               NALU_HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
                                               var, d_values);
            }
         }
      }

      /* Add values for FEMRhsSet */
      if (data.fem_rhs_true)
      {
#if 0    // Use AddFEMValues
         nalu_hypre_TMemcpy(data.d_fem_rhs_values, data.fem_rhs_values, NALU_HYPRE_Real,
                       data.fem_nvars, memory_location, NALU_HYPRE_MEMORY_HOST);

         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               for (index[2] = pdata.ilowers[box][2];
                    index[2] <= pdata.iuppers[box][2]; index[2]++)
               {
                  for (index[1] = pdata.ilowers[box][1];
                       index[1] <= pdata.iuppers[box][1]; index[1]++)
                  {
                     for (index[0] = pdata.ilowers[box][0];
                          index[0] <= pdata.iuppers[box][0]; index[0]++)
                     {
                        NALU_HYPRE_SStructVectorAddFEMValues(b, part, index,
                                                        data.d_fem_rhs_values);
                     }
                  }
               }
            }
         }
#else    // Use AddFEMBoxValues
         /* TODO: There is probably a smarter way to do this copy */
         for (i = 0; i < data.max_boxsize; i++)
         {
            j = i * data.fem_nvars;
            nalu_hypre_TMemcpy(&d_values[j], data.fem_rhs_values, NALU_HYPRE_Real,
                          data.fem_nvars, memory_location, NALU_HYPRE_MEMORY_HOST);
         }
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (box = 0; box < pdata.nboxes; box++)
            {
               NALU_HYPRE_SStructVectorAddFEMBoxValues(
                  b, part, pdata.ilowers[box], pdata.iuppers[box], d_values);
            }
         }
#endif
      }

      /* RhsAddToValues: add to some RHS values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.rhsadd_nboxes; box++)
         {
            size = 1;
            for (j = 0; j < 3; j++)
            {
               size *= (pdata.rhsadd_iuppers[box][j] -
                        pdata.rhsadd_ilowers[box][j] + 1);
            }

            for (j = 0; j < size; j++)
            {
               values[j] = pdata.rhsadd_values[box];
            }

            nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                          memory_location, NALU_HYPRE_MEMORY_HOST);

            NALU_HYPRE_SStructVectorAddToBoxValues(b, part,
                                              pdata.rhsadd_ilowers[box],
                                              pdata.rhsadd_iuppers[box],
                                              pdata.rhsadd_vars[box], d_values);
         }
      }

      /* FEMRhsAddToValues: add to some RHS values */
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.fem_rhsadd_nboxes; box++)
         {
            for (index[2] = pdata.fem_rhsadd_ilowers[box][2];
                 index[2] <= pdata.fem_rhsadd_iuppers[box][2]; index[2]++)
            {
               for (index[1] = pdata.fem_rhsadd_ilowers[box][1];
                    index[1] <= pdata.fem_rhsadd_iuppers[box][1]; index[1]++)
               {
                  for (index[0] = pdata.fem_rhsadd_ilowers[box][0];
                       index[0] <= pdata.fem_rhsadd_iuppers[box][0]; index[0]++)
                  {
                     NALU_HYPRE_SStructVectorAddFEMValues(b, part, index,
                                                     pdata.fem_rhsadd_values[box]);
                  }
               }
            }
         }
      }

      NALU_HYPRE_SStructVectorAssemble(b);
   }

   /*-----------------------------------------------------------
    * Set up the initial solution vector
    *-----------------------------------------------------------*/

   if (read_fromfile_flag & 0x4)
   {
      if (!myid)
      {
         nalu_hypre_printf("Reading SStructVector x0 from file: %s\n", argv[read_fromfile_index[2]]);
      }

      NALU_HYPRE_SStructVectorRead(comm, argv[read_fromfile_index[2]], &x);
   }
   else
   {
      NALU_HYPRE_SStructVectorCreate(comm, grid, &x);

      /* NALU_HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
      if ( object_type != NALU_HYPRE_SSTRUCT )
      {
         NALU_HYPRE_SStructVectorSetObjectType(x, object_type);
      }

      NALU_HYPRE_SStructVectorInitialize(x);

      /*-----------------------------------------------------------
       * If requested, reset linear system so that it has
       * exact solution:
       *
       *   u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
       *
       *-----------------------------------------------------------*/

      if (cosine)
      {
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               scale = (part + 1.0) * (var + 1.0);
               for (box = 0; box < pdata.nboxes; box++)
               {
                  /*
                     GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                     pdata.vartypes[var], ilower, iupper);
                  */
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 var, ilower, iupper);
                  SetCosineVector(scale, ilower, iupper, values);

                  nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size,
                                memory_location, NALU_HYPRE_MEMORY_HOST);

                  NALU_HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                                  var, d_values);
               }
            }
         }
      }

      NALU_HYPRE_SStructVectorAssemble(x);
   }

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("SStruct Interface", comm);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Get the objects out
    * NOTE: This should go after the cosine part, but for the bug
    *-----------------------------------------------------------*/

   if (object_type == NALU_HYPRE_PARCSR)
   {
      NALU_HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
      NALU_HYPRE_SStructVectorGetObject(b, (void **) &par_b);
      NALU_HYPRE_SStructVectorGetObject(x, (void **) &par_x);
   }
   else if (object_type == NALU_HYPRE_STRUCT)
   {
      NALU_HYPRE_SStructMatrixGetObject(A, (void **) &sA);
      NALU_HYPRE_SStructVectorGetObject(b, (void **) &sb);
      NALU_HYPRE_SStructVectorGetObject(x, (void **) &sx);
   }

   /*-----------------------------------------------------------
    * Finish resetting the linear system
    *-----------------------------------------------------------*/

   if (cosine)
   {
      /* This if/else is due to a bug in SStructMatvec */
      if (object_type == NALU_HYPRE_SSTRUCT)
      {
         /* Apply A to cosine vector to yield righthand side */
         nalu_hypre_SStructMatvec(1.0, A, x, 0.0, b);
         /* Reset initial guess to zero */
         nalu_hypre_SStructMatvec(0.0, A, b, 0.0, x);
      }
      else if (object_type == NALU_HYPRE_PARCSR)
      {
         /* Apply A to cosine vector to yield righthand side */
         NALU_HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b );
         /* Reset initial guess to zero */
         NALU_HYPRE_ParCSRMatrixMatvec(0.0, par_A, par_b, 0.0, par_x );
      }
      else if (object_type == NALU_HYPRE_STRUCT)
      {
         /* Apply A to cosine vector to yield righthand side */
         nalu_hypre_StructMatvec(1.0, sA, sx, 0.0, sb);
         /* Reset initial guess to zero */
         nalu_hypre_StructMatvec(0.0, sA, sb, 0.0, sx);
      }
   }

   /*-----------------------------------------------------------
    * Set up a gradient matrix G
    *-----------------------------------------------------------*/

   if (gradient_matrix)
   {
      NALU_HYPRE_SStructVariable vartypes[1] = {NALU_HYPRE_SSTRUCT_VARIABLE_NODE};
      NALU_HYPRE_Int offsets[3][2][3] = { {{0, 0, 0}, {-1, 0, 0}},
         {{0, 0, 0}, {0, -1, 0}},
         {{0, 0, 0}, {0, 0, -1}}
      };
      NALU_HYPRE_Real stencil_values[2] = {1.0, -1.0};

      /* Set up the domain grid */

      NALU_HYPRE_SStructGridCreate(comm, data.ndim, data.nparts, &G_grid);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (box = 0; box < pdata.nboxes; box++)
         {
            NALU_HYPRE_SStructGridSetExtents(G_grid, part,
                                        pdata.ilowers[box], pdata.iuppers[box]);
         }
         NALU_HYPRE_SStructGridSetVariables(G_grid, part, 1, vartypes);
         for (box = 0; box < pdata.glue_nboxes; box++)
         {
            if (pdata.glue_shared[box])
            {
               NALU_HYPRE_SStructGridSetSharedPart(G_grid, part,
                                              pdata.glue_ilowers[box],
                                              pdata.glue_iuppers[box],
                                              pdata.glue_offsets[box],
                                              pdata.glue_nbor_parts[box],
                                              pdata.glue_nbor_ilowers[box],
                                              pdata.glue_nbor_iuppers[box],
                                              pdata.glue_nbor_offsets[box],
                                              pdata.glue_index_maps[box],
                                              pdata.glue_index_dirs[box]);
            }
            else
            {
               NALU_HYPRE_SStructGridSetNeighborPart(G_grid, part,
                                                pdata.glue_ilowers[box],
                                                pdata.glue_iuppers[box],
                                                pdata.glue_nbor_parts[box],
                                                pdata.glue_nbor_ilowers[box],
                                                pdata.glue_nbor_iuppers[box],
                                                pdata.glue_index_maps[box],
                                                pdata.glue_index_dirs[box]);
            }
         }
      }
      NALU_HYPRE_SStructGridAssemble(G_grid);

      /* Set up the gradient stencils */

      G_stencils = nalu_hypre_CTAlloc(NALU_HYPRE_SStructStencil,  data.ndim, NALU_HYPRE_MEMORY_HOST);
      for (s = 0; s < data.ndim; s++)
      {
         NALU_HYPRE_SStructStencilCreate(data.ndim, 2, &G_stencils[s]);
         for (entry = 0; entry < 2; entry++)
         {
            NALU_HYPRE_SStructStencilSetEntry(
               G_stencils[s], entry, offsets[s][entry], 0);
         }
      }

      /* Set up the gradient graph */

      NALU_HYPRE_SStructGraphCreate(comm, grid, &G_graph);
      NALU_HYPRE_SStructGraphSetDomainGrid(G_graph, G_grid);
      NALU_HYPRE_SStructGraphSetObjectType(G_graph, NALU_HYPRE_PARCSR);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            NALU_HYPRE_SStructGraphSetStencil(G_graph, part, var, G_stencils[var]);
         }
      }
      NALU_HYPRE_SStructGraphAssemble(G_graph);

      /* Set up the matrix */

      NALU_HYPRE_SStructMatrixCreate(comm, G_graph, &G);
      NALU_HYPRE_SStructMatrixSetObjectType(G, NALU_HYPRE_PARCSR);
      NALU_HYPRE_SStructMatrixInitialize(G);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < data.ndim; var++)
         {
            for (i = 0; i < 2; i++)
            {
               for (j = 0; j < pdata.max_boxsize; j++)
               {
                  values[j] = stencil_values[i];
               }

               nalu_hypre_TMemcpy(d_values, values, NALU_HYPRE_Real, values_size, memory_location, NALU_HYPRE_MEMORY_HOST);

               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  NALU_HYPRE_SStructMatrixSetBoxValues(G, part, ilower, iupper,
                                                  var, 1, &i, d_values);
               }
            }
         }
      }

      NALU_HYPRE_SStructMatrixAssemble(G);
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      NALU_HYPRE_SStructVectorGather(b);
      NALU_HYPRE_SStructVectorGather(x);
      NALU_HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);

      if (gradient_matrix)
      {
         NALU_HYPRE_SStructMatrixPrint("sstruct.out.G",  G, 0);
      }
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

#if DEBUG
   {
      FILE *file;
      char  filename[255];

      /* result is 1's on the interior of the grid */
      nalu_hypre_SStructMatvec(1.0, A, b, 0.0, x);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.matvec", x, 0);

      /* result is all 1's */
      nalu_hypre_SStructCopy(b, x);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.copy", x, 0);

      /* result is all 2's */
      nalu_hypre_SStructScale(2.0, x);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.scale", x, 0);

      /* result is all 0's */
      nalu_hypre_SStructAxpy(-2.0, b, x);
      NALU_HYPRE_SStructVectorPrint("sstruct.out.axpy", x, 0);

      /* result is 1's with 0's on some boundaries */
      nalu_hypre_SStructCopy(b, x);
      nalu_hypre_sprintf(filename, "sstruct.out.gatherpre.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               nalu_hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  nalu_hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }
      fclose(file);

      /* result is all 1's */
      NALU_HYPRE_SStructVectorGather(x);
      nalu_hypre_sprintf(filename, "sstruct.out.gatherpost.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               nalu_hypre_fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  nalu_hypre_fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }

      /* re-initializes x to 0 */
      nalu_hypre_SStructAxpy(-1.0, b, x);
   }
#endif

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(d_values, memory_location);

   /*-----------------------------------------------------------
    * Solve the system using SysPFMG or Split
    *-----------------------------------------------------------*/

   if (solver_id == 3)
   {
      time_index = nalu_hypre_InitializeTiming("SysPFMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructSysPFMGCreate(comm, &solver);
      NALU_HYPRE_SStructSysPFMGSetMaxIter(solver, 100);
      NALU_HYPRE_SStructSysPFMGSetTol(solver, tol);
      NALU_HYPRE_SStructSysPFMGSetRelChange(solver, 0);
      /* weighted Jacobi = 1; red-black GS = 2 */
      NALU_HYPRE_SStructSysPFMGSetRelaxType(solver, relax);
      if (usr_jacobi_weight)
      {
         NALU_HYPRE_SStructSysPFMGSetJacobiWeight(solver, jacobi_weight);
      }
      NALU_HYPRE_SStructSysPFMGSetNumPreRelax(solver, n_pre);
      NALU_HYPRE_SStructSysPFMGSetNumPostRelax(solver, n_post);
      NALU_HYPRE_SStructSysPFMGSetSkipRelax(solver, skip);
      /*NALU_HYPRE_StructPFMGSetDxyz(solver, dxyz);*/
      NALU_HYPRE_SStructSysPFMGSetPrintLevel(solver, 1);
      NALU_HYPRE_SStructSysPFMGSetLogging(solver, 1);
      NALU_HYPRE_SStructSysPFMGSetup(solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("SysPFMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructSysPFMGSolve(solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_SStructSysPFMGGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      NALU_HYPRE_SStructSysPFMGDestroy(solver);
   }

   else if ((solver_id >= 0) && (solver_id < 10) && (solver_id != 3))
   {
      time_index = nalu_hypre_InitializeTiming("Split Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructSplitCreate(comm, &solver);
      NALU_HYPRE_SStructSplitSetMaxIter(solver, 100);
      NALU_HYPRE_SStructSplitSetTol(solver, tol);
      if (solver_id == 0)
      {
         NALU_HYPRE_SStructSplitSetStructSolver(solver, NALU_HYPRE_SMG);
      }
      else if (solver_id == 1)
      {
         NALU_HYPRE_SStructSplitSetStructSolver(solver, NALU_HYPRE_PFMG);
      }
      else if (solver_id == 8)
      {
         NALU_HYPRE_SStructSplitSetStructSolver(solver, NALU_HYPRE_Jacobi);
      }
      NALU_HYPRE_SStructSplitSetup(solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("Split Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructSplitSolve(solver, A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_SStructSplitGetNumIterations(solver, &num_iterations);
      NALU_HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver, &final_res_norm);

      NALU_HYPRE_SStructSplitDestroy(solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 10) && (solver_id < 20))
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructPCGCreate(comm, &solver);
      NALU_HYPRE_PCGSetMaxIter( (NALU_HYPRE_Solver) solver, 100 );
      NALU_HYPRE_PCGSetTol( (NALU_HYPRE_Solver) solver, tol );
      NALU_HYPRE_PCGSetTwoNorm( (NALU_HYPRE_Solver) solver, 1 );
      NALU_HYPRE_PCGSetRelChange( (NALU_HYPRE_Solver) solver, 0 );
      NALU_HYPRE_PCGSetPrintLevel( (NALU_HYPRE_Solver) solver, 1 );
      NALU_HYPRE_PCGSetRecomputeResidual( (NALU_HYPRE_Solver) solver, recompute_res);

      if ((solver_id == 10) || (solver_id == 11))
      {
         /* use Split solver as preconditioner */
         NALU_HYPRE_SStructSplitCreate(comm, &precond);
         NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
         NALU_HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 10)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
         }
         else if (solver_id == 11)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
         }
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                              (NALU_HYPRE_Solver) precond);
      }

      else if (solver_id == 13)
      {
         /* use SysPFMG solver as preconditioner */
         NALU_HYPRE_SStructSysPFMGCreate(comm, &precond);
         NALU_HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSysPFMGSetTol(precond, 0.0);
         NALU_HYPRE_SStructSysPFMGSetZeroGuess(precond);
         /* weighted Jacobi = 1; red-black GS = 2 */
         NALU_HYPRE_SStructSysPFMGSetRelaxType(precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_SStructSysPFMGSetJacobiWeight(precond, jacobi_weight);
         }
         NALU_HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
         NALU_HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
         NALU_HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
         /*NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSetup,
                              (NALU_HYPRE_Solver) precond);

      }
      else if (solver_id == 18)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                              (NALU_HYPRE_Solver) precond);
      }

      NALU_HYPRE_PCGSetup( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                      (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_PCGSolve( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                      (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_PCGGetNumIterations( (NALU_HYPRE_Solver) solver, &num_iterations );
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, &final_res_norm );
      NALU_HYPRE_SStructPCGDestroy(solver);

      if ((solver_id == 10) || (solver_id == 11))
      {
         NALU_HYPRE_SStructSplitDestroy(precond);
      }
      else if (solver_id == 13)
      {
         NALU_HYPRE_SStructSysPFMGDestroy(precond);
      }
   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/

   if ( lobpcgFlag && ( solver_id < 10 || solver_id >= 20 ) && verbosity )
   {
      nalu_hypre_printf("\nLOBPCG works with solvers 10, 11, 13 and 18 only\n");
   }

   if ( lobpcgFlag && (solver_id >= 10) && (solver_id < 20) )
   {

      interpreter = nalu_hypre_CTAlloc(mv_InterfaceInterpreter, 1, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructSetupInterpreter( interpreter );
      NALU_HYPRE_SStructSetupMatvec(&matvec_fn);

      if (myid != 0)
      {
         verbosity = 0;
      }

      if ( pcgIterations > 0 )
      {

         time_index = nalu_hypre_InitializeTiming("PCG Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_SStructPCGCreate(comm, &solver);
         NALU_HYPRE_PCGSetMaxIter( (NALU_HYPRE_Solver) solver, pcgIterations );
         NALU_HYPRE_PCGSetTol( (NALU_HYPRE_Solver) solver, pcgTol );
         NALU_HYPRE_PCGSetTwoNorm( (NALU_HYPRE_Solver) solver, 1 );
         NALU_HYPRE_PCGSetRelChange( (NALU_HYPRE_Solver) solver, 0 );
         NALU_HYPRE_PCGSetPrintLevel( (NALU_HYPRE_Solver) solver, 0 );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            NALU_HYPRE_SStructSplitCreate(comm, &precond);
            NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
            NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
            NALU_HYPRE_SStructSplitSetZeroGuess(precond);
            if (solver_id == 10)
            {
               NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
            }
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                 (NALU_HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            NALU_HYPRE_SStructSysPFMGCreate(comm, &precond);
            NALU_HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            NALU_HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            NALU_HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            NALU_HYPRE_SStructSysPFMGSetRelaxType(precond, 1);
            NALU_HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSetup,
                                 (NALU_HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                 (NALU_HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
            }
         }


         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", comm);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (NALU_HYPRE_Solver*)&lobpcg_solver);
         NALU_HYPRE_LOBPCGSetMaxIter((NALU_HYPRE_Solver)lobpcg_solver, maxIterations);
         NALU_HYPRE_LOBPCGSetPrecondUsageMode((NALU_HYPRE_Solver)lobpcg_solver, pcgMode);
         NALU_HYPRE_LOBPCGSetTol((NALU_HYPRE_Solver)lobpcg_solver, tol);
         NALU_HYPRE_LOBPCGSetPrintLevel((NALU_HYPRE_Solver)lobpcg_solver, verbosity);

         NALU_HYPRE_LOBPCGSetPrecond((NALU_HYPRE_Solver)lobpcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSetup,
                                (NALU_HYPRE_Solver)solver);

         NALU_HYPRE_LOBPCGSetup((NALU_HYPRE_Solver)lobpcg_solver, (NALU_HYPRE_Matrix)A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         if ( lobpcgSeed )
         {
            mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
         }
         else
         {
            mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
         }

         time_index = nalu_hypre_InitializeTiming("PCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve((NALU_HYPRE_Solver)lobpcg_solver, constrains,
                           eigenvectors, eigenvalues );

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", comm);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( (NALU_HYPRE_Solver)lobpcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( (NALU_HYPRE_Solver)lobpcg_solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( (NALU_HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( (NALU_HYPRE_Solver)lobpcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         NALU_HYPRE_SStructPCGDestroy(solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            NALU_HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            NALU_HYPRE_SStructSysPFMGDestroy(precond);
         }

         NALU_HYPRE_LOBPCGDestroy((NALU_HYPRE_Solver)lobpcg_solver);
         mv_MultiVectorDestroy( eigenvectors );
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {

         time_index = nalu_hypre_InitializeTiming("LOBPCG Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, (NALU_HYPRE_Solver*)&solver);
         NALU_HYPRE_LOBPCGSetMaxIter( (NALU_HYPRE_Solver) solver, maxIterations );
         NALU_HYPRE_LOBPCGSetTol( (NALU_HYPRE_Solver) solver, tol );
         NALU_HYPRE_LOBPCGSetPrintLevel( (NALU_HYPRE_Solver) solver, verbosity );

         if ((solver_id == 10) || (solver_id == 11))
         {
            /* use Split solver as preconditioner */
            NALU_HYPRE_SStructSplitCreate(comm, &precond);
            NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
            NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
            NALU_HYPRE_SStructSplitSetZeroGuess(precond);
            if (solver_id == 10)
            {
               NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
            }
            else if (solver_id == 11)
            {
               NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
            }
            NALU_HYPRE_LOBPCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                    (NALU_HYPRE_Solver) precond);
         }

         else if (solver_id == 13)
         {
            /* use SysPFMG solver as preconditioner */
            NALU_HYPRE_SStructSysPFMGCreate(comm, &precond);
            NALU_HYPRE_SStructSysPFMGSetMaxIter(precond, 1);
            NALU_HYPRE_SStructSysPFMGSetTol(precond, 0.0);
            NALU_HYPRE_SStructSysPFMGSetZeroGuess(precond);
            /* weighted Jacobi = 1; red-black GS = 2 */
            NALU_HYPRE_SStructSysPFMGSetRelaxType(precond, 1);
            NALU_HYPRE_SStructSysPFMGSetNumPreRelax(precond, n_pre);
            NALU_HYPRE_SStructSysPFMGSetNumPostRelax(precond, n_post);
            NALU_HYPRE_SStructSysPFMGSetSkipRelax(precond, skip);
            /*NALU_HYPRE_StructPFMGSetDxyz(precond, dxyz);*/
            NALU_HYPRE_LOBPCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSolve,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSysPFMGSetup,
                                    (NALU_HYPRE_Solver) precond);

         }
         else if (solver_id == 18)
         {
            /* use diagonal scaling as preconditioner */
            precond = NULL;
            NALU_HYPRE_LOBPCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                    (NALU_HYPRE_Solver) precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
            }
         }

         NALU_HYPRE_LOBPCGSetup( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", comm);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                              blockSize,
                                                              x );
         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         if ( lobpcgSeed )
         {
            mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
         }
         else
         {
            mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
         }

         time_index = nalu_hypre_InitializeTiming("LOBPCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve
         ( (NALU_HYPRE_Solver) solver, constrains, eigenvectors, eigenvalues );

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", comm);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( (NALU_HYPRE_Solver)solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( (NALU_HYPRE_Solver)solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( (NALU_HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( (NALU_HYPRE_Solver)solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         NALU_HYPRE_LOBPCGDestroy((NALU_HYPRE_Solver)solver);

         if ((solver_id == 10) || (solver_id == 11))
         {
            NALU_HYPRE_SStructSplitDestroy(precond);
         }
         else if (solver_id == 13)
         {
            NALU_HYPRE_SStructSysPFMGDestroy(precond);
         }

         mv_MultiVectorDestroy( eigenvectors );
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree( interpreter, NALU_HYPRE_MEMORY_HOST);

   }

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of PCG
    *-----------------------------------------------------------*/

   if ((solver_id >= 20) && (solver_id < 30))
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRPCGCreate(comm, &par_solver);
      NALU_HYPRE_PCGSetMaxIter( par_solver, 100 );
      NALU_HYPRE_PCGSetTol( par_solver, tol );
      NALU_HYPRE_PCGSetTwoNorm( par_solver, 1 );
      NALU_HYPRE_PCGSetRelChange( par_solver, 0 );
      NALU_HYPRE_PCGSetPrintLevel( par_solver, 1 );
      NALU_HYPRE_PCGSetRecomputeResidual( (NALU_HYPRE_Solver) par_solver, recompute_res);

      if (solver_id == 20)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { NALU_HYPRE_BoomerAMGSetOldDefault(par_precond); }
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         NALU_HYPRE_PCGSetPrecond( par_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                              par_precond );
      }
      else if (solver_id == 21)
      {
         /* use Euclid as preconditioner */
         NALU_HYPRE_EuclidCreate(comm, &par_precond);
         NALU_HYPRE_EuclidSetParams(par_precond, argc, argv);
         NALU_HYPRE_PCGSetPrecond(par_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                             par_precond);
      }
      else if (solver_id == 22)
      {
         /* use ParaSails as preconditioner */
         NALU_HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         NALU_HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         NALU_HYPRE_PCGSetPrecond( par_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSetup,
                              par_precond );
      }

      else if (solver_id == 28)
      {
         /* use diagonal scaling as preconditioner */
         par_precond = NULL;
         NALU_HYPRE_PCGSetPrecond(  par_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                               par_precond );
      }

      NALU_HYPRE_PCGSetup( par_solver, (NALU_HYPRE_Matrix) par_A,
                      (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_PCGSolve( par_solver, (NALU_HYPRE_Matrix) par_A,
                      (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_PCGGetNumIterations( par_solver, &num_iterations );
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
      NALU_HYPRE_ParCSRPCGDestroy(par_solver);

      if (solver_id == 20)
      {
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 21)
      {
         NALU_HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 22)
      {
         NALU_HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 30) && (solver_id < 40))
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructGMRESCreate(comm, &solver);
      NALU_HYPRE_GMRESSetKDim( (NALU_HYPRE_Solver) solver, 5 );
      NALU_HYPRE_GMRESSetMaxIter( (NALU_HYPRE_Solver) solver, 100 );
      NALU_HYPRE_GMRESSetTol( (NALU_HYPRE_Solver) solver, tol );
      NALU_HYPRE_GMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, 1 );
      NALU_HYPRE_GMRESSetLogging( (NALU_HYPRE_Solver) solver, 1 );

      if ((solver_id == 30) || (solver_id == 31))
      {
         /* use Split solver as preconditioner */
         NALU_HYPRE_SStructSplitCreate(comm, &precond);
         NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
         NALU_HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 30)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
         }
         else if (solver_id == 31)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
         }
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                (NALU_HYPRE_Solver) precond );
      }

      else if (solver_id == 38)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                (NALU_HYPRE_Solver) precond );
      }

      NALU_HYPRE_GMRESSetup( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                        (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                        (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_GMRESGetNumIterations( (NALU_HYPRE_Solver) solver, &num_iterations );
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, &final_res_norm );
      NALU_HYPRE_SStructGMRESDestroy(solver);

      if ((solver_id == 30) || (solver_id == 31))
      {
         NALU_HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 40) && (solver_id < 50))
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRGMRESCreate(comm, &par_solver);
      NALU_HYPRE_GMRESSetKDim(par_solver, 5);
      NALU_HYPRE_GMRESSetMaxIter(par_solver, 100);
      NALU_HYPRE_GMRESSetTol(par_solver, tol);
      NALU_HYPRE_GMRESSetPrintLevel(par_solver, 1);
      NALU_HYPRE_GMRESSetLogging(par_solver, 1);

      if (solver_id == 40)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { NALU_HYPRE_BoomerAMGSetOldDefault(par_precond); }
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         NALU_HYPRE_GMRESSetPrecond( par_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                par_precond);
      }
      else if (solver_id == 41)
      {
         /* use Euclid as preconditioner */
         NALU_HYPRE_EuclidCreate(comm, &par_precond);
         NALU_HYPRE_EuclidSetParams(par_precond, argc, argv);
         NALU_HYPRE_GMRESSetPrecond(par_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                               par_precond);
      }
      else if (solver_id == 42)
      {
         /* use ParaSails as preconditioner */
         NALU_HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         NALU_HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         NALU_HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         NALU_HYPRE_GMRESSetPrecond( par_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSetup,
                                par_precond);
      }

      NALU_HYPRE_GMRESSetup( par_solver, (NALU_HYPRE_Matrix) par_A,
                        (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve( par_solver, (NALU_HYPRE_Matrix) par_A,
                        (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_GMRESGetNumIterations( par_solver, &num_iterations);
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      NALU_HYPRE_ParCSRGMRESDestroy(par_solver);

      if (solver_id == 40)
      {
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 41)
      {
         NALU_HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 42)
      {
         NALU_HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 50) && (solver_id < 60))
   {
      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructBiCGSTABCreate(comm, &solver);
      NALU_HYPRE_BiCGSTABSetMaxIter( (NALU_HYPRE_Solver) solver, 100 );
      NALU_HYPRE_BiCGSTABSetTol( (NALU_HYPRE_Solver) solver, tol );
      NALU_HYPRE_BiCGSTABSetPrintLevel( (NALU_HYPRE_Solver) solver, 1 );
      NALU_HYPRE_BiCGSTABSetLogging( (NALU_HYPRE_Solver) solver, 1 );

      if ((solver_id == 50) || (solver_id == 51))
      {
         /* use Split solver as preconditioner */
         NALU_HYPRE_SStructSplitCreate(comm, &precond);
         NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
         NALU_HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 50)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
         }
         else if (solver_id == 51)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
         }
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                   (NALU_HYPRE_Solver) precond );
      }

      else if (solver_id == 58)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                   (NALU_HYPRE_Solver) precond );
      }

      NALU_HYPRE_BiCGSTABSetup( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                           (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BiCGSTABSolve( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                           (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_BiCGSTABGetNumIterations( (NALU_HYPRE_Solver) solver, &num_iterations );
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, &final_res_norm );
      NALU_HYPRE_SStructBiCGSTABDestroy(solver);

      if ((solver_id == 50) || (solver_id == 51))
      {
         NALU_HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of BiCGSTAB
    *-----------------------------------------------------------*/

   if ((solver_id >= 60) && (solver_id < 70))
   {
      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRBiCGSTABCreate(comm, &par_solver);
      NALU_HYPRE_BiCGSTABSetMaxIter(par_solver, 100);
      NALU_HYPRE_BiCGSTABSetTol(par_solver, tol);
      NALU_HYPRE_BiCGSTABSetPrintLevel(par_solver, 1);
      NALU_HYPRE_BiCGSTABSetLogging(par_solver, 1);

      if (solver_id == 60)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { NALU_HYPRE_BoomerAMGSetOldDefault(par_precond); }
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         NALU_HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   par_precond);
      }
      else if (solver_id == 61)
      {
         /* use Euclid as preconditioner */
         NALU_HYPRE_EuclidCreate(comm, &par_precond);
         NALU_HYPRE_EuclidSetParams(par_precond, argc, argv);
         NALU_HYPRE_BiCGSTABSetPrecond(par_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                  par_precond);
      }

      else if (solver_id == 62)
      {
         /* use ParaSails as preconditioner */
         NALU_HYPRE_ParCSRParaSailsCreate(comm, &par_precond );
         NALU_HYPRE_ParCSRParaSailsSetParams(par_precond, 0.1, 1);
         NALU_HYPRE_ParCSRParaSailsSetSym(par_precond, 0);
         NALU_HYPRE_BiCGSTABSetPrecond( par_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRParaSailsSetup,
                                   par_precond);
      }

      NALU_HYPRE_BiCGSTABSetup( par_solver, (NALU_HYPRE_Matrix) par_A,
                           (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BiCGSTABSolve( par_solver, (NALU_HYPRE_Matrix) par_A,
                           (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_BiCGSTABGetNumIterations( par_solver, &num_iterations);
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      NALU_HYPRE_ParCSRBiCGSTABDestroy(par_solver);

      if (solver_id == 60)
      {
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
      else if (solver_id == 61)
      {
         NALU_HYPRE_EuclidDestroy(par_precond);
      }
      else if (solver_id == 62)
      {
         NALU_HYPRE_ParCSRParaSailsDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Flexible GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 70) && (solver_id < 80))
   {
      time_index = nalu_hypre_InitializeTiming("FlexGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_SStructFlexGMRESCreate(comm, &solver);
      NALU_HYPRE_FlexGMRESSetKDim( (NALU_HYPRE_Solver) solver, 5 );
      NALU_HYPRE_FlexGMRESSetMaxIter( (NALU_HYPRE_Solver) solver, 100 );
      NALU_HYPRE_FlexGMRESSetTol( (NALU_HYPRE_Solver) solver, tol );
      NALU_HYPRE_FlexGMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, 1 );
      NALU_HYPRE_FlexGMRESSetLogging( (NALU_HYPRE_Solver) solver, 1 );

      if ((solver_id == 70) || (solver_id == 71))
      {
         /* use Split solver as preconditioner */
         NALU_HYPRE_SStructSplitCreate(comm, &precond);
         NALU_HYPRE_SStructSplitSetMaxIter(precond, 1);
         NALU_HYPRE_SStructSplitSetTol(precond, 0.0);
         NALU_HYPRE_SStructSplitSetZeroGuess(precond);
         if (solver_id == 70)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_SMG);
         }
         else if (solver_id == 71)
         {
            NALU_HYPRE_SStructSplitSetStructSolver(precond, NALU_HYPRE_PFMG);
         }
         NALU_HYPRE_FlexGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSolve,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructSplitSetup,
                                    (NALU_HYPRE_Solver) precond );
      }

      else if (solver_id == 78)
      {
         /* use diagonal scaling as preconditioner */
         precond = NULL;
         NALU_HYPRE_FlexGMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScale,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SStructDiagScaleSetup,
                                    (NALU_HYPRE_Solver) precond );
      }

      NALU_HYPRE_FlexGMRESSetup( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("FlexGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_FlexGMRESSolve( (NALU_HYPRE_Solver) solver, (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_FlexGMRESGetNumIterations( (NALU_HYPRE_Solver) solver, &num_iterations );
      NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, &final_res_norm );
      NALU_HYPRE_SStructFlexGMRESDestroy(solver);

      if ((solver_id == 70) || (solver_id == 71))
      {
         NALU_HYPRE_SStructSplitDestroy(precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of Flexible GMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 80) && (solver_id < 90))
   {
      time_index = nalu_hypre_InitializeTiming("FlexGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRFlexGMRESCreate(comm, &par_solver);
      NALU_HYPRE_FlexGMRESSetKDim(par_solver, 5);
      NALU_HYPRE_FlexGMRESSetMaxIter(par_solver, 100);
      NALU_HYPRE_FlexGMRESSetTol(par_solver, tol);
      NALU_HYPRE_FlexGMRESSetPrintLevel(par_solver, 1);
      NALU_HYPRE_FlexGMRESSetLogging(par_solver, 1);

      if (solver_id == 80)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { NALU_HYPRE_BoomerAMGSetOldDefault(par_precond); }
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         NALU_HYPRE_FlexGMRESSetPrecond( par_solver,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                    (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                    par_precond);
      }

      NALU_HYPRE_FlexGMRESSetup( par_solver, (NALU_HYPRE_Matrix) par_A,
                            (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("FlexGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_FlexGMRESSolve( par_solver, (NALU_HYPRE_Matrix) par_A,
                            (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_FlexGMRESGetNumIterations( par_solver, &num_iterations);
      NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      NALU_HYPRE_ParCSRFlexGMRESDestroy(par_solver);

      if (solver_id == 80)
      {
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR version of LGMRES
    *-----------------------------------------------------------*/

   if ((solver_id >= 90) && (solver_id < 100))
   {
      time_index = nalu_hypre_InitializeTiming("LGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRLGMRESCreate(comm, &par_solver);
      NALU_HYPRE_LGMRESSetKDim(par_solver, 10);
      NALU_HYPRE_LGMRESSetAugDim(par_solver, 2);
      NALU_HYPRE_LGMRESSetMaxIter(par_solver, 100);
      NALU_HYPRE_LGMRESSetTol(par_solver, tol);
      NALU_HYPRE_LGMRESSetPrintLevel(par_solver, 1);
      NALU_HYPRE_LGMRESSetLogging(par_solver, 1);

      if (solver_id == 90)
      {
         /* use BoomerAMG as preconditioner */
         NALU_HYPRE_BoomerAMGCreate(&par_precond);
         if (old_default) { NALU_HYPRE_BoomerAMGSetOldDefault(par_precond); }
         NALU_HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
         NALU_HYPRE_BoomerAMGSetTol(par_precond, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(par_precond, 1);
         NALU_HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
         NALU_HYPRE_LGMRESSetPrecond( par_solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                 par_precond);
      }

      NALU_HYPRE_LGMRESSetup( par_solver, (NALU_HYPRE_Matrix) par_A,
                         (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("LGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_LGMRESSolve( par_solver, (NALU_HYPRE_Matrix) par_A,
                         (NALU_HYPRE_Vector) par_b, (NALU_HYPRE_Vector) par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_LGMRESGetNumIterations( par_solver, &num_iterations);
      NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm( par_solver, &final_res_norm);
      NALU_HYPRE_ParCSRLGMRESDestroy(par_solver);

      if (solver_id == 90)
      {
         NALU_HYPRE_BoomerAMGDestroy(par_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using ParCSR hybrid DSCG/BoomerAMG
    *-----------------------------------------------------------*/

   if (solver_id == 120)
   {
      time_index = nalu_hypre_InitializeTiming("Hybrid Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridCreate(&par_solver);
      NALU_HYPRE_ParCSRHybridSetTol(par_solver, tol);
      NALU_HYPRE_ParCSRHybridSetTwoNorm(par_solver, 1);
      NALU_HYPRE_ParCSRHybridSetRelChange(par_solver, 0);
      NALU_HYPRE_ParCSRHybridSetPrintLevel(par_solver, 1); //13
      NALU_HYPRE_ParCSRHybridSetLogging(par_solver, 1);
      NALU_HYPRE_ParCSRHybridSetSolverType(par_solver, solver_type);
      NALU_HYPRE_ParCSRHybridSetRecomputeResidual(par_solver, recompute_res);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      /*
      NALU_HYPRE_ParCSRHybridSetPMaxElmts(par_solver, 8);
      NALU_HYPRE_ParCSRHybridSetRelaxType(par_solver, 18);
      NALU_HYPRE_ParCSRHybridSetCycleRelaxType(par_solver, 9, 3);
      NALU_HYPRE_ParCSRHybridSetCoarsenType(par_solver, 8);
      NALU_HYPRE_ParCSRHybridSetInterpType(par_solver, 3);
      NALU_HYPRE_ParCSRHybridSetMaxCoarseSize(par_solver, 20);
      */
#endif

#if SECOND_TIME
      nalu_hypre_ParVector *par_x2 =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParVectorComm(par_x), nalu_hypre_ParVectorGlobalSize(par_x),
                               nalu_hypre_ParVectorPartitioning(par_x));
      nalu_hypre_ParVectorInitialize(par_x2);
      nalu_hypre_ParVectorCopy(par_x, par_x2);

      NALU_HYPRE_ParCSRHybridSetup(par_solver, par_A, par_b, par_x);
      NALU_HYPRE_ParCSRHybridSolve(par_solver, par_A, par_b, par_x);

      nalu_hypre_ParVectorCopy(par_x2, par_x);
#endif

      nalu_hypre_GpuProfilingPushRange("HybridSolve");
      //cudaProfilerStart();

      NALU_HYPRE_ParCSRHybridSetup(par_solver, par_A, par_b, par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("Hybrid Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridSolve(par_solver, par_A, par_b, par_x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_ParCSRHybridGetNumIterations(par_solver, &num_iterations);
      NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(par_solver, &final_res_norm);

      /*
      NALU_HYPRE_Real time[4];
      NALU_HYPRE_ParCSRHybridGetSetupSolveTime(par_solver, time);
      if (myid == 0)
      {
         printf("ParCSRHybrid: Setup-Time1 %f, Solve-Time1 %f, Setup-Time2 %f, Solve-Time2 %f\n",
                time[0], time[1], time[2], time[3]);
      }
      */

      NALU_HYPRE_ParCSRHybridDestroy(par_solver);

      nalu_hypre_GpuProfilingPopRange();
      //cudaProfilerStop();

#if SECOND_TIME
      nalu_hypre_ParVectorDestroy(par_x2);
#endif
   }

   /*-----------------------------------------------------------
    * Solve the system using Struct solvers
    *-----------------------------------------------------------*/

   if (solver_id == 200)
   {
      time_index = nalu_hypre_InitializeTiming("SMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructSMGCreate(comm, &struct_solver);
      NALU_HYPRE_StructSMGSetMemoryUse(struct_solver, 0);
      NALU_HYPRE_StructSMGSetMaxIter(struct_solver, 50);
      NALU_HYPRE_StructSMGSetTol(struct_solver, tol);
      NALU_HYPRE_StructSMGSetRelChange(struct_solver, 0);
      NALU_HYPRE_StructSMGSetNumPreRelax(struct_solver, n_pre);
      NALU_HYPRE_StructSMGSetNumPostRelax(struct_solver, n_post);
      NALU_HYPRE_StructSMGSetPrintLevel(struct_solver, 1);
      NALU_HYPRE_StructSMGSetLogging(struct_solver, 1);
      NALU_HYPRE_StructSMGSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("SMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructSMGSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_StructSMGGetNumIterations(struct_solver, &num_iterations);
      NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      NALU_HYPRE_StructSMGDestroy(struct_solver);
   }

   else if ( solver_id == 201 || solver_id == 203 || solver_id == 204 )
   {
      time_index = nalu_hypre_InitializeTiming("PFMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructPFMGCreate(comm, &struct_solver);
      NALU_HYPRE_StructPFMGSetMaxIter(struct_solver, 50);
      NALU_HYPRE_StructPFMGSetTol(struct_solver, tol);
      NALU_HYPRE_StructPFMGSetRelChange(struct_solver, 0);
      NALU_HYPRE_StructPFMGSetRAPType(struct_solver, rap);
      NALU_HYPRE_StructPFMGSetRelaxType(struct_solver, relax);
      if (usr_jacobi_weight)
      {
         NALU_HYPRE_StructPFMGSetJacobiWeight(struct_solver, jacobi_weight);
      }
      NALU_HYPRE_StructPFMGSetNumPreRelax(struct_solver, n_pre);
      NALU_HYPRE_StructPFMGSetNumPostRelax(struct_solver, n_post);
      NALU_HYPRE_StructPFMGSetSkipRelax(struct_solver, skip);
      /*NALU_HYPRE_StructPFMGSetDxyz(struct_solver, dxyz);*/
      NALU_HYPRE_StructPFMGSetPrintLevel(struct_solver, 1);
      NALU_HYPRE_StructPFMGSetLogging(struct_solver, 1);
      NALU_HYPRE_StructPFMGSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PFMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructPFMGSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_StructPFMGGetNumIterations(struct_solver, &num_iterations);
      NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      NALU_HYPRE_StructPFMGDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Cyclic Reduction
    *-----------------------------------------------------------*/

   else if ( solver_id == 205 )
   {
      NALU_HYPRE_StructVector  sr;

      time_index = nalu_hypre_InitializeTiming("CycRed Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructCycRedCreate(comm, &struct_solver);
      NALU_HYPRE_StructCycRedSetTDim(struct_solver, cycred_tdim);
      NALU_HYPRE_StructCycRedSetBase(struct_solver, data.ndim,
                                cycred_index, cycred_stride);
      NALU_HYPRE_StructCycRedSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("CycRed Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructCycRedSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      num_iterations = 1;
      NALU_HYPRE_StructVectorCreate(comm,
                               nalu_hypre_StructVectorGrid(sb), &sr);
      NALU_HYPRE_StructVectorInitialize(sr);
      NALU_HYPRE_StructVectorAssemble(sr);
      NALU_HYPRE_StructVectorCopy(sb, sr);
      nalu_hypre_StructMatvec(-1.0, sA, sx, 1.0, sr);
      /* Using an inner product instead of a norm to help with testing */
      final_res_norm = nalu_hypre_StructInnerProd(sr, sr);
      if (final_res_norm < 1.0e-20)
      {
         final_res_norm = 0.0;
      }
      NALU_HYPRE_StructVectorDestroy(sr);

      NALU_HYPRE_StructCycRedDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using SparseMSG
    *-----------------------------------------------------------*/

   else if (solver_id == 202)
   {
      time_index = nalu_hypre_InitializeTiming("SparseMSG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructSparseMSGCreate(comm, &struct_solver);
      NALU_HYPRE_StructSparseMSGSetMaxIter(struct_solver, 50);
      NALU_HYPRE_StructSparseMSGSetJump(struct_solver, jump);
      NALU_HYPRE_StructSparseMSGSetTol(struct_solver, tol);
      NALU_HYPRE_StructSparseMSGSetRelChange(struct_solver, 0);
      NALU_HYPRE_StructSparseMSGSetRelaxType(struct_solver, relax);
      if (usr_jacobi_weight)
      {
         NALU_HYPRE_StructSparseMSGSetJacobiWeight(struct_solver, jacobi_weight);
      }
      NALU_HYPRE_StructSparseMSGSetNumPreRelax(struct_solver, n_pre);
      NALU_HYPRE_StructSparseMSGSetNumPostRelax(struct_solver, n_post);
      NALU_HYPRE_StructSparseMSGSetPrintLevel(struct_solver, 1);
      NALU_HYPRE_StructSparseMSGSetLogging(struct_solver, 1);
      NALU_HYPRE_StructSparseMSGSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("SparseMSG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructSparseMSGSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_StructSparseMSGGetNumIterations(struct_solver, &num_iterations);
      NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(struct_solver,
                                                        &final_res_norm);
      NALU_HYPRE_StructSparseMSGDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using Jacobi
    *-----------------------------------------------------------*/

   else if ( solver_id == 208 )
   {
      time_index = nalu_hypre_InitializeTiming("Jacobi Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructJacobiCreate(comm, &struct_solver);
      NALU_HYPRE_StructJacobiSetMaxIter(struct_solver, 100);
      NALU_HYPRE_StructJacobiSetTol(struct_solver, tol);
      NALU_HYPRE_StructJacobiSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("Jacobi Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructJacobiSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_StructJacobiGetNumIterations(struct_solver, &num_iterations);
      NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm(struct_solver,
                                                     &final_res_norm);
      NALU_HYPRE_StructJacobiDestroy(struct_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using CG
    *-----------------------------------------------------------*/

   if ((solver_id > 209) && (solver_id < 220))
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructPCGCreate(comm, &struct_solver);
      NALU_HYPRE_PCGSetMaxIter( (NALU_HYPRE_Solver)struct_solver, 100 );
      NALU_HYPRE_PCGSetTol( (NALU_HYPRE_Solver)struct_solver, tol );
      NALU_HYPRE_PCGSetTwoNorm( (NALU_HYPRE_Solver)struct_solver, 1 );
      NALU_HYPRE_PCGSetRelChange( (NALU_HYPRE_Solver)struct_solver, 0 );
      NALU_HYPRE_PCGSetPrintLevel( (NALU_HYPRE_Solver)struct_solver, 1 );
      NALU_HYPRE_PCGSetRecomputeResidual( (NALU_HYPRE_Solver)struct_solver, recompute_res);

      if (solver_id == 210)
      {
         /* use symmetric SMG as preconditioner */
         NALU_HYPRE_StructSMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         NALU_HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSMGSetLogging(struct_precond, 0);
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) struct_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                              (NALU_HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 211)
      {
         /* use symmetric PFMG as preconditioner */
         NALU_HYPRE_StructPFMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructPFMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         NALU_HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*NALU_HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         NALU_HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructPFMGSetLogging(struct_precond, 0);
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) struct_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                              (NALU_HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 212)
      {
         /* use symmetric SparseMSG as preconditioner */
         NALU_HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         NALU_HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) struct_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                              (NALU_HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 217)
      {
         /* use two-step Jacobi as preconditioner */
         NALU_HYPRE_StructJacobiCreate(comm, &struct_precond);
         NALU_HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         NALU_HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructJacobiSetZeroGuess(struct_precond);
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) struct_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                              (NALU_HYPRE_Solver) struct_precond);
      }

      else if (solver_id == 218)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) struct_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                              (NALU_HYPRE_Solver) struct_precond);
      }

      NALU_HYPRE_PCGSetup
      ( (NALU_HYPRE_Solver)struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_PCGSolve
      ( (NALU_HYPRE_Solver) struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_PCGGetNumIterations( (NALU_HYPRE_Solver)struct_solver, &num_iterations );
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver)struct_solver, &final_res_norm );
      NALU_HYPRE_StructPCGDestroy(struct_solver);

      if (solver_id == 210)
      {
         NALU_HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 211)
      {
         NALU_HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 212)
      {
         NALU_HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 217)
      {
         NALU_HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using Hybrid
    *-----------------------------------------------------------*/

   if ((solver_id > 219) && (solver_id < 230))
   {
      time_index = nalu_hypre_InitializeTiming("Hybrid Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructHybridCreate(comm, &struct_solver);
      NALU_HYPRE_StructHybridSetDSCGMaxIter(struct_solver, 100);
      NALU_HYPRE_StructHybridSetPCGMaxIter(struct_solver, 100);
      NALU_HYPRE_StructHybridSetTol(struct_solver, tol);
      /*NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor(struct_solver, 1.0e-200);*/
      NALU_HYPRE_StructHybridSetConvergenceTol(struct_solver, cf_tol);
      NALU_HYPRE_StructHybridSetTwoNorm(struct_solver, 1);
      NALU_HYPRE_StructHybridSetRelChange(struct_solver, 0);
      if (solver_type == 2) /* for use with GMRES */
      {
         NALU_HYPRE_StructHybridSetStopCrit(struct_solver, 0);
         NALU_HYPRE_StructHybridSetKDim(struct_solver, 10);
      }
      NALU_HYPRE_StructHybridSetPrintLevel(struct_solver, 1);
      NALU_HYPRE_StructHybridSetLogging(struct_solver, 1);
      NALU_HYPRE_StructHybridSetSolverType(struct_solver, solver_type);
      NALU_HYPRE_StructHybridSetRecomputeResidual(struct_solver, recompute_res);

      if (solver_id == 220)
      {
         /* use symmetric SMG as preconditioner */
         NALU_HYPRE_StructSMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         NALU_HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSMGSetLogging(struct_precond, 0);
         NALU_HYPRE_StructHybridSetPrecond(struct_solver,
                                      NALU_HYPRE_StructSMGSolve,
                                      NALU_HYPRE_StructSMGSetup,
                                      struct_precond);
      }

      else if (solver_id == 221)
      {
         /* use symmetric PFMG as preconditioner */
         NALU_HYPRE_StructPFMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructPFMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         NALU_HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*NALU_HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         NALU_HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructPFMGSetLogging(struct_precond, 0);
         NALU_HYPRE_StructHybridSetPrecond(struct_solver,
                                      NALU_HYPRE_StructPFMGSolve,
                                      NALU_HYPRE_StructPFMGSetup,
                                      struct_precond);
      }

      else if (solver_id == 222)
      {
         /* use symmetric SparseMSG as preconditioner */
         NALU_HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         NALU_HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         NALU_HYPRE_StructHybridSetPrecond(struct_solver,
                                      NALU_HYPRE_StructSparseMSGSolve,
                                      NALU_HYPRE_StructSparseMSGSetup,
                                      struct_precond);
      }

      NALU_HYPRE_StructHybridSetup(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("Hybrid Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructHybridSolve(struct_solver, sA, sb, sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_StructHybridGetNumIterations(struct_solver, &num_iterations);
      NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm(struct_solver, &final_res_norm);
      NALU_HYPRE_StructHybridDestroy(struct_solver);

      if (solver_id == 220)
      {
         NALU_HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 221)
      {
         NALU_HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 222)
      {
         NALU_HYPRE_StructSparseMSGDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if ((solver_id > 229) && (solver_id < 240))
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructGMRESCreate(comm, &struct_solver);
      NALU_HYPRE_GMRESSetMaxIter( (NALU_HYPRE_Solver)struct_solver, 100 );
      NALU_HYPRE_GMRESSetTol( (NALU_HYPRE_Solver)struct_solver, tol );
      NALU_HYPRE_GMRESSetRelChange( (NALU_HYPRE_Solver)struct_solver, 0 );
      NALU_HYPRE_GMRESSetPrintLevel( (NALU_HYPRE_Solver)struct_solver, 1 );
      NALU_HYPRE_GMRESSetLogging( (NALU_HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 230)
      {
         /* use symmetric SMG as preconditioner */
         NALU_HYPRE_StructSMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         NALU_HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSMGSetLogging(struct_precond, 0);
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                                (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 231)
      {
         /* use symmetric PFMG as preconditioner */
         NALU_HYPRE_StructPFMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructPFMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         NALU_HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*NALU_HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         NALU_HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructPFMGSetLogging(struct_precond, 0);
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                (NALU_HYPRE_Solver)struct_precond);
      }
      else if (solver_id == 232)
      {
         /* use symmetric SparseMSG as preconditioner */
         NALU_HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         NALU_HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 237)
      {
         /* use two-step Jacobi as preconditioner */
         NALU_HYPRE_StructJacobiCreate(comm, &struct_precond);
         NALU_HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         NALU_HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructJacobiSetZeroGuess(struct_precond);
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                                (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 238)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                                (NALU_HYPRE_Solver)struct_precond);
      }

      NALU_HYPRE_GMRESSetup
      ( (NALU_HYPRE_Solver)struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve
      ( (NALU_HYPRE_Solver)struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_GMRESGetNumIterations( (NALU_HYPRE_Solver)struct_solver, &num_iterations);
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver)struct_solver, &final_res_norm);
      NALU_HYPRE_StructGMRESDestroy(struct_solver);

      if (solver_id == 230)
      {
         NALU_HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 231)
      {
         NALU_HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 232)
      {
         NALU_HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 237)
      {
         NALU_HYPRE_StructJacobiDestroy(struct_precond);
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using BiCGTAB
    *-----------------------------------------------------------*/

   if ((solver_id > 239) && (solver_id < 250))
   {
      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_StructBiCGSTABCreate(comm, &struct_solver);
      NALU_HYPRE_BiCGSTABSetMaxIter( (NALU_HYPRE_Solver)struct_solver, 100 );
      NALU_HYPRE_BiCGSTABSetTol( (NALU_HYPRE_Solver)struct_solver, tol );
      NALU_HYPRE_BiCGSTABSetPrintLevel( (NALU_HYPRE_Solver)struct_solver, 1 );
      NALU_HYPRE_BiCGSTABSetLogging( (NALU_HYPRE_Solver)struct_solver, 1 );

      if (solver_id == 240)
      {
         /* use symmetric SMG as preconditioner */
         NALU_HYPRE_StructSMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSMGSetMemoryUse(struct_precond, 0);
         NALU_HYPRE_StructSMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSMGSetLogging(struct_precond, 0);
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSMGSetup,
                                   (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 241)
      {
         /* use symmetric PFMG as preconditioner */
         NALU_HYPRE_StructPFMGCreate(comm, &struct_precond);
         NALU_HYPRE_StructPFMGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructPFMGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructPFMGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructPFMGSetRAPType(struct_precond, rap);
         NALU_HYPRE_StructPFMGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructPFMGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructPFMGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructPFMGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructPFMGSetSkipRelax(struct_precond, skip);
         /*NALU_HYPRE_StructPFMGSetDxyz(struct_precond, dxyz);*/
         NALU_HYPRE_StructPFMGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructPFMGSetLogging(struct_precond, 0);
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructPFMGSetup,
                                   (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 242)
      {
         /* use symmetric SparseMSG as preconditioner */
         NALU_HYPRE_StructSparseMSGCreate(comm, &struct_precond);
         NALU_HYPRE_StructSparseMSGSetMaxIter(struct_precond, 1);
         NALU_HYPRE_StructSparseMSGSetJump(struct_precond, jump);
         NALU_HYPRE_StructSparseMSGSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructSparseMSGSetZeroGuess(struct_precond);
         NALU_HYPRE_StructSparseMSGSetRelaxType(struct_precond, relax);
         if (usr_jacobi_weight)
         {
            NALU_HYPRE_StructSparseMSGSetJacobiWeight(struct_precond, jacobi_weight);
         }
         NALU_HYPRE_StructSparseMSGSetNumPreRelax(struct_precond, n_pre);
         NALU_HYPRE_StructSparseMSGSetNumPostRelax(struct_precond, n_post);
         NALU_HYPRE_StructSparseMSGSetPrintLevel(struct_precond, 0);
         NALU_HYPRE_StructSparseMSGSetLogging(struct_precond, 0);
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructSparseMSGSetup,
                                   (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 247)
      {
         /* use two-step Jacobi as preconditioner */
         NALU_HYPRE_StructJacobiCreate(comm, &struct_precond);
         NALU_HYPRE_StructJacobiSetMaxIter(struct_precond, 2);
         NALU_HYPRE_StructJacobiSetTol(struct_precond, 0.0);
         NALU_HYPRE_StructJacobiSetZeroGuess(struct_precond);
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructJacobiSetup,
                                   (NALU_HYPRE_Solver)struct_precond);
      }

      else if (solver_id == 248)
      {
         /* use diagonal scaling as preconditioner */
         struct_precond = NULL;
         NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver)struct_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_StructDiagScaleSetup,
                                   (NALU_HYPRE_Solver)struct_precond);
      }

      NALU_HYPRE_BiCGSTABSetup
      ( (NALU_HYPRE_Solver)struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BiCGSTABSolve
      ( (NALU_HYPRE_Solver)struct_solver, (NALU_HYPRE_Matrix)sA, (NALU_HYPRE_Vector)sb,
        (NALU_HYPRE_Vector)sx);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", comm);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_BiCGSTABGetNumIterations( (NALU_HYPRE_Solver)struct_solver, &num_iterations);
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver)struct_solver, &final_res_norm);
      NALU_HYPRE_StructBiCGSTABDestroy(struct_solver);

      if (solver_id == 240)
      {
         NALU_HYPRE_StructSMGDestroy(struct_precond);
      }
      else if (solver_id == 241)
      {
         NALU_HYPRE_StructPFMGDestroy(struct_precond);
      }
      else if (solver_id == 242)
      {
         NALU_HYPRE_StructSparseMSGDestroy(struct_precond);
      }
      else if (solver_id == 247)
      {
         NALU_HYPRE_StructJacobiDestroy(struct_precond);
      }
   }

   /*-----------------------------------------------------------
    * Gather the solution vector
    *-----------------------------------------------------------*/

   NALU_HYPRE_SStructVectorGather(x);

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      FILE *file;
      char  filename[255];

      NALU_HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);

      /* print out with shared data replicated */
      if (!read_fromfile_flag)
      {
         values   = nalu_hypre_TAlloc(NALU_HYPRE_Real, data.max_boxsize, NALU_HYPRE_MEMORY_HOST);
         d_values = nalu_hypre_TAlloc(NALU_HYPRE_Real, data.max_boxsize, memory_location);
         for (part = 0; part < data.nparts; part++)
         {
            pdata = data.pdata[part];
            for (var = 0; var < pdata.nvars; var++)
            {
               nalu_hypre_sprintf(filename, "sstruct.out.xx.%02d.%02d.%05d", part, var, myid);
               if ((file = fopen(filename, "w")) == NULL)
               {
                  nalu_hypre_printf("Error: can't open output file %s\n", filename);
                  exit(1);
               }
               for (box = 0; box < pdata.nboxes; box++)
               {
                  GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                                 pdata.vartypes[var], ilower, iupper);
                  NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                                  var, d_values);
                  nalu_hypre_TMemcpy(values, d_values, NALU_HYPRE_Real, data.max_boxsize,
                                NALU_HYPRE_MEMORY_HOST, memory_location);
                  nalu_hypre_fprintf(file, "\nBox %d:\n\n", box);
                  size = 1;
                  for (j = 0; j < data.ndim; j++)
                  {
                     size *= (iupper[j] - ilower[j] + 1);
                  }
                  for (j = 0; j < size; j++)
                  {
                     nalu_hypre_fprintf(file, "%.14e\n", values[j]);
                  }
               }
               fflush(file);
               fclose(file);
            }
         }
         nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(d_values, memory_location);
      }
   }

   if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
   {
      nalu_hypre_printf("\n");
      nalu_hypre_printf("Iterations = %d\n", num_iterations);
      nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
      nalu_hypre_printf("\n");
   }

   /*-----------------------------------------------------------
    * Verify GetBoxValues()
    *-----------------------------------------------------------*/

#if 0
   {
      NALU_HYPRE_SStructVector   xnew;
      NALU_HYPRE_ParVector       par_xnew;
      NALU_HYPRE_StructVector    sxnew;
      NALU_HYPRE_Real            rnorm, bnorm;

      NALU_HYPRE_SStructVectorCreate(comm, grid, &xnew);
      NALU_HYPRE_SStructVectorSetObjectType(xnew, object_type);
      NALU_HYPRE_SStructVectorInitialize(xnew);

      /* get/set replicated shared data */
      values = nalu_hypre_TAlloc(NALU_HYPRE_Real,  data.max_boxsize, NALU_HYPRE_MEMORY_HOST);
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               NALU_HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               NALU_HYPRE_SStructVectorSetBoxValues(xnew, part, ilower, iupper,
                                               var, values);
            }
         }
      }
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_SStructVectorAssemble(xnew);

      /* Compute residual norm - this if/else is due to a bug in SStructMatvec */
      if (object_type == NALU_HYPRE_SSTRUCT)
      {
         NALU_HYPRE_SStructInnerProd(b, b, &bnorm);
         nalu_hypre_SStructMatvec(-1.0, A, xnew, 1.0, b);
         NALU_HYPRE_SStructInnerProd(b, b, &rnorm);
      }
      else if (object_type == NALU_HYPRE_PARCSR)
      {
         bnorm = nalu_hypre_ParVectorInnerProd(par_b, par_b);
         NALU_HYPRE_SStructVectorGetObject(xnew, (void **) &par_xnew);
         NALU_HYPRE_ParCSRMatrixMatvec(-1.0, par_A, par_xnew, 1.0, par_b );
         rnorm = nalu_hypre_ParVectorInnerProd(par_b, par_b);
      }
      else if (object_type == NALU_HYPRE_STRUCT)
      {
         bnorm = nalu_hypre_StructInnerProd(sb, sb);
         NALU_HYPRE_SStructVectorGetObject(xnew, (void **) &sxnew);
         nalu_hypre_StructMatvec(-1.0, sA, sxnew, 1.0, sb);
         rnorm = nalu_hypre_StructInnerProd(sb, sb);
      }
      bnorm = nalu_hypre_sqrt(bnorm);
      rnorm = nalu_hypre_sqrt(rnorm);

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("solver relnorm = %16.14e\n", final_res_norm);
         nalu_hypre_printf("check  relnorm = %16.14e, bnorm = %16.14e, rnorm = %16.14e\n",
                      (rnorm / bnorm), bnorm, rnorm);
         nalu_hypre_printf("\n");
      }

      NALU_HYPRE_SStructVectorDestroy(xnew);
   }
#endif

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   NALU_HYPRE_SStructMatrixDestroy(A);
   NALU_HYPRE_SStructVectorDestroy(b);
   NALU_HYPRE_SStructVectorDestroy(x);
   if (gradient_matrix)
   {
      for (s = 0; s < data.ndim; s++)
      {
         NALU_HYPRE_SStructStencilDestroy(G_stencils[s]);
      }
      nalu_hypre_TFree(G_stencils, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_SStructGraphDestroy(G_graph);
      NALU_HYPRE_SStructGridDestroy(G_grid);
      NALU_HYPRE_SStructMatrixDestroy(G);
   }

   if (!read_fromfile_flag)
   {
      NALU_HYPRE_SStructGridDestroy(grid);
      NALU_HYPRE_SStructGraphDestroy(graph);

      for (s = 0; s < data.nstencils; s++)
      {
         NALU_HYPRE_SStructStencilDestroy(stencils[s]);
      }
      nalu_hypre_TFree(stencils, NALU_HYPRE_MEMORY_HOST);

      DestroyData(data);
      nalu_hypre_TFree(parts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(refine, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(distribute, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(block, NALU_HYPRE_MEMORY_HOST);
   }
   /*nalu_hypre_FinalizeMemoryDebug(); */

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == NALU_HYPRE_MEMORY_HOST)
   {
      if (nalu_hypre_total_bytes[nalu_hypre_MEMORY_DEVICE] || nalu_hypre_total_bytes[nalu_hypre_MEMORY_UNIFIED])
      {
         nalu_hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
         nalu_hypre_assert(0);
      }
   }
#endif

   return (0);
}
