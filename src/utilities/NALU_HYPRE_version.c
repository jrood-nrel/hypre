/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Version utility functions
 *
 *****************************************************************************/

#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_Version( char **version_ptr )
{
   NALU_HYPRE_Int  len = 30;
   char      *version;

   /* compute string length */
   len += strlen(NALU_HYPRE_RELEASE_VERSION);

   version = nalu_hypre_CTAlloc(char, len, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_sprintf(version, "HYPRE Release Version %s", NALU_HYPRE_RELEASE_VERSION);

   *version_ptr = version;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VersionNumber( NALU_HYPRE_Int  *major_ptr,
                     NALU_HYPRE_Int  *minor_ptr,
                     NALU_HYPRE_Int  *patch_ptr,
                     NALU_HYPRE_Int  *single_ptr )
{
   NALU_HYPRE_Int  major, minor, patch, single;
   NALU_HYPRE_Int  nums[3], i, j;
   char      *ptr = (char *) NALU_HYPRE_RELEASE_VERSION;

   /* get major/minor/patch numbers */
   for (i = 0; i < 3; i++)
   {
      char str[4];

      for (j = 0; (j < 3) && (*ptr != '.') && (*ptr != '\0'); j++)
      {
         str[j] = *ptr;
         ptr++;
      }
      str[j] = '\0';
      nums[i] = atoi((char *)str);
      ptr++;
   }
   major = nums[0];
   minor = nums[1];
   patch = nums[2];

   single = (NALU_HYPRE_Int) NALU_HYPRE_RELEASE_NUMBER;

   if (major_ptr)   {*major_ptr   = major;}
   if (minor_ptr)   {*minor_ptr   = minor;}
   if (patch_ptr)   {*patch_ptr   = patch;}
   if (single_ptr)  {*single_ptr  = single;}

   return nalu_hypre_error_flag;
}

