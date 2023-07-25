!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

! -*- fortran -*-
!******************************************************************************
! 
!  Header file for HYPRE library
! 
! ****************************************************************************



! --------------------------------------------------------------------------
!  Structures
! --------------------------------------------------------------------------

! --------------------------------------------------------------------------
!  Constants
! --------------------------------------------------------------------------

      integer NALU_HYPRE_UNITIALIZED
      parameter( NALU_HYPRE_UNITIALIZED = -999 )

      integer NALU_HYPRE_PETSC_MAT_PARILUT_SOLVER
      parameter( NALU_HYPRE_PETSC_MAT_PARILUT_SOLVER = 222 )
      integer NALU_HYPRE_PARILUT
      parameter( NALU_HYPRE_PARILUT =                  333 )

      integer NALU_HYPRE_STRUCT
      parameter( NALU_HYPRE_STRUCT =  1111 )
      integer NALU_HYPRE_SSTRUCT
      parameter( NALU_HYPRE_SSTRUCT = 3333 )
      integer NALU_HYPRE_PARCSR
      parameter( NALU_HYPRE_PARCSR =  5555 )

      integer NALU_HYPRE_ISIS
      parameter( NALU_HYPRE_ISIS =    9911 )
      integer NALU_HYPRE_PETSC
      parameter( NALU_HYPRE_PETSC =   9933 )

      integer NALU_HYPRE_PFMG
      parameter( NALU_HYPRE_PFMG =    10 )
      integer NALU_HYPRE_SMG
      parameter( NALU_HYPRE_SMG =     11 )

      integer NALU_HYPRE_MEMORY_HOST
      parameter( NALU_HYPRE_MEMORY_HOST =   0 )
      integer NALU_HYPRE_MEMORY_DEVICE
      parameter( NALU_HYPRE_MEMORY_DEVICE = 1 )

      integer NALU_HYPRE_EXEC_HOST
      parameter( NALU_HYPRE_EXEC_HOST =   0 )
      integer NALU_HYPRE_EXEC_DEVICE
      parameter( NALU_HYPRE_EXEC_DEVICE = 1 )
