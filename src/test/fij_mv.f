!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!*****************************************************************************
! NALU_HYPRE_IJMatrix Fortran interface
!*****************************************************************************

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixcreate(fcomm, filower, fiupper,
     1                                 fjlower, fjupper, fmatrix)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fcomm
      integer filower
      integer fiupper
      integer fjlower
      integer fjupper

      call NALU_HYPRE_IJMatrixCreate(fcomm, filower, fiupper, fjlower, 
     1                          fjupper, fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixcreate error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixdestroy(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_IJMatrixDestroy(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixdestroy error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixInitialize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixinitialize(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_IJMatrixInitialize(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixinitialize error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixassemble(fmatrix)
      
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_IJMatrixAssemble(fmatrix, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixassemble error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixSetRowSizes
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixsetrowsizes(fmatrix, fizes)
      
      integer ierr
      integer*8 fmatrix
      integer fsizes

      call NALU_HYPRE_IJMatrixSetRowSizes(fmatrix, fsizes, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixsetrowsizes error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixSetDiagOffdSizes
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixsetdiagoffdsizes(fmatrix, fdiag_sizes,
     1                                           foffd_sizes)
      
      integer ierr
      integer*8 fmatrix
      integer fdiag_sizes
      integer foffd_sizes

      call NALU_HYPRE_IJMatrixSetDiagOffdSizes(fmatrix, fdiag_sizes, 
     1                                    foffd_sizes, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixsetdiagoffdsizes error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixSetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixsetvalues(fmatrix, fnrows, fncols, 
     1                                    frows, fcols, fvalues)
      
      integer ierr
      integer*8 fmatrix
      integer fnrows
      integer fncols
      integer frows
      integer fcols
      double precision fvalues

      call NALU_HYPRE_IJMatrixSetValues(fmatrix, fnrows, fncols, frows, 
     1                             fcols, fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixsetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixAddToValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixaddtovalues(fmatrix, fnrows, fncols,
     1                                      frows, fcols, fvalues)
      
      integer ierr
      integer*8 fmatrix
      integer fnrows
      integer fncols
      integer frows
      integer fcols
      double precision fvalues

      call NALU_HYPRE_IJMatrixAddToValues(fmatrix, fnrows, fncols, frows,
     1                               fcols, fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixaddtovalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixSetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixsetobjecttype(fmatrix, ftype)
      
      integer ierr
      integer*8 fmatrix
      integer ftype

      call NALU_HYPRE_IJMatrixSetObjectType(fmatrix, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixsetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixGetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixgetobjecttype(fmatrix, ftype)
      
      integer ierr
      integer*8 fmatrix
      integer ftype

      call NALU_HYPRE_IJMatrixGetObjectType(fmatrix, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixgetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixGetObject
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixgetobject(fmatrix, fobject)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fobject

      call NALU_HYPRE_IJMatrixGetObject(fmatrix, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixgetobject error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixRead
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixread(ffilename, fcomm, fobject_type,
     1                               fmatrix)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fcomm
      integer fobject_type
      character*(*) ffilename

      call NALU_HYPRE_IJMatrixRead(ffilename, fcomm, fobject_type, fmatrix,
     1                        ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixread error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJMatrixPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixprint(fmatrix, ffilename)
      
      integer ierr
      integer*8 fmatrix
      character*(*) ffilename

      call NALU_HYPRE_IJMatrixPrint(fmatrix, ffilename, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixprint error = ', ierr
      endif
 
      return
      end



!--------------------------------------------------------------------------
! nalu_hypre_IJMatrixSetObject
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijmatrixsetobject(fmatrix, fobject)
      
      integer ierr
      integer*8 fmatrix
      integer*8 fobject

      call nalu_hypre_IJMatrixSetObject(fmatrix, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijmatrixsetobject error = ', ierr
      endif
 
      return
      end



!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorcreate(fcomm, fjlower, fjupper, fvector)
      
      integer ierr
      integer*8 fvector
      integer fcomm
      integer fjlower
      integer fjupper

      call NALU_HYPRE_IJVectorCreate(fcomm, fjlower, fjupper, fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorcreate error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectordestroy(fvector)
      
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_IJVectorDestroy(fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectordestroy error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorInitialize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorinitialize(fvector)
      
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_IJVectorInitialize(fvector, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorinitialize error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorSetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorsetvalues(fvector, fnum_values, 
     1                                    findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call NALU_HYPRE_IJVectorSetValues(fvector, fnum_values, findices,
     1                             fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorsetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorAddToValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectoraddtovalues(fvector, fnum_values,
     1                                      findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call NALU_HYPRE_IJVectorAddToValues(fvector, fnum_values, findices,
     1                               fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectoraddtovalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorassemble(fvector)
      
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_IJVectorAssemble(fvector , ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorassemble error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorGetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorgetvalues(fvector, fnum_values, 
     1                                    findices, fvalues)
      
      integer ierr
      integer*8 fvector
      integer fnum_values
      integer findices
      double precision fvalues

      call NALU_HYPRE_IJVectorGetValues(fvector, fnum_values, findices,
     1                             fvalues, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorgetvalues error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorSetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorsetobjecttype(fvector, ftype)
      
      integer ierr
      integer*8 fvector
      integer ftype

      call NALU_HYPRE_IJVectorSetObjectType(fvector, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorsetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorGetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorgetobjecttype(fvector, ftype)
      
      integer ierr
      integer*8 fvector
      integer ftype

      call NALU_HYPRE_IJVectorGetObjectType(fvector, ftype, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorgetobjecttype error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorGetObject
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorgetobject(fvector, fobject)
      
      integer ierr
      integer*8 fvector
      integer*8 fobject

      call NALU_HYPRE_IJVectorGetObject(fvector, fobject, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorgetobject error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorRead
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorread(ffilename, fcomm, fobject_type,
     1                               fvector)
      
      integer ierr
      integer*8 fvector
      integer*8 fcomm
      integer fobject_type
      character*(*) ffilename

      call NALU_HYPRE_IJVectorRead(ffilename, fcomm, fobject_type, fvector,
     1                        ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorread error = ', ierr
      endif
 
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_IJVectorPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_ijvectorprint(fvector, ffilename)
      
      integer ierr
      integer*8 fvector
      character*(*) ffilename

      call NALU_HYPRE_IJVectorPrint(fvector, ffilename, ierr)

      if(ierr .ne. 0) then
         print *, 'fnalu_hypre_ijvectorprint error = ', ierr
      endif
 
      return
      end
