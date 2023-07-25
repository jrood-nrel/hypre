!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!*****************************************************************************
!  Routines to test NALU_HYPRE_ParCSRMatrix Fortran interface
!*****************************************************************************

!--------------------------------------------------------------------------
!  fnalu_hypre_parcsrmatrixcreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixcreate(fcomm, fglobal_num_rows, 
     1                                     fglobal_num_cols,
     1                                     frow_starts, fcol_starts, 
     3                                     fnum_cols_offd,
     2                                     fnum_nonzeros_diag, 
     5                                     fnum_nonzeros_offd, fmatrix)
      integer   ierr
      integer   fcomm
      integer   fglobal_num_rows
      integer   fglobal_num_cols
      integer   frow_starts
      integer   fcol_starts
      integer   fnum_cols_offd
      integer   fnum_nonzeros_diag
      integer   fnum_nonzeros_offd
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixCreate(fcomm, fglobal_num_rows, 
     1                              fglobal_num_cols, frow_starts,
     2                              fcol_starts, fnum_cols_offd,
     3                              fnum_nonzeros_diag, 
     4                              fnum_nonzeros_offd, fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixcreate: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixdestroy (fmatrix)
      integer ierr
      integer*8 fmatrix
   
      call NALU_HYPRE_ParCSRMatrixDestroy(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixdestroy: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixInitialize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixinitialize (fmatrix)
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixInitialize(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixinitialize: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixRead
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixread (fcomm, ffile_name, fmatrix)

      integer fcomm
      character*(*) ffile_name
      integer*8 fmatrix
      integer ierr

      call NALU_HYPRE_ParCSRMatrixRead(fcomm, ffile_name, fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixread: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixprint (fmatrix, fname)

      integer ierr
      integer*8 fmatrix
      character*(*) fname

      call NALU_HYPRE_ParCSRMatrixPrint(fmatrix, fname, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixprint: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetComm
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetcomm (fmatrix, fcomm)

      integer ierr
      integer fcomm
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixGetComm(fmatrix, fcomm)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetcomm: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetDims
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetdims (fmatrix, fM, fN)
      
      integer ierr
      integer fM
      integer fN
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixGetDims(fmatrix, fM, fN, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetdims: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetRowPartitioning
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetrowpartit (fmatrix, frow_ptr) 

      integer ierr
      integer*8 fmatrix
      integer*8 frow_ptr

      call NALU_HYPRE_ParCSRMatrixGetRowPartiti(fmatrix, frow_ptr, 
     1                                          ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetrowpartitioning: error = ',
     1                                                     ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetColPartitioning
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetcolpartit (fmatrix, fcol_ptr) 

      integer ierr
      integer*8 fmatrix
      integer*8 fcol_ptr

      call NALU_HYPRE_ParCSRMatrixGetColPartiti(fmatrix, fcol_ptr, 
     1                                          ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetcolpartitioning: error = ',
     1                                                     ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetLocalRange
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetlocalrange (fmatrix, frow_start, 
     1                                             frow_end, fcol_start,
     2                                             fcol_end)

      integer ierr
      integer frow_start
      integer frow_end
      integer fcol_start
      integer fcol_end
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixGetLocalRange(fmatrix, frow_start, 
     1                                     frow_end, fcol_start, 
     2                                     fcol_end, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetlocalrange: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixGetRow
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixgetrow (fmatrix, frow, fsize, 
     1                                      fcolptr, fvalptr)

      integer ierr
      integer frow
      integer fsize
      integer*8 fcolptr
      integer*8 fvalptr
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixGetRow(fmatrix, frow, fsize, fcolptr, 
     1                              fvalptr, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixgetrow: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixRestoreRow
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixrestorerow (fmatrix, frow, fsize, 
     1                                          fcolptr, fvalptr)

      integer ierr
      integer frow
      integer fsize
      integer*8 fcolptr
      integer*8 fvalptr
      integer*8 fmatrix

      call NALU_HYPRE_ParCSRMatrixRestoreRow(fmatrix, frow, fsize, fcolptr,
     1                                  fvalptr, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixrestorerow: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_CSRMatrixtoParCSRMatrix
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_csrmatrixtoparcsrmatrix (fcomm, fA_CSR, 
     1                                           frow_part, fcol_part,
     2                                           fmatrix)

      integer ierr
      integer fcomm
      integer frow_part
      integer fcol_part
      integer*8 fA_CSR
      integer*8 fmatrix

      call NALU_HYPRE_CSRMatriXToParCSRMatrix(fcomm, fA_CSR, frow_part, 
     1                                   fcol_part, fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_csrmatrixtoparcsrmatrix: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixMatvec
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixmatvec (falpha, fA, fx, fbeta, fy)

      integer ierr
      double precision falpha
      double precision fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_ParCSRMatrixMatvec(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixmatvec: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParCSRMatrixMatvecT
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixmatvect (falpha, fA, fx, fbeta, fy)
      
      integer ierr
      double precision falpha
      double precision fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_ParCSRMatrixMatvecT(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixmatvect: error = ', ierr
      endif
  
      return
      end



!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorcreate(fcomm, fsize, fpartion, fvector)

      integer ierr
      integer fcomm
      integer fsize
      integer*8 fvector
      integer*8 fpartion

      call NALU_HYPRE_ParVectorCreate(fcomm, fsize, fpartion, fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorcreate: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParMultiVectorCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parmultivectorcreate(fcomm, fsize, fpartion, 
     1                                       fnumvecs, fvector)

      integer ierr
      integer fcomm
      integer fsize
      integer fnumvecs
      integer*8 fvector
      integer*8 fpartion

      call NALU_HYPRE_ParMultiVectorCreate(fcomm, fsize, fpartion, fnumvecs,
     1                                fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parmultivectorcreate: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectordestroy (fvector)

      integer ierr
      integer*8 fvector

      call NALU_HYPRE_ParVectorDestroy(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectordestroy: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorInitialize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorinitialize (fvector)
   
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_ParVectorInitialize(fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorinitialize: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorRead
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorread (fcomm, fvector, fname)

      integer ierr
      integer fcomm
      character*(*) fname
      integer*8 fvector

      call NALU_HYPRE_ParVectorRead(fcomm, fname, fvector, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorread: error = ', ierr
      endif
  
      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorprint (fvector, fname, fsize)

      integer ierr
      integer fsize
      character*(*) fname
      integer*8 fvector

      call NALU_HYPRE_ParVectorPrint (fvector, fname, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorprint: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorSetConstantValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorsetconstantvalue (fvector, fvalue)

      integer ierr
      double precision fvalue
      integer*8 fvector

      call NALU_HYPRE_ParVectorSetConstantValue (fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorconstantvalues: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorSetRandomValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorsetrandomvalues (fvector, fseed)

      integer ierr
      integer fseed
      integer*8 fvector

      call NALU_HYPRE_ParVectorSetRandomValues (fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorrandomvalues: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorCopy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorcopy (fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_ParVectorCopy (fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorcopy: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorCloneShallow
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorcloneshallow (fx)

      integer ierr
      integer*8 fx

      call NALU_HYPRE_ParVectorCloneShallow (fx, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorcloneshallow: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorScale
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorscale (fvalue, fx)

      integer ierr
      double precision fvalue
      integer*8 fx

      call NALU_HYPRE_ParVectorScale (fvalue, fx, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorscale: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorAxpy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectoraxpy (fvalue, fx, fy)

      integer ierr
      double precision fvalue
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_ParVectorAxpy (fvalue, fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectoraxpy: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! NALU_HYPRE_ParVectorInnerProd
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorinnerprod (fx, fy, fprod)

      integer ierr
      double precision fprod
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_ParVectorInnerProd (fx, fy, fprod, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorinnerprod: error = ', ierr
      endif

      return
      end



!--------------------------------------------------------------------------
! nalu_hypre_ParCSRMatrixGlobalNumRows
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixglobalnumrow (fmatrix, fnrows)

      integer ierr
      integer fnrows
      integer*8 fmatrix

      call nalu_hypre_ParCSRMatrixGlobalNumRows (fmatrix, fnrows, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixglobalnumrows: error = ', ierr
      endif

      return
      end

!--------------------------------------------------------------------------
! nalu_hypre_ParCSRMatrixRowStarts
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parcsrmatrixrowstarts (fmatrix, frows)

      integer ierr
      integer*8 frows
      integer*8 fmatrix

      call nalu_hypre_ParCSRMatrixRowStarts (fmatrix, frows, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parcsrmatrixrowstarts: error = ', ierr
      endif

      return
      end



!--------------------------------------------------------------------------
! nalu_hypre_ParVectorSetDataOwner
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_parvectorsetdataowner (fv, fown)

      integer ierr
      integer fown
      integer*8 fv

      call nalu_hypre_SetParVectorDataOwner (fv, fown, ierr)

      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_parvectorsetdataowner: error = ', ierr
      endif

      return
      end
