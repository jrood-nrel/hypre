!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!**************************************************
!      Routines to test struct_mv fortran interface
!**************************************************


!**************************************************
!           NALU_HYPRE_StructStencil routines
!**************************************************

!******************************************
!      fnalu_hypre_structstencilcreate
!******************************************
      subroutine fnalu_hypre_structstencilcreate(fdim, fdim1, fstencil)
      integer ierr
      integer fdim
      integer fdim1
      integer*8 fstencil

      call NALU_HYPRE_StructStencilCreate(fdim, fdim1, fstencil, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structstencilcreate: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structstencilsetelement
!******************************************
      subroutine fnalu_hypre_structstencilsetelement(fstencil, findx,
     1                                          foffset)
      integer ierr
      integer findx
      integer foffset(*)
      integer*8 fstencil

      call NALU_HYPRE_StructStencilSetElement(fstencil, findx, foffset,
     1                                   ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structstencilsetelement: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structstencildestroy
!******************************************
      subroutine fnalu_hypre_structstencildestroy(fstencil)
      integer ierr
      integer*8 fstencil

      call NALU_HYPRE_StructStencilDestroy(fstencil, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structstencildestroy: error = ', ierr
      endif

      return
      end



!**************************************************
!           NALU_HYPRE_StructGrid routines
!**************************************************

!******************************************
!      fnalu_hypre_structgridcreate
!******************************************
      subroutine fnalu_hypre_structgridcreate(fcomm, fdim, fgrid)
      integer ierr
      integer fcomm
      integer fdim
      integer*8 fgrid

      call NALU_HYPRE_StructGridCreate(fcomm, fdim, fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgridcreate: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structgriddestroy
!******************************************
      subroutine fnalu_hypre_structgriddestroy(fgrid)
      integer ierr
      integer*8 fgrid

      call NALU_HYPRE_StructGridDestroy(fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgriddestroy: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structgridsetextents
!******************************************
      subroutine fnalu_hypre_structgridsetextents(fgrid, flower, fupper)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer*8 fgrid

      call NALU_HYPRE_StructGridSetExtents(fgrid, flower, fupper, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgridsetelement: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structgridsetperiodic
!******************************************
      subroutine fnalu_hypre_structgridsetperiodic(fgrid, fperiod)
      integer ierr
      integer fperiod(*)
      integer*8 fgrid

      call NALU_HYPRE_StructGridSetPeriodic(fgrid, fperiod, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgridsetperiodic: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structgridassemble
!******************************************
      subroutine fnalu_hypre_structgridassemble(fgrid)
      integer ierr
      integer*8 fgrid

      call NALU_HYPRE_StructGridAssemble(fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgridassemble: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structgridsetnumghost
!******************************************
      subroutine fnalu_hypre_structgridsetnumghost(fgrid, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fgrid

      call NALU_HYPRE_StructGridSetNumGhost(fgrid, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structgridsetnumghost: error = ', ierr
      endif

      return
      end



!**************************************************
!           NALU_HYPRE_StructMatrix routines
!**************************************************

!******************************************
!      fnalu_hypre_structmatrixcreate
!******************************************
      subroutine fnalu_hypre_structmatrixcreate(fcomm, fgrid, fstencil, 
     1                                     fmatrix)
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fstencil
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixCreate(fcomm, fgrid, fstencil, fmatrix,
     1                              ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixcreate: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixdestroy
!******************************************
      subroutine fnalu_hypre_structmatrixdestroy(fmatrix)
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixDestroy(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixdestroy: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixinitialize
!******************************************
      subroutine fnalu_hypre_structmatrixinitialize(fmatrix)
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixInitialize(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixinitialize: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetvalues
!******************************************
      subroutine fnalu_hypre_structmatrixsetvalues(fmatrix, fgridindx, 
     1                                        fnumsindx, fsindx, fvals)
      integer ierr
      integer fgridindx(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetValues(fmatrix, fgridindx, fnumsindx, 
     1                                 fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetboxvalues
!******************************************
      subroutine fnalu_hypre_structmatrixsetboxvalues(fmatrix, flower,
     1                                           fupper, fnumsindx,
     2                                           fsindx, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetBoxValues(fmatrix, flower, fupper,
     1                                    fnumsindx, fsindx, fvals,
     2                                    ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetboxvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixgetboxvalues
!******************************************
      subroutine fnalu_hypre_structmatrixgetboxvalues(fmatrix, flower,
     1                                           fupper, fnumsindx,
     2                                           fsindx, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixGetBoxValues(fmatrix, flower, fupper,
     1                                    fnumsindx, fsindx, fvals,
     2                                    ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixgetboxvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetconstantentries
!******************************************
      subroutine fnalu_hypre_structmatrixsetconstante(fmatrix, fnument,
     1                                           fentries)
      integer ierr
      integer fnument(*)
      integer fentries(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetConstantEn(fmatrix, fnument,
     1                                     fentries, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetconstantentries: error =', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetconstantvalues
!******************************************
      subroutine fnalu_hypre_structmatrixsetconstantv(fmatrix,
     1                                           fnumsindx, fsindx,
     2                                           fvals)
      integer ierr
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetConstantVa(fmatrix, fnumsindx, 
     1                                         fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetconstantvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixaddtovalues
!******************************************
      subroutine fnalu_hypre_structmatrixaddtovalues(fmatrix, fgrdindx,
     1                                          fnumsindx, fsindx,
     2                                          fvals)
      integer ierr
      integer fgrdindx(*)
      integer fnumsindx(*)
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixAddToValues(fmatrix, fgrdindx,
     1                                   fnumsindx, fsindx, fvals,
     2                                   ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixaddtovalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixaddtoboxvalues
!******************************************
      subroutine fnalu_hypre_structmatrixaddtoboxvalues(fmatrix, filower,
     1                                             fiupper, fnumsindx,
     2                                             fsindx, fvals)
      integer ierr
      integer filower(*)
      integer fiupper(*)
      integer fnumsindx
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixAddToBoxValues(fmatrix, filower, fiupper,
     1                                      fnumsindx, fsindx, fvals,
     2                                      ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixaddtovalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixaddtoconstantvalues
!******************************************
      subroutine fnalu_hypre_structmatrixaddtoconstant(fmatrix, fnumsindx,
     2                                            fsindx, fvals)
      integer ierr
      integer fnumsindx
      integer fsindx(*)
      double precision fvals(*)
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetConstantVa(fmatrix, fnumsindx, 
     1                                         fsindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixaddtoconstantvalues: error = ',
     1                             ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixassemble
!******************************************
      subroutine fnalu_hypre_structmatrixassemble(fmatrix)
      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixAssemble(fmatrix, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixassemble: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetnumghost
!******************************************
      subroutine fnalu_hypre_structmatrixsetnumghost(fmatrix, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetNumGhost(fmatrix, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetnumghost: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixgetgrid
!******************************************
      subroutine fnalu_hypre_structmatrixgetgrid(fmatrix, fgrid)
      integer ierr
      integer*8 fmatrix
      integer*8 fgrid

      call NALU_HYPRE_StructMatrixGetGrid(fmatrix, fgrid, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixgetgrid: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixsetsymmetric
!******************************************
      subroutine fnalu_hypre_structmatrixsetsymmetric(fmatrix, fsymmetric)
      integer ierr
      integer fsymmetric
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixSetSymmetric(fmatrix, fsymmetric, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixsetsymmetric: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixprint
!******************************************
      subroutine fnalu_hypre_structmatrixprint(fmatrix, fall)
      integer ierr
      integer fall
      integer*8 fmatrix

      call NALU_HYPRE_StructMatrixPrint(fmatrix, fall, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixprint: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structmatrixmatvec
!******************************************
      subroutine fnalu_hypre_structmatrixmatvec(falpha, fA, fx, fbeta, fy)
      integer ierr
      integer falpha
      integer fbeta
      integer*8 fA
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_StructMatrixMatvec(falplah, fA, fx, fbeta, fy, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structmatrixmatvec: error = ', ierr
      endif

      return
      end



!**************************************************
!           NALU_HYPRE_StructVector routines
!**************************************************

!******************************************
!      fnalu_hypre_structvectorcreate
!******************************************
      subroutine fnalu_hypre_structvectorcreate(fcomm, fgrid, fvector)
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fvector

      call NALU_HYPRE_StructVectorCreate(fcomm, fgrid, fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorcreate: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectordestroy
!******************************************
      subroutine fnalu_hypre_structvectordestroy(fvector)
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_StructVectorDestroy(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectordestroy: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorinitialize
!******************************************
      subroutine fnalu_hypre_structvectorinitialize(fvector)
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_StructVectorInitialize(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorinitialize: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorsetvalues
!******************************************
      subroutine fnalu_hypre_structvectorsetvalues(fvector, fgridindx,
     1                                          fvals)
      integer ierr
      integer fgridindx(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorSetValues(fvector, fgridindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorsetvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorsetboxvalues
!******************************************
      subroutine fnalu_hypre_structvectorsetboxvalues(fvector, flower,
     1                                           fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorSetBoxValues(fvector, flower, fupper,
     1                                    fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorsetboxvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorsetconstantvalues
!******************************************
      subroutine fnalu_hypre_structvectorsetconstantv(fvector, fvals)
      integer ierr
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorSetConstantVa(fvector, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorsetconstantvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectoraddtovalues
!******************************************
      subroutine fnalu_hypre_structvectoraddtovalues(fvector, fgrdindx,
     1                                          fvals)
      integer ierr
      integer fgrdindx(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorAddToValues(fvector, fgrdindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectoraddtovalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectoraddtoboxvalues
!******************************************
      subroutine fnalu_hypre_structvectoraddtoboxvalu(fvector, flower, 
     1                                             fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorAddToBoxValue(fvector, flower, fupper,
     1                                      fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectoraddtoboxvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorscalevalues
!******************************************
      subroutine fnalu_hypre_structvectorscalevalues(fvector, ffactor)
      integer ierr
      double precision ffactor
      integer*8 fvector

      call NALU_HYPRE_StructVectorScaleValues(fvector, ffactor, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorscalevalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorgetvalues
!******************************************
      subroutine fnalu_hypre_structvectorgetvalues(fvector, fgrdindx,
     1                                          fvals)
      integer ierr
      integer fgrdindx(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorGetValues(fvector, fgrdindx, fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorgetvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorgetboxvalues
!******************************************
      subroutine fnalu_hypre_structvectorgetboxvalues(fvector, flower, 
     1                                           fupper, fvals)
      integer ierr
      integer flower(*)
      integer fupper(*)
      double precision fvals(*)
      integer*8 fvector

      call NALU_HYPRE_StructVectorGetBoxValues(fvector, flower, fupper,
     1                                    fvals, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorgetboxvalues: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorassemble
!******************************************
      subroutine fnalu_hypre_structvectorassemble(fvector)
      integer ierr
      integer*8 fvector

      call NALU_HYPRE_StructVectorAssemble(fvector, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorassemble: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorsetnumghost
!******************************************
      subroutine fnalu_hypre_structvectorsetnumghost(fvector, fnumghost)
      integer ierr
      integer fnumghost
      integer*8 fvector

      call NALU_HYPRE_StructVectorSetNumGhost(fvector, fnumghost, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorsetnumghost: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorcopy
!******************************************
      subroutine fnalu_hypre_structvectorcopy(fx, fy)
      integer ierr
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_StructVectorCopy(fx, fy, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorcopy: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorgetmigratecommpkg
!******************************************
      subroutine fnalu_hypre_structvectorgetmigrateco(ffromvec, ftovec, 
     1                                                fcommpkg)
      integer ierr
      integer*8 ffromvec
      integer*8 ftovec
      integer*8 fcommpkg

      call NALU_HYPRE_StructVectorGetMigrateCom(ffromvec, ftovec, fcommpkg,
     1                                     ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorgetmigratecommpkg: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectormigrate
!******************************************
      subroutine fnalu_hypre_structvectormigrate(fcommpkg, ffromvec,
     1                                        ftovec)
      integer ierr
      integer*8 ffromvec
      integer*8 ftovec
      integer*8 fcommpkg

      call NALU_HYPRE_StructVectorMigrate(fcommpkg, ffromvec, ftovec, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectormigrate: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_commpkgdestroy
!******************************************
      subroutine fnalu_hypre_commpkgdestroy(fcommpkg)
      integer ierr
      integer*8 fcommpkg

      call NALU_HYPRE_DestroyCommPkg(fcommpkg, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_commpkgdestroy: error = ', ierr
      endif

      return
      end

!******************************************
!      fnalu_hypre_structvectorprint
!******************************************
      subroutine fnalu_hypre_structvectorprint(fvector, fall)
      integer ierr
      integer fall
      integer*8 fvector

      call NALU_HYPRE_StructVectorPrint(fvector, fall, ierr)
      if (ierr .ne. 0) then
         print *, 'fnalu_hypre_structvectorprint: error = ', ierr
      endif

      return
      end
