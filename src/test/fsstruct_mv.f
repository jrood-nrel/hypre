!     Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
!     NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

!***************************************************************************
! NALU_HYPRE_SStruct fortran interface
!***************************************************************************


!***************************************************************************
!              NALU_HYPRE_SStructGraph routines
!***************************************************************************

!-------------------------------------------------------------------------
! NALU_HYPRE_SStructGraphCreate
!-------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphcreate(fcomm, fgrid, fgraphptr)
     
      integer ierr
      integer fcomm
      integer*8 fgrid
      integer*8 fgraphptr

      call NALU_HYPRE_SStructGraphCreate(fcomm, fgrid, fgraphptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGraphDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphdestroy(fgraph)
      
      integer ierr
      integer*8 fgraph

      call NALU_HYPRE_SStructGraphDestroy(fgraph, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
! NALU_HYPRE_SStructGraphSetStencil
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphsetstencil(fgraph, fpart, fvar, 
     1                                         fstencil)

      integer ierr
      integer part
      integer var
      integer*8 fgraph
      integer*8 fstencil

      call NALU_HYPRE_SStructGraphSetStencil(fgraph, fpart, fvar, fstencil, 
     1                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphsetstencil error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructGraphAddEntries-
!    THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphaddentries(fgraph, fpart, findex, 
     1                                         fvar, fto_part,
     1                                         fto_index, fto_var)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fto_part
      integer fto_index
      integer fto_var
      integer*8 fgraph

      call NALU_HYPRE_SStructGraphAddEntries(fgraph, fpart, findex, fvar,
     1                                  fto_part, fto_index, fto_var, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphaddedntries error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructGraphAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphassemble(fgraph)
     
      integer ierr
      integer*8 fgraph

      call NALU_HYPRE_SStructGraphAssemble(fgraph, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphassemble error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructGraphSetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgraphsetobjecttyp(fgraph, ftype)
                                                 
      integer ierr
      integer ftype
      integer*8 fgraph

      call NALU_HYPRE_SStructGraphSetObjectType(fgraph, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgraphsetobjecttype error = ', ierr
      endif

      return
      end




!***************************************************************************
!              NALU_HYPRE_SStructGrid routines
!***************************************************************************

!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructGridCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridcreate(fcomm, fndim, fnparts,
     1                                    fgridptr)
                                         
      integer ierr
      integer fcomm
      integer fndim
      integer fnparts
      integer*8 fgridptr

      call NALU_HYPRE_SStructGridCreate(fcomm, fndim, fnparts, fgridptr, 
     1                             ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgriddestroy(fgrid)

      integer ierr
      integer*8 fgrid

      call NALU_HYPRE_SStructGridDestroy(fgrid, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgriddestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridSetExtents
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridsetextents(fgrid, fpart, filower, 
     1                                        fiupper)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer*8 fgrid

      call NALU_HYPRE_SStructGridSetExtents(fgrid, fpart, filower, fiupper, 
     1                                 ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridsetextents error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridSetVariables
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridsetvariables(fgrid, fpart, fnvars, 
     1                                          fvartypes)

      integer ierr
      integer fpart
      integer fnvars
      integer*8 fgrid
      integer*8 fvartypes

      call NALU_HYPRE_SStructGridSetVariables(fgrid, fpart, fnvars, 
     1                                   fvartypes, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridsetvariables error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridAddVariables
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridaddvariables(fgrid, fpart, findex,
     1                                          fnvars, fvartypes)

      integer ierr
      integer fpart
      integer findex
      integer fnvars
      integer*8 fgrid
      integer*8 fvartypes

      call NALU_HYPRE_SStructGridAddVariables(fgrid, fpart, findex, fnvars,
     1                                   fvartypes, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridaddvariables error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridSetNeighborBox
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridsetneighborbo(fgrid, fpart, filower,
     1                                            fiupper, fnbor_part,
     2                                            fnbor_ilower,
     3                                            fnbor_iupper,
     4                                            findex_map)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fnbor_part
      integer fnbor_ilower
      integer fnbor_iupper
      integer findex_map
      integer*8 fgrid

      call NALU_HYPRE_SStructGridSetNeighborBox(fgrid, fpart, filower, 
     1                                     fiupper, fnbor_part,
     2                                     fnbor_ilower, fnbor_iupper,
     3                                     findex_map, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridsetneighborbox error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridassemble(fgrid)

      integer ierr
      integer*8 fgrid

      call NALU_HYPRE_SStructGridAssemble(fgrid, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridassemble error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridSetPeriodic
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridsetperiodic(fgrid, fpart, fperiodic)

      integer ierr
      integer fpart
      integer fperiodic
      integer*8 fgrid

      call NALU_HYPRE_SStructGridSetPeriodic(fgrid, fpart, fperiodic, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridsetperiodic error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructGridSetNumGhost
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructgridsetnumghost(fgrid, fnum_ghost)

      integer ierr
      integer fnumghost
      integer*8 fgrid

      call NALU_HYPRE_SStructGridSetNumGhost(fgrid, fnum_ghost, ierr)       

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructgridsetnumghost error = ', ierr
      endif

      return
      end




!***************************************************************************
!              NALU_HYPRE_SStructMatrix routines
!***************************************************************************

!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixcreate(fcomm, fgraph, fmatrix_ptr)

      integer ierr
      integer fcomm
      integer*8 fgraph
      integer*8 fmatrix_ptr

      call NALU_HYPRE_SStructMatrixCreate(fcomm, fgraph, fmatrix_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixdestroy(fmatrix)

      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixDestroy(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixdestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixInitialize
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixinitialize(fmatrix)

      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixInitialize(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixinitialize error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixSetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixsetvalues(fmatrix, fpart, findex, 
     1                                         fvar, fnentries, 
     2                                         fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixSetValues(fmatrix, fpart, findex, fvar, 
     1                                  fnentries, fentries, fvalues, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixsetvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixSetBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixsetboxvalue(fmatrix, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fnentries, 
     3                                            fentries, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixSetBoxValues(fmatrix, fpart, filower, 
     1                                     fiupper, fvar, fnentries, 
     2                                     fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixsetboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixGetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixgetvalues(fmatrix, fpart, findex, 
     1                                         fvar, fnentries, 
     2                                         fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixGetValues(fmatrix, fpart, findex, fvar, 
     1                                  fnentries, fentries, fvalues, 
     2                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixgetvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixGetBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixgetboxvalue(fmatrix, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fnentries,
     3                                            fentries, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixGetBoxValues(fmatrix, fpart, filower, 
     1                                     fiupper, fvar, fnentries, 
     2                                     fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixgetboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixAddToValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixaddtovalues(fmatrix, fpart, findex,
     1                                           fvar, fnentries, 
     2                                           fentries, fvalues)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixAddToValues(fmatrix, fpart, findex, fvar, 
     1                                    fnentries, fentries, fvalues, 
     2                                    ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixaddtovalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixAddToBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixaddtoboxval(fmatrix, fpart, 
     1                                             filower, fiupper,
     2                                             fvar, fnentries, 
     3                                             fentries, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer fnentries
      integer fentries
      integer*8 fmatrix
      double precision fvalues

      call NALU_HYPRE_SStructMatrixAddToBoxValu(fmatrix, fpart, filower, 
     1                                       fiupper, fvar, fnentries, 
     2                                       fentries, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixaddtoboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixassemble(fmatrix)

      integer ierr
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixAssemble(fmatrix, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixassemble error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixSetSymmetric
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixsetsymmetri(fmatrix, fpart, fvar,
     1                                            fto_var, fsymmetric)

      integer ierr
      integer fpart
      integer fvar
      integer fto_var
      integer fsymmetric
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixSetSymmetric(fmatrix, fpart, fvar, 
     1                                     fto_var, fsymmetric, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixsetsymmetric error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixSetNSSymmetric
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixsetnssymmet(fmatrix, fsymmetric)

      integer ierr
      integer fsymmetric
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixSetNSSymmetr(fmatrix, fsymmetric, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixsetnssymmetric error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixSetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixsetobjectty(fmatrix, ftype)

      integer ierr
      integer ftype
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixSetObjectTyp(fmatrix, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixsetobjecttype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixGetObject
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixgetobject(fmatrix, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixGetObject(fmatrix, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixgetobject error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixGetObject2
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixgetobject2(fmatrix, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fmatrix

      call NALU_HYPRE_SStructMatrixGetObject2(fmatrix, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixgetobject2 error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixprint(ffilename, fmatrix, fall)

      integer ierr
      integer fall
      integer*8 fmatrix
      character*(*) ffilename

      call NALU_HYPRE_SStructMatrixPrint(ffilename, fmatrix, fall, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixprint error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructMatrixMatvec
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructmatrixmatvec(falpha, fA, fx, fbeta, fy)

      integer ierr
      integer*8 fA
      integer*8 fx
      integer*8 fy
      double precision falpha
      double precision fbeta

      call NALU_HYPRE_SStructMatrixMatvec(falpha, fA, fx, fbeta, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructmatrixmatvec error = ', ierr
      endif

      return
      end




!***************************************************************************
!              NALU_HYPRE_SStructStencil routines
!***************************************************************************

!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructStencilCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructstencilcreate(fndim, fsize, fstencil_ptr)

      integer ierr
      integer fndim
      integer fsize
      integer*8 fstencil_ptr

      call NALU_HYPRE_SStructStencilCreate(fndim, fsize, fstencil_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructstencilcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructStencilDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructstencildestroy(fstencil)

      integer ierr
      integer*8 fstencil

      call NALU_HYPRE_SStructStencilDestroy(fstencil, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructstencildestroy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructStencilSetEntry
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructstencilsetentry(fstencil, fentry, 
     1                                         foffset, fvar)

      integer ierr
      integer fentry
      integer foffset
      integer fvar
      integer*8 fstencil

      call NALU_HYPRE_SStructStencilSetEntry(fstencil, fentry, foffset, fvar,
     1                                  ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructstencilsetentry error = ', ierr
      endif

      return
      end




!***************************************************************************
!              NALU_HYPRE_SStructVector routines
!***************************************************************************

!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorCreate
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorcreate(fcomm, fgrid, fvector_ptr)

      integer ierr
      integer fcomm
      integer*8 fvector_ptr
      integer*8 fgrid

      call NALU_HYPRE_SStructVectorCreate(fcomm, fgrid, fvector_ptr, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorcreate error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!  NALU_HYPRE_SStructVectorDestroy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectordestroy(fvector)

      integer ierr
      integer*8 fvector

      call NALU_HYPRE_SStructVectorDestroy(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectordestroy error = ', ierr
      endif

      return
      end


!---------------------------------------------------------
!  NALU_HYPRE_SStructVectorInitialize
!---------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorinitialize(fvector)

      integer ierr
      integer*8 fvector
   
      call NALU_HYPRE_SStructVectorInitialize(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorinitialize error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorSetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorsetvalues(fvector, fpart, findex, 
     1                                         fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call NALU_HYPRE_SStructVectorSetValues(fvector, fpart, findex, fvar,
     1                                  fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorsetvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorSetBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorsetboxvalue(fvector, fpart,
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call NALU_HYPRE_SStructVectorSetBoxValues(fvector, fpart, filower, 
     1                                     fiupper, fvar, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorsetboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorAddToValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectoraddtovalues(fvector, fpart, findex,
     1                                           fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call NALU_HYPRE_SStructVectorAddToValues(fvector, fpart, findex, fvar, 
     1                                    fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectoraddtovalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorAddToBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectoraddtoboxval(fvector, fpart,
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)
      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call NALU_HYPRE_SStructVectorAddToBoxValu(fvector, fpart, filower,
     1                                       fiupper, fvar, fvalues,
     2                                       ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectoraddtoboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorAssemble
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorassemble(fvector)

      integer ierr
      integer*8 fvector

      call NALU_HYPRE_SStructVectorAssemble(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorassemble error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorGather
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorgather(fvector)

      integer ierr
      integer*8 fvector

      call NALU_HYPRE_SStructVectorGather(fvector, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorgather error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorGetValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorgetvalues(fvector, fpart, findex, 
     1                                         fvar, fvalue)

      integer ierr
      integer fpart
      integer findex
      integer fvar
      integer*8 fvector
      double precision fvalue

      call NALU_HYPRE_SStructVectorGetValues(fvector, fpart, findex, fvar, 
     1                                  fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorgetvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorGetBoxValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorgetboxvalue(fvector, fpart, 
     1                                            filower, fiupper, 
     2                                            fvar, fvalues)

      integer ierr
      integer fpart
      integer filower
      integer fiupper
      integer fvar
      integer*8 fvector
      double precision fvalues

      call NALU_HYPRE_SStructVectorGetBoxValues(fvector, fpart, filower,
     1                                     fiupper, fvar, fvalues, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorgetboxvalues error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorSetConstantValues
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorsetconstant(fvector, fvalue)

      integer ierr
      integer*8 fvector
      double precision fvalue

      call NALU_HYPRE_SStructVectorSetConstantV(fvector, fvalue, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorsetconstantvalues error = ',
     1                                       ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorSetObjectType
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorsetobjectty(fvector, ftype)

      integer ierr
      integer ftype
      integer*8 fvector

      call NALU_HYPRE_SStructVectorSetObjectTyp(fvector, ftype, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorsetobjecttype error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorGetObject
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorgetobject(fvector, fobject)

      integer ierr
      integer*8 fobject
      integer*8 fvector

      call NALU_HYPRE_SStructVectorGetObject(fvector, fobject, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorgetobject error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorPrint
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorprint(ffilename, fvector, fall)

      integer ierr
      integer fall
      integer*8 fvector
      character*(*) ffilename

      call NALU_HYPRE_SStructVectorPrint(ffilename, fvector, fall, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorprint error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorCopy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorcopy(fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy

      call NALU_HYPRE_SStructVectorCopy(fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorcopy error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructVectorScale
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructvectorscale(falpha, fy)

      integer ierr
      integer*8 fy
      double precision falpha

      call NALU_HYPRE_SStructVectorScale(falpha, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructvectorscale error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructInnerProd
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructinnerprod(fx, fy, fresult)

      integer ierr
      integer*8 fx
      integer*8 fy
      double precision fresult

      call NALU_HYPRE_SStructInnerProd(fx, fy, fresult, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructinnerprod error = ', ierr
      endif

      return
      end


!--------------------------------------------------------------------------
!   NALU_HYPRE_SStructAxpy
!--------------------------------------------------------------------------
      subroutine fnalu_hypre_sstructaxpy(falpha, fx, fy)

      integer ierr
      integer*8 fx
      integer*8 fy
      double precision falpha

      call NALU_HYPRE_SStructAxpy(falpha, fx, fy, ierr)

      if (ierr .ne. 0) then
         print *, ' fnalu_hypre_sstructaxpy error = ', ierr
      endif

      return
      end
