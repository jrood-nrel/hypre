# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

PDFLATEX = pdflatex --interaction=nonstopmode

##################################################################
# Targets
##################################################################

# all: html pdf
all: pdf

install:

clean:
	@/bin/rm -fr html latex xml

distclean: clean

##################################################################
# Rules
##################################################################

html:
	doxygen conf.doxygen

pdf: html
	@cd latex; ${PDFLATEX} refman; ${PDFLATEX} refman

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$(NALU_HYPRE_RELEASE_VERSION)}" > version.tex; \
	echo "\def\HYPREVersionDate{$(NALU_HYPRE_RELEASE_DATE)}" >> version.tex
