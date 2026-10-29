# Builds tau_decay against a TAUOLA++ install (configure --prefix=$TAUOLA_DIR,
# HepMC not needed: the HEPEVT interface is used).  macOS: the Fortran core
# needs libgfortran, found through the gfortran on PATH.
TAUOLA_DIR ?= $(HOME)/opt/tauola
CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2
GFORTRAN_LIB := $(dir $(shell gfortran -print-file-name=libgfortran.dylib))

tau_decay: tau_decay.cxx
	$(CXX) $(CXXFLAGS) -I$(TAUOLA_DIR)/include $< -o $@ \
	    -L$(TAUOLA_DIR)/lib -lTauolaCxxInterface -lTauolaHEPEVT -lTauolaFortran \
	    -L$(GFORTRAN_LIB) -lgfortran \
	    -Wl,-rpath,$(TAUOLA_DIR)/lib -Wl,-rpath,$(GFORTRAN_LIB)

clean:
	rm -f tau_decay

.PHONY: clean
