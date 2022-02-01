#

TARGET := qtest qbench mpiqtest mpiqbench mpicnot

ifdef usefcc
    CC := mpifcc
    CXX := mpiFCC
else
    CC := mpicc
    CXX := mpiCC
endif

BUILDDIR := build
BUILDDIR := $(abspath $(BUILDDIR))

ifdef usefcc
CFLAGS   += -Nclang -Knolargepage 
CXXFLAGS += -Nclang -Knolargepage -ffj-fjprof
endif

CFLAGS   += -v -I../include -fopenmp -lpthread -std=c14
CXXFLAGS += -Wno-inconsistent-missing-override -I../include -fopenmp -lpthread -std=c++14
LDFLAGS := -L../lib -lvqcsim_static -lcppsim_static -lcsim_static

ifdef debug
    CFLAGS   += -O0 -g
    CXXFLAGS += -O0 -g
    DEBUG_FLAG = -DDEBUG
else
ifdef usefcc
    CFLAGS   += -Kfast -DNDEBUG
    CXXFLAGS += -Kfast -DNDEBUG
else
    CFLAGS   += -O3 -DNDEBUG
    CXXFLAGS += -O3 -DNDEBUG
endif
    DEBUG_FLAG =
endif
CFLAGS   += -D_USE_MPI
CXXFLAGS += -D_USE_MPI

#$(warning MAKE = $(MAKE))
$(warning CXX = $(CXX))
$(warning CXXFLAGS = $(CXXFLAGS))

#OLD_SHELL := $(SHELL)
#SHELL = $(warning [Making: $@]   [Dependencies: $^]   [Changed: $?])$(OLD_SHELL)

.PHONY : all clean test $(POCL_DIR)
.DEFAULT : all

all : $(TARGET)

% : %.cpp
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAG) $@.cpp $(LDFLAGS) -o $@

clean :
	rm -rf $(BUILDDIR) $(TARGET)

#### TESTS ####
TEST_DIRS = $(wildcard test/*)
.PHONY : $(TEST_DIRS)

test : all $(TEST_DIRS)

check : test $(TEST_DIRS)

$(TEST_DIRS) : FORCE
	$(MAKE) -C $@ $(MAKECMDGOALS)

FORCE:

