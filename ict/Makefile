#
#
#

CC := mpifcc
CXX := mpiFCC

BUILDDIR := build
BUILDDIR := $(abspath $(BUILDDIR))

CFLAGS   := -Nclang -v -I../include -fopenmp -Knolargepage -lpthread
CXXFLAGS := -Nclang -Wno-inconsistent-missing-override -I../include -fopenmp -Knolargepage -lpthread

CFLAGS   += -std=c14
CXXFLAGS += -std=c++14

LDFLAGS := -L../lib -lvqcsim_static -lcppsim_static -lcsim_static

ifdef debug
    CFLAGS   += -O0 -g
    CXXFLAGS += -O0 -g
    DEBUG_FLAG = -DDEBUG
else
    CFLAGS   += -Kfast -DNDEBUG
    CXXFLAGS += -Kfast -DNDEBUG
    DEBUG_FLAG =
endif

$(warning MAKE = $(MAKE))
$(warning CXX = $(CXX))
$(warning CXXFLAGS = $(CXXFLAGS))

#OLD_SHELL := $(SHELL)
#SHELL = $(warning [Making: $@]   [Dependencies: $^]   [Changed: $?])$(OLD_SHELL)

TARGET := qtest qbench

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

