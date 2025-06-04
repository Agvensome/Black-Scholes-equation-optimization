# Files
EXEC := af
CXX := g++
SRC := $(wildcard ./src/*.cc)
OBJ := $(patsubst ./src/%.cc,./build/%.o,$(SRC))
LIB := $(patsubst ./src/%.cc, ./lib/lib%.so, $(SRC))
INCLUDE_DIR := -I./include
LIB_DIR := -L./lib
LIB_DIR_NO_L := /lib
ALL_LIB := $(patsubst ./src/%.cc, -l%, $(SRC))
SEARCH_LIBSO := $(CURDIR)$(LIB_DIR_NO_L)

# Options
ifneq ($(DEBUG_MODE), yes)
	CFLAGS := -O3 -MMD -MP
else
	CFLAGS := -g -MMD -MP
endif

# Rules


all: $(EXEC) $(LIB)

ifeq ($(wildcard $(EXEC)), )
$(EXEC): main.cc $(LIB)
	$(CXX) $(LIB_DIR) $(INCLUDE_DIR) -o $@ $^ $(ALL_LIB)
endif

./lib/lib%.so: ./build/%.o
	$(CXX) -shared -fPIC -o $@ $^

./build/%.o: ./src/%.cc ./include/%.hh
	$(CXX) $(CFLAGS) -fPIC $(INCLUDE_DIR) -c $< -o $@




.PHONY: echo install
echo:
	@echo $(LIB)
	@echo $(ALL_LIB)
	@echo $(SEARCH_LIBSO)

install:
	mkdir -p lib
	mkdir -p build
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(SEARCH_LIBSO)
	
clean:
	rm -f ./build/*.d ./build/*.o ./lib/*.so $(EXEC)
