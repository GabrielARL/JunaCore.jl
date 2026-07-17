CC ?= cc
CFLAGS ?= -O
CPPFLAGS ?=
LDFLAGS ?=

TOOLS_DIR ?= ..
COMMON := mod2sparse.c mod2dense.c mod2convert.c rcode.c alloc.c intio.c open.c

.PHONY: all clean

all: $(TOOLS_DIR)/make-ldpc $(TOOLS_DIR)/make-gen $(TOOLS_DIR)/print-pchk

$(TOOLS_DIR)/make-ldpc: make-ldpc.c rand.c distrib.c $(COMMON) randfile
	$(CC) $(CPPFLAGS) $(CFLAGS) -DRAND_FILE='"src/randfile"' \
		make-ldpc.c rand.c distrib.c $(COMMON) $(LDFLAGS) -lm -o $@

$(TOOLS_DIR)/make-gen: make-gen.c $(COMMON)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		make-gen.c $(COMMON) $(LDFLAGS) -lm -o $@

$(TOOLS_DIR)/print-pchk: print-pchk.c rand.c $(COMMON) randfile
	$(CC) $(CPPFLAGS) $(CFLAGS) -DRAND_FILE='"src/randfile"' \
		print-pchk.c rand.c $(COMMON) $(LDFLAGS) -lm -o $@

clean:
	rm -f $(TOOLS_DIR)/make-ldpc $(TOOLS_DIR)/make-gen $(TOOLS_DIR)/print-pchk
