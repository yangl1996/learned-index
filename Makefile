CC=clang++ -march=native
CFLAGS=-I/usr/local/include/eigen3

all:
	$(CC) main.cpp -o learned-index $(CFLAGS)

.PHONY: clean

clean:
	rm -f learned-index

