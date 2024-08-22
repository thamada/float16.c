CC=gcc -Wall

build:
	${CC} test.c -o run
	${CC} cat16.c -o cat16


c: clean

clean:
	rm -f *~ .*~ run cat16


