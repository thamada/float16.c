CC=gcc -Wall

build:
	${CC} test.c -o run



c: clean

clean:
	rm -f *~ .*~ run


