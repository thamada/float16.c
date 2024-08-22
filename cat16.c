// map an entire file to heap memory
// and dump them every 16-bit(2-byte)
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // ファイルを開く
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    perror("open");
    exit(EXIT_FAILURE);
  }

  // ファイルサイズを取得
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    close(fd);
    exit(EXIT_FAILURE);
  }

  // ファイル全体をメモリにマップ
  char *file_in_memory = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (file_in_memory == MAP_FAILED) {
    perror("mmap");
    close(fd);
    exit(EXIT_FAILURE);
  }

  // ファイルの内容をポインタを使って読み出し
  off_t nbytes = sizeof(uint16_t);

  for (off_t i = 0; i < sb.st_size; i += nbytes) {
    uint16_t bits16 = 0;
    for (off_t ic = 0; ic < nbytes;  ic++) {
      uint32_t bits8 = 0xFF & ((uint32_t)file_in_memory[i + ic]);
      bits16 = bits16 | (bits8 << (ic*8)); // little endian
    }
    union { float f; uint32_t i; } u;
    u.i = ((uint32_t)bits16) << 16; // bfloat16 format
    char* space = " ";
    if (u.f < 0.0) space = "";
    printf("%08llX, %04X, %s%e\n", i/nbytes, bits16, space, u.f);
  }

  // メモリマッピングを解除
  if (munmap(file_in_memory, sb.st_size) == -1) {
    perror("munmap");
  }

  // ファイルを閉じる
  close(fd);

  return 0;
}
