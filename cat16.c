// map an entire file to heap memory
// and dump them every 16-bit(2-byte)
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "bf16.h"

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

  // ファイルの内容をポインタを使って(2バイトごとに)読み出し
  for (off_t i = 0; i < sb.st_size; i += 2) {
    uint16_t bits8_L = 0xFF & ((uint32_t)file_in_memory[i+0]);
    uint16_t bits8_H = 0xFF & ((uint32_t)file_in_memory[i+1]);
    uint16_t bits16 = 0xFFFF & ((bits8_H << 8) | bits8_L); // little endian

    float x_fp32 = BF16_to_FP32(UINT16_to_BF16((uint16_t)bits16));
    char* space = " ";
    if (x_fp32 < 0.0) space = "";
    printf("%08llX, %04X, %s%e\n", i/2, bits16, space, x_fp32);
  }


  // メモリマッピングを解除
  if (munmap(file_in_memory, sb.st_size) == -1) {
    perror("munmap");
  }

  // ファイルを閉じる
  close(fd);

  return 0;
}
