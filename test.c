#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include "bf16.h"

/**
  log(2.**23) / log(10.0) = 6.9
  log(2.**7)  / log(10.0) = 2.1
*/

int main(int argc, char *argv[]) {
  float x = 7.7777777777;
  degima_bf16_t y = FP32_to_BF16(x);
  float z = BF16_to_FP32(y);

  printf("%.8e\n", x);
  printf("%.4e\n", z);
  printf("Hello\n");

  return 0;
}

