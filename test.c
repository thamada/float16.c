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

long time_in_ms() {
  // return time in milliseconds
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void tokei(long t0) {
  long tt = time_in_ms() - t0;
  double t = tt/1000.;
  static unsigned flag = 0;
  if (0 == tt % 1000) {
    if (1 == flag) {
      printf("%.3f sec\n", t);
      flag = 0;
    }
  } else {
    flag = 1;
  }
  return;
}

void rand_test() {
  srand(time(NULL));

  double err_max = 0.;
  long t0 = time_in_ms();
  unsigned long long i = 0ULL;
  while (1) {
    float x = rand() / (RAND_MAX * 1.);
    degima_bf16_t y = FP32_to_BF16(x);
    float z = BF16_to_FP32(y);
    double err_rel = ((double)fabs(z-x)) / ((double)x);
    if (err_rel > err_max) {
      printf("%016llx: %.8e, %.4e, %e, %e\n", i, x, z, err_rel, err_max);
      err_max = err_rel;
    }

    tokei(t0);

    i++;
  }

  return;
}


int main(int argc, char *argv[]) {

  rand_test();

  return 0;
}

