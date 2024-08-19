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
#include "fp16.h"

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

double wallclock(long t0) {
  long tt = time_in_ms() - t0;
  double t = tt/1000.;
  static unsigned flag = 0;
  if (0 == tt % 1000) {
    if (1 == flag) {
      fprintf(stderr, "%.3f sec\n", t);
      flag = 0;
    }
  } else {
    flag = 1;
  }
  return t;
}

float kernel_bf16(float x) {
    degima_bf16_t y = FP32_to_BF16(x);
    return BF16_to_FP32(y);
}

float kernel_fp16(float x) {
    degima_fp16_t y = FP32_to_FP16(x);
    return FP16_to_FP32(y);
}

//#define KERNEL(x) kernel_fp16(x)
#define KERNEL(x) kernel_bf16(x)

void test() {
  srand(time(NULL));

  double err_max = 0.;
  long t0 = time_in_ms();
  unsigned long long i = 0ULL;
  while (1) {
    float x = rand() / (RAND_MAX * 1.);
    float z = KERNEL(x);
    double err_rel = ((double)fabs(z-x)) / ((double)x);
    if (err_rel > err_max) {
      err_max = err_rel;
      printf("%016llx: %.8e, %.4e, %e\n", i, x, z, err_max);
    }
    double wallclocktime_sec = wallclock(t0);
    if(wallclocktime_sec > 10.) break;
    i++;
  }

  return;
}


int main(int argc, char *argv[]) {
  test();
  return 0;
}

