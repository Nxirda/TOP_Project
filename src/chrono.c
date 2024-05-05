#define _GNU_SOURCE

#include "chrono.h"

//
duration_t
chrono_elapsed (chrono_t self)
{

  f64 secs = (f64)(self.stop.tv_sec - self.start.tv_sec);
  i32 nanos = (i32)(self.stop.tv_nsec - self.start.tv_nsec);

  return (duration_t){
    .secs = secs,
    .nanos = nanos,
  };
}

//
f64
duration_as_s_f64 (duration_t self)
{
  return self.secs + (f64)self.nanos * 1.0e-9;
}

//
f64
duration_as_ms_f64 (duration_t self)
{
  return self.secs * 1.0e+3 + (f64)self.nanos * 1.0e-6;
}

//
f64
duration_as_us_f64 (duration_t self)
{
  return self.secs * 1.0e+6 + (f64)self.nanos * 1.0e-3;
}

//
f64
duration_as_ns_f64 (duration_t self)
{
  return self.secs * 1.0e+9 + (f64)self.nanos;
}
