/**
 * Copyright (c) 2022 Raoul Harel
 * All rights reserved
 */

#include <iostream>

#include "cudu.h"

void cudu::abort_if_false(
    const bool status,
    const std::string& file,
    const unsigned line,
    const std::string& message) {

  if (!status) {
    std::cerr << "assertion failed at " << file << " line " << line << ": "
              << message << std::endl;

    std::abort();
  }
}

void cudu::abort_on_error(
    const cudaError_t status,
    const std::string& file,
    const unsigned line,
    const std::string& message) {

  if (status != cudaSuccess) {
    std::cerr << "assertion failed at " << file << " line " << line << ": "
              << message << ": " << cudaGetErrorName(status) << "(" << status
              << "): " << cudaGetErrorString(status) << std::endl;

    std::abort();
  }
}

unsigned cudu::Workload::max_threads_per_block() {
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  return device_props.maxThreadsPerBlock;
}

cudu::Workload cudu::Workload::for_jobs(const unsigned nr_jobs) {
  Workload workload;
  workload.nr_threads = max_threads_per_block() / 2;
  workload.nr_blocks =
      unsigned(std::ceil(float(nr_jobs) / workload.nr_threads));
  return workload;
}
