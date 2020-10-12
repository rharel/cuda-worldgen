#include <iostream>

#include "cudu.h"

void cudu::abort_on_error(
    const cudaError_t status, 
    const std::string& file,
    const unsigned line,
    const std::string& message)
{
    if (status != cudaSuccess)
    {
        std::cerr
            << "assertion failed at " << file << " line " << line << ": "
            << message << ": " 
            << cudaGetErrorName(status) << "(" << status << "): "
            << cudaGetErrorString(status) << std::endl;

        std::abort();
    }
}

cudu::Workload cudu::Workload::for_jobs(
    const unsigned nr_jobs,
    const unsigned nr_threads_max)
{
    for (size_t i = 1; i <= sqrt(nr_jobs); ++i)
    {
        if (nr_jobs % i == 0 && 
            nr_jobs / i <= nr_threads_max)
        {
            Workload workload;
            workload.nr_blocks = unsigned(i);
            workload.nr_threads = unsigned(nr_jobs / i);
            return workload;
        }
    }
    Workload workload;
    workload.nr_threads = unsigned(std::min(nr_threads_max, nr_jobs));
    workload.nr_blocks = unsigned(std::ceil(float(nr_jobs) / workload.nr_threads));
    return workload;
}
