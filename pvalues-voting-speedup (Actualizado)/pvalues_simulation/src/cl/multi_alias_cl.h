#pragma once

#include "alias_cl.h"

typedef struct {
    small_t n;
    small_t c;
    log_t beta_n;
    size mem;

    alias_pair_cl* pairs;
    log_t* log_p;
    sample_t* result;

    cl_mem result_dev;
    cl_mem pairs_dev;
    cl_mem log_p_dev;
} multi_alias_cl;

static int create_multi_alias_cl(multi_alias_cl* mult, const small_t n, const small_t c, const double* probs, const double beta_n) {
    log_t* log_p = (log_t*)malloc(c*sizeof(log_t));
    for (small_t i=0; i<c; i++) log_p[i] = probs[i] == 0.0 ? 0 : (log_t)(BIGM*log(probs[i]));
    mult->beta_n = (log_t)(BIGM*beta_n);
    mult->log_p = log_p;

    const small_t N = n+1;
    size mem = (c-1)*N*N*sizeof(alias_pair_cl);
    alias_pair_cl* pairs = (alias_pair_cl*)malloc(mem);
    double p_remain = 1.0;
    for (small_t i=0; i<c-1; i++) {
        double pi = probs[i];
        double pir = pi/p_remain;
        for (small_t j=0; j<N; j++) create_binomial_table_cl(&pairs[N*N*i + N*j], j, pir);
        p_remain -= pi;
    }
    mult->pairs = pairs;

    mult->result = (sample_t*)malloc(bytes_output);

    mult->mem = mem;
    mult->n = n;
    mult->c = c;

    cl_int status;

    //memory objects
    mult->result_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_output, NULL, &status);
    mult->pairs_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, mult->mem, NULL, &status);
    mult->log_p_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, c * sizeof(log_t), NULL, &status);

    status = clEnqueueWriteBuffer(queue, mult->log_p_dev, CL_FALSE, 0, c * sizeof(log_t), mult->log_p, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, mult->pairs_dev, CL_FALSE, 0, mult->mem, mult->pairs, 0, NULL, NULL);

    CL_CHECK_STATUS(status, "Copy buffers")

    //kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mult->pairs_dev);
    CL_CHECK_STATUS(status, "arg 0")

    status = clSetKernelArg(kernel, 1, sizeof(cl_short), &n);
    CL_CHECK_STATUS(status, "arg 1")

    status = clSetKernelArg(kernel, 2, sizeof(cl_short), &c);
    CL_CHECK_STATUS(status, "arg 2")

    status = clSetKernelArg(kernel, 3, sizeof(log_t), &mult->beta_n);
    CL_CHECK_STATUS(status, "arg 3")

    status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &mult->result_dev);
    CL_CHECK_STATUS(status, "arg 7")

    status = clSetKernelArg(kernel, 8, sizeof(cl_mem), &mult->log_p_dev);
    CL_CHECK_STATUS(status, "arg 8")

    clFlush(queue);
    clFinish(queue);

    return 0;
}

static int free_multi_alias_cl(const multi_alias_cl* mult) {
    free(mult->pairs);
    free(mult->log_p);
    free(mult->result);

    cl_int status;
    CL_CHECK_ERROR(status,clReleaseMemObject(mult->result_dev))
    CL_CHECK_ERROR(status,clReleaseMemObject(mult->pairs_dev))
    CL_CHECK_ERROR(status,clReleaseMemObject(mult->log_p_dev))

    return 0;
}

static size sample_multi_alias_cl(multi_alias_cl* mult, size trial_min, size trial_max) {
    uint64_t samples = trial_max - trial_min;
    uint64_t step = (samples + threads_cl - 1) / threads_cl;

    cl_int status;

    status = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &trial_min);
    CL_CHECK_STATUS(status, "arg 4")

    status = clSetKernelArg(kernel, 5, sizeof(cl_ulong), &trial_max);
    CL_CHECK_STATUS(status, "arg 5")

    status = clSetKernelArg(kernel, 6, sizeof(cl_ulong), &step);
    CL_CHECK_STATUS(status, "arg 6")

    //execute
    const size_t gwSize = threads_cl;
    const size_t lwSize = LOCAL_WORK_SIZE;
    cl_event event;
    int current = 0;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gwSize, &lwSize, 0, NULL, &event);
    if (status != CL_SUCCESS) {
        printf("Kernel enqueue failed! GW Size %llu, LW Size %d, Error code %d\n", threads_cl, LOCAL_WORK_SIZE, status);
        return -1;
    }
    clWaitForEvents(current, &event);

    //read results
    cl_ulong bytes = bytes_output;
    status = clEnqueueReadBuffer(
            queue, mult->result_dev, CL_TRUE, 0,
            bytes, mult->result,
            0, NULL, &event);
    CL_CHECK_STATUS(status, "Enqueue read buffer")
    status = clWaitForEvents(1, &event);
    CL_CHECK_STATUS(status, "Read buffer")

    clFlush(queue);
    clFinish(queue);
    clReleaseEvent(event);

    //process results
    size total = 0;
    for (cl_ulong i = 0; i < threads_cl; i++) total += mult->result[i];
    return total;
}
