#pragma once

#define CL_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <assert.h>

typedef unsigned long long sample_t;
typedef long long log_t;

static log_t* lgac_cl;

PACK(typedef struct {real_t prob; short alias;}) alias_pair_cl;

#define LOCAL_WORK_SIZE 64
static uint64_t threads_cl = 6400;
static uint64_t bytes_output;

static cl_context context;
static cl_program program;
static cl_kernel kernel;
static cl_command_queue queue;

static cl_mem lgac_dev;

static size step_num = 1ll<<30;

//__LINE__ works with GNU GCC.
#define CL_CHECK_ERROR(status,expr) \
	status = expr; \
	if(status != CL_SUCCESS) { \
		printf("Error %d in " #expr " at line: %d\n",status,__LINE__); \
		return 1; \
	}

#define CL_CHECK_STATUS(status,expr) \
	if(status != CL_SUCCESS) { \
		printf("Status error %d in " #expr " at line: %d\n",status,__LINE__); \
		return 1; \
	}

static void create_program(const char* program_name) {
    FILE* file = fopen(program_name, "rb");
    assert(file != NULL);

    fseek(file, 0, SEEK_END);
    size_t sourceSize = ftell(file);
    rewind(file);
    assert(sourceSize > 0);
    char* source = (char*)malloc(sourceSize*sizeof(char));
    size_t result = fread(source, 1, sourceSize, file);
    fclose(file);
    assert(result == sourceSize);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &sourceSize, NULL);
    free(source);
}

static int load_kernel() {
    uint64_t threadsPorBloque = LOCAL_WORK_SIZE;
    uint64_t bloques = (threads_cl+threadsPorBloque-1)/threadsPorBloque;
    // corregir valores
    threads_cl = threadsPorBloque*bloques;
    bytes_output = sizeof(sample_t) * threads_cl;

    int plat_id = 0, dev_id = 0;
    //query platforms
    cl_uint numPlatforms;
    cl_int status;
    CL_CHECK_ERROR(status,clGetPlatformIDs(0, NULL, &numPlatforms))
    assert(numPlatforms > 0);
    cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
    CL_CHECK_ERROR(status,clGetPlatformIDs(numPlatforms,platforms,NULL))
    cl_platform_id platform = platforms[plat_id];
    free(platforms);

    //query devices on platforms
    cl_uint numDevices;
    CL_CHECK_ERROR(status,clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&numDevices))
    assert(numDevices > 0);
    cl_device_id* devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    CL_CHECK_ERROR(status,clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,numDevices,devices,NULL))

    context = clCreateContext(NULL,numDevices,devices,NULL,NULL,NULL);
    cl_device_id device_id = devices[dev_id];

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
    CL_CHECK_STATUS(status,"Create command queue")

    //load proper kernel
    create_program("pvalues-simulation/src/cl/kernel.cl");

    //build kernel
//    status = clBuildProgram(program,1, devices, "-g", NULL, NULL);
    status = clBuildProgram(program,1, devices, "", NULL, NULL);

    //check for kernel errors
    if (status != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        free(log);
        return 1;
    }
    //kernel
    kernel = clCreateKernel(program, "sample_alias", &status);
    CL_CHECK_STATUS(status, "Create kernel")
    free(devices);

    lgac_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, (n_max+1)*sizeof(log_t),NULL,&status);
    status = clEnqueueWriteBuffer(queue, lgac_dev, CL_FALSE, 0, (n_max+1)*sizeof(log_t), lgac_cl, 0, NULL, NULL);

    CL_CHECK_STATUS(status,"Create output buffer")

    clSetKernelArg(kernel, 9, sizeof(cl_mem), &lgac_dev);
    CL_CHECK_STATUS(status, "arg 9")

    return 0;
}

static int free_kernel() {
    cl_int status;

    CL_CHECK_ERROR(status,clReleaseMemObject(lgac_dev))
    CL_CHECK_ERROR(status,clReleaseCommandQueue(queue))
    CL_CHECK_ERROR(status,clReleaseKernel(kernel))
    CL_CHECK_ERROR(status,clReleaseProgram(program))
    CL_CHECK_ERROR(status,clReleaseContext(context))

    return 0;
}
