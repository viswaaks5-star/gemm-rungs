// Stand-in for NVML, for tests/mock only (see include/cuda_runtime.h).
#pragma once

typedef struct nvmlDevice_st* nvmlDevice_t;
enum nvmlReturn_t { NVML_SUCCESS = 0 };
enum nvmlClockType_t { NVML_CLOCK_GRAPHICS = 0, NVML_CLOCK_SM = 1, NVML_CLOCK_MEM = 2 };
struct nvmlPciInfo_t {
    unsigned int domain, bus, device;
};

#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80
#define nvmlClocksThrottleReasonSwPowerCap 0x0000000000000004LL
#define nvmlClocksThrottleReasonHwSlowdown 0x0000000000000008LL
#define nvmlClocksThrottleReasonSwThermalSlowdown 0x0000000000000020LL
#define nvmlClocksThrottleReasonHwThermalSlowdown 0x0000000000000040LL
#define nvmlClocksThrottleReasonHwPowerBrakeSlowdown 0x0000000000000080LL

extern "C" {
const char* nvmlErrorString(nvmlReturn_t);
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int*);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int, nvmlDevice_t*);
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t, nvmlPciInfo_t*);
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t, nvmlClockType_t, unsigned int*);
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t, nvmlClockType_t, unsigned int*);
nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t, unsigned long long*);
nvmlReturn_t nvmlSystemGetDriverVersion(char*, unsigned int);
}
