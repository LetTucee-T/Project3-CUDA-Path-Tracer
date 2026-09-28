#include "camera_sampling_test_cases.h"

namespace {
void cudaCheck(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) { cudaCheck(cudaMalloc(&data, count*sizeof(T))); }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
__global__ void probeCameraSampling(const cameraTest::Query* queries, cameraTest::Result* results, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) results[i] = cameraTest::evaluate(queries[i]);
}
}

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const int result = cameraTest::runSuite([](const std::vector<cameraTest::Query>& queries) {
        using namespace cameraTest;
        std::vector<Result> results(queries.size());
        DeviceArray<Query> input(queries.size()); DeviceArray<Result> output(queries.size());
        cudaCheck(cudaMemcpy(input.data, queries.data(), queries.size()*sizeof(Query), cudaMemcpyHostToDevice));
        probeCameraSampling<<<(int(queries.size())+127)/128, 128>>>(input.data, output.data, int(queries.size()));
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(results.data(), output.data, results.size()*sizeof(Result), cudaMemcpyDeviceToHost));
        // Same public helpers on host and device: compare flags, exact RNG words,
        // and float geometry with tolerance for host/device arithmetic differences.
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto host = evaluate(queries[i]); const auto& gpu = results[i];
            require(host.frameOK == gpu.frameOK && host.rayOK == gpu.rayOK, "Host/device validity mismatch");
            require(host.randomWord == gpu.randomWord, "Host/device integer random stream mismatch");
            for (int axis = 0; axis < 2; ++axis) {
                near(gpu.randomSample[axis], host.randomSample[axis], 2e-7f);
                near(gpu.randomDisk[axis], host.randomDisk[axis]); near(gpu.disk[axis], host.disk[axis]);
            }
            if (gpu.frameOK) {
                nearVector(gpu.frame.forward, host.frame.forward);
                nearVector(gpu.frame.right, host.frame.right); nearVector(gpu.frame.up, host.frame.up);
            }
            if (!gpu.rayOK || !queries[i].enabled || queries[i].camera.lensRadius == 0) {
                exactRay(gpu.ray, queries[i].pinhole);
            } else {
                nearVector(gpu.ray.origin, host.ray.origin); nearVector(gpu.ray.direction, host.ray.direction);
            }
        }
        return results;
    });
    cudaDeviceReset();
    return result;
}
