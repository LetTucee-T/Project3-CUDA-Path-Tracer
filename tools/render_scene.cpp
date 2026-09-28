#include "cameraOrbit.h"
#include "pathtrace.h"
#include "image.h"
#include "json.hpp"
#ifdef DOF_METRICS
#include "metrics.h"
#endif
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
// Match main.cpp's initial camera update, including its float arithmetic.
void initializeCamera(Camera& cam) {
    float phi, theta, radius;
    cameraOrbit::angles(cam, phi, theta, radius);
    cameraOrbit::apply(cam, phi, theta, radius);
}
int main(int argc,char** argv) {
    if (argc<3 || argc>4) return 1;
    uchar4* output=nullptr;
    try {
        const int firstIteration=argc==4?std::stoi(argv[3]):1;
        const auto loadStart=Clock::now(); Scene scene(argv[1]);
        const double loadMs=elapsed(loadStart);
        initializeCamera(scene.state.camera);
        std::filesystem::create_directories(std::filesystem::absolute(argv[2]).parent_path());
        const auto& camera=scene.state.camera;
        const int w=camera.resolution.x,h=camera.resolution.y,spp=scene.state.iterations;
        if (w<=0 || h<=0 || spp<=0 || firstIteration<1 || firstIteration+spp>4000000)
            throw std::runtime_error("Invalid render parameters");
        cudaCheck(cudaMalloc(&output,size_t(w)*h*sizeof(uchar4)));
        pathtraceInit(&scene);
        for (int i=1;i<=8;++i) pathtrace(output,0,i);
        pathtraceFree();
        const auto initStart=Clock::now(); pathtraceInit(&scene);
        const double initMs=elapsed(initStart);
        const auto loopStart=Clock::now();
        for (int i=0;i<spp;++i) pathtrace(output,0,firstIteration+i);
        cudaCheck(cudaDeviceSynchronize());
        const double loopMs=elapsed(loopStart);
        nlohmann::json data={{"resolution",{w,h}},{"spp",spp},{"first_iteration",firstIteration},
            {"warmup_spp",8},{"depth",scene.state.traceDepth},{"dof",scene.state.enableDepthOfField},
            {"lens_radius",camera.lensRadius},{"focus_distance",camera.focalDistance},
            {"antialiasing",scene.state.enableAntialiasing},{"compaction",scene.state.enableStreamCompaction},
            {"sorting",scene.state.enableMaterialSorting},{"bvh",scene.state.enableBVH},
            {"triangles",scene.triangles.size()},{"objects",scene.geoms.size()},
            {"scene_load_ms",loadMs},{"initialization_ms",initMs},{"render_loop_ms",loopMs}};
#ifdef DOF_METRICS
        const auto metrics=getPathtraceMetrics();
        if(metrics.samples!=spp) throw std::runtime_error("Wrong sample counter");
        const auto attributes=getCameraKernelAttributes();
        data["kind"]="cuda_event_instrumented";
        data["camera_kernel_ms"]=metrics.cameraMs;
        data["camera_host_section_ms"]=metrics.cameraHostMs;
        data["intersection_ms"]=metrics.intersectionMs;
        data["intersection_calls"]=metrics.intersectionCalls;
        data["camera_registers_per_thread"]=attributes.numRegs;
        data["camera_local_bytes_per_thread"]=attributes.localSizeBytes;
#else
        data["kind"]="production_renderer_headless";
#endif
        pathtraceFree(); cudaCheck(cudaFree(output)); output=nullptr;
        const std::filesystem::path report(argv[2]);
        const auto base=report.parent_path()/report.stem();
        std::ofstream raw(base.string()+".f32",std::ios::binary);
        Image image(w,h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            const glm::vec3 pixel=scene.state.image[x+y*w]/float(spp);
            if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y)||!std::isfinite(pixel.z)
                ||pixel.x<0||pixel.y<0||pixel.z<0) throw std::runtime_error("Invalid radiance");
            const float rgb[]={pixel.x,pixel.y,pixel.z};raw.write(reinterpret_cast<const char*>(rgb),sizeof(rgb));
            image.setPixel(w-1-x,y,pixel);
        }
        raw.close();if(!raw)throw std::runtime_error("Raw output failed");
        image.savePNG(base.string(), scene.state.displayTransform, scene.state.exposure);
        image.saveHDR(base.string());
        cudaDeviceProp gpu{};cudaCheck(cudaGetDeviceProperties(&gpu,0));data["gpu"]=gpu.name;
        std::ofstream file(report);file<<data.dump(2)<<'\n';if(!file)throw std::runtime_error("Report failed");
        std::cout<<"METRICS "<<data.dump()<<'\n';
        cudaCheck(cudaDeviceReset());return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';pathtraceFree();if(output)cudaFree(output);cudaDeviceReset();return 1;
    }
}
