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
    if (argc<3) {
        std::cerr << "Usage: scene_render scene.json output.json [first_iteration] [--profile] "
            "[--sync-every-stage] [--readback-every-sample] [--display-every-sample] [--no-distance-pruning] "
            "[--bvh-builder median|sah] [--bvh-leaf-size N] [--bvh-bins N] "
            "[--profile-intersections] [--intersection-profile-stride N] "
            "[--reference-traversal] [--checked-traversal] [--bvh-layout wide|compact]\n";
        return 1;
    }
    uchar4* output=nullptr;
    try {
        int firstIteration=1;
        bool haveFirstIteration=false, readbackEverySample=false, displayEverySample=false;
        PathtraceOptions options;
        BVHBuildOptions buildOptions;
        for (int i=3;i<argc;++i) {
            const std::string argument(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + argument);
                return argv[i];
            };
            const auto integer = [&]() {
                const auto text=value(); size_t end=0; const int n=std::stoi(text,&end);
                if(end!=text.size() || n<1) throw std::runtime_error("Invalid positive integer for " + argument);
                return n;
            };
            if(argument=="--profile") options.profile=true;
            else if(argument=="--profile-intersections") options.profileIntersections=true;
            else if(argument=="--intersection-profile-stride") options.intersectionProfileStride=integer();
            else if(argument=="--bvh-builder") {
                const auto name=value();
                if(name=="median") buildOptions.splitMethod=BVHSplitMethod::Median;
                else if(name=="sah") buildOptions.splitMethod=BVHSplitMethod::BinnedSAH;
                else throw std::runtime_error("BVH builder must be median or sah");
            }
            else if(argument=="--bvh-leaf-size") buildOptions.maxLeafTriangles=integer();
            else if(argument=="--bvh-bins") buildOptions.binCount=integer();
            else if(argument=="--sync-every-stage") options.synchronizeEachStage=true;
            else if(argument=="--readback-every-sample") readbackEverySample=true;
            else if(argument=="--display-every-sample") displayEverySample=true;
            else if(argument=="--no-distance-pruning") options.crossMeshPruning=false;
            else if(argument=="--reference-traversal") options.cachedBVHBounds=false;
            else if(argument=="--checked-traversal") options.validatedBVHTraversal=false;
            else if(argument=="--bvh-layout") {
                const auto layout=value();
                if(layout=="wide") options.compactBVHNodes=false;
                else if(layout=="compact") options.compactBVHNodes=true;
                else throw std::runtime_error("BVH layout must be wide or compact");
            }
            else if(argument.rfind("--",0)!=0 && !haveFirstIteration) {
                size_t end=0; firstIteration=std::stoi(argument,&end);
                if(end!=argument.size()) throw std::runtime_error("Invalid first iteration");
                haveFirstIteration=true;
            } else throw std::runtime_error("Unknown argument: "+argument);
        }
        const auto loadStart=Clock::now(); Scene scene(argv[1],buildOptions);
        const double loadMs=elapsed(loadStart);
        initializeCamera(scene.state.camera);
        std::filesystem::create_directories(std::filesystem::absolute(argv[2]).parent_path());
        const auto& camera=scene.state.camera;
        const int w=camera.resolution.x,h=camera.resolution.y,spp=scene.state.iterations;
        if (w<=0 || h<=0 || spp<=0 || firstIteration<1 || firstIteration+spp>4000000)
            throw std::runtime_error("Invalid render parameters");
        if(displayEverySample) cudaCheck(cudaMalloc(&output,size_t(w)*h*sizeof(uchar4)));
        const auto sample=[&](int iteration) {
            pathtrace(output,0,iteration);
            if(readbackEverySample) pathtraceReadback();
        };
        pathtraceInit(&scene,options);
        for (int i=1;i<=8;++i) sample(i);
        pathtraceReadback();
        pathtraceFree();
        const auto initStart=Clock::now(); pathtraceInit(&scene,options);
        cudaCheck(cudaDeviceSynchronize()); // Complete initialization before timing the loop.
        const double initMs=elapsed(initStart);
        const auto loopStart=Clock::now();
        for (int i=0;i<spp;++i) sample(firstIteration+i);
        pathtraceReadback(); // Timing includes GPU completion and the final CPU image.
        cudaCheck(cudaDeviceSynchronize());
        const double loopMs=elapsed(loopStart);
        nlohmann::json data={{"resolution",{w,h}},{"spp",spp},{"first_iteration",firstIteration},
            {"warmup_spp",8},{"depth",scene.state.traceDepth},{"dof",scene.state.enableDepthOfField},
            {"lens_radius",camera.lensRadius},{"focus_distance",camera.focalDistance},
            {"antialiasing",scene.state.enableAntialiasing},{"compaction",scene.state.enableStreamCompaction},
            {"sorting",scene.state.enableMaterialSorting},{"bvh",scene.state.enableBVH},
            {"triangles",scene.triangles.size()},{"objects",scene.geoms.size()},
            {"scene_load_ms",loadMs},{"initialization_ms",initMs},{"render_loop_ms",loopMs},
            {"cross_mesh_pruning",options.crossMeshPruning},{"synchronize_each_stage",options.synchronizeEachStage},
            {"readback_every_sample",readbackEverySample},{"display_every_sample",displayEverySample}};
        const auto statistics=pathtraceGetMetrics();
        data["image_readbacks"]=statistics.imageReadbacks;
        data["image_readback_bytes"]=statistics.readbackBytes;
        data["stage_synchronizations"]=statistics.stageSynchronizations;
        data["bvh_builder"]=buildOptions.splitMethod==BVHSplitMethod::Median?"median":"sah";
        data["bvh_leaf_size"]=buildOptions.maxLeafTriangles;
        data["bvh_bins"]=buildOptions.binCount;
        data["cached_bvh_bounds"]=options.cachedBVHBounds;
        data["validated_bvh_traversal"]=options.cachedBVHBounds && options.validatedBVHTraversal;
        data["bvh_layout"]=options.compactBVHNodes?"compact":"wide";
        data["bvh_node_bytes"]=options.compactBVHNodes?sizeof(CompactBVHNode):sizeof(BVHNode);
        data["bvh_node_and_index_bytes"]=scene.bvhNodes.size()
            * (options.compactBVHNodes?sizeof(CompactBVHNode):sizeof(BVHNode))
            + scene.bvhTriangleIndices.size()*sizeof(int);
        data["bvh_build_ms"]=scene.bvhStats.buildMilliseconds;
        data["bvh_validation_ms"]=scene.bvhStats.validationMilliseconds;
        data["bvh_nodes"]=scene.bvhStats.nodeCount;
        data["bvh_leaves"]=scene.bvhStats.leafCount;
        data["bvh_max_depth"]=scene.bvhStats.maxDepth;
        if(options.profileIntersections) {
            auto& work=data["intersection_work"];
            work["pixel_stride"]=options.intersectionProfileStride;
            work["note"]="Separate replay of incoming paths with pixelIndex % stride == 0, before shading/sorting. Counts are sampled, not extrapolated. Render-loop time includes diagnostic overhead; do not use it for speed comparisons. Root rejections include ordinary misses as well as distance pruning.";
            work["bounces"]=nlohmann::json::array();
            for(size_t depth=0;depth<statistics.intersectionWork.size();++depth) {
                const auto& bounce=statistics.intersectionWork[depth];
                nlohmann::json row={{"depth",depth},{"sampled_rays",bounce.sampledRays},
                    {"mismatches",bounce.mismatches},{"objects",nlohmann::json::array()}};
                for(size_t g=0;g<bounce.perGeometry.size();++g) {
                    const auto& c=bounce.perGeometry[g];
                    row["objects"].push_back({{"object_index",g},{"is_mesh",scene.geoms[g].type==MESH},
                        {"queries",c.queries},{"bounded_queries",c.boundedQueries},{"hits",c.hits},
                        {"root_rejects",c.rootRejects},{"node_visits",c.nodeVisits},{"aabb_tests",c.aabbTests},
                        {"triangle_tests",c.triangleTests},{"fallbacks",c.fallbacks},{"max_stack",c.maxStack}});
                }
                work["bounces"].push_back(row);
            }
        }
        if(options.profile) {
            const char* names[]={"prepare","camera","intersection","sorting","shading",
                "compaction","gather","display","readback"};
            for(size_t i=0;i<statistics.phases.size();++i) {
                const auto& t=statistics.phases[i];
                data["profile"][names[i]]={{"gpu_stream_ms",t.gpuMilliseconds},
                    {"host_section_ms",t.hostMilliseconds},{"calls",t.calls}};
            }
            data["stage_synchronization_ms"]=statistics.stageSynchronizationMilliseconds;
            data["profile_note"]="Separate diagnostic run; CUDA-event stream intervals include launch gaps. Host/GPU times overlap and must not be summed. Events are resolved per sample.";
        }
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
        data["kind"]=options.profileIntersections?"intersection_work_diagnostic"
            :options.profile?"cuda_event_instrumented":"production_renderer_headless";
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
