// Included after the shared CUDA test helpers. Expected values were computed
// independently with 80-digit decimal arithmetic, not the production routine.
#pragma once

struct TrianglePrecisionResult { float t; glm::vec3 weights; };
__global__ void queryTrianglePrecision(const TriangleQuery* input, TrianglePrecisionResult* output, int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i].t=triangleIntersectionTest(input[i].triangle,input[i].ray,&output[i].weights);
}

template<class Test> void runTrianglePrecisionTests(Test& test,const std::filesystem::path& root) {
    std::ifstream file(root/"tests/data/thin_triangle_regressions.json");
    nlohmann::json data;file>>data;
    const auto vec=[](const nlohmann::json& j){return glm::vec3(j[0].get<float>(),j[1].get<float>(),j[2].get<float>());};
    test("robust triangle rejects non-finite inputs and initializes miss weights",[&]{
        std::vector<TriangleQuery> queries;
        for(float invalid:{std::numeric_limits<float>::quiet_NaN(),INFINITY,-INFINITY})
            for(int field=0;field<5;++field)for(int axis=0;axis<3;++axis){
                TriangleQuery q{triangle(),path().ray};
                glm::vec3* fields[]={&q.ray.origin,&q.ray.direction,&q.triangle.v0,&q.triangle.v1,&q.triangle.v2};
                (*fields[field])[axis]=invalid;queries.push_back(q);
            }
        DeviceArray<TriangleQuery> inputs(queries);DeviceArray<TrianglePrecisionResult> output(queries.size());
        queryTrianglePrecision<<<1,128>>>(inputs.data,output.data,int(queries.size()));
        cudaCheck(cudaGetLastError());const auto results=output.read();
        for(size_t i=0;i<results.size();++i){
            near(results[i].t,-1,0);nearVector(results[i].weights,glm::vec3(0),0);
            near(triangleIntersectionTest(queries[i].triangle,queries[i].ray),-1,0);
        }
    });
    test("thin triangles: 80-digit oracle, all vertex permutations and power-of-two scales",[&]{
        std::vector<TriangleQuery> queries;std::vector<float> expected;std::vector<glm::vec3> weights;
        for(const auto& row:data["cases"]) {
            const glm::vec3 vertices[]={vec(row["vertices"][0]),vec(row["vertices"][1]),vec(row["vertices"][2])};
            const auto w=vec(row["expected_barycentrics"]);
            for(int exponent:{-30,0,30}) {
                const float scale=std::ldexp(1.0f,exponent);int order[]={0,1,2};
                do {
                    TriangleQuery q{};q.triangle.v0=vertices[order[0]]*scale;
                    q.triangle.v1=vertices[order[1]]*scale;q.triangle.v2=vertices[order[2]]*scale;
                    q.ray={vec(row["origin"])*scale,vec(row["direction"])*scale};queries.push_back(q);
                    expected.push_back(row["expected_t"].get<float>());weights.push_back({w[order[0]],w[order[1]],w[order[2]]});
                }while(std::next_permutation(order,order+3));
            }
        }
        DeviceArray<TriangleQuery> inputs(queries);DeviceArray<TrianglePrecisionResult> output(queries.size());
        queryTrianglePrecision<<<(int(queries.size())+127)/128,128>>>(inputs.data,output.data,int(queries.size()));
        cudaCheck(cudaGetLastError());const auto results=output.read();
        for(size_t i=0;i<results.size();++i){
            const auto& r=results[i];
            near(r.t,expected[i],expected[i]<0?0:2*FLT_EPSILON*std::abs(expected[i]));
            near(triangleIntersectionTest(queries[i].triangle,queries[i].ray),expected[i],expected[i]<0?0:2*FLT_EPSILON*std::abs(expected[i]));
            if(r.t>0){nearVector(r.weights,weights[i],2e-6f);near(r.weights.x+r.weights.y+r.weights.z,1,2e-7f);}
            else nearVector(r.weights,glm::vec3(0),0);
        }
    });
    test("thin shared edges and vertices have no cracks for either winding or ray direction",[&]{
        std::vector<TriangleQuery> queries;
        const float width=std::ldexp(1.0f,-30);
        for(int axis=0;axis<3;++axis)for(int sign:{-1,1})for(int i=0;i<=1024;++i){
            const auto permute=[axis](glm::vec3 v){return glm::vec3(v[axis],v[(axis+1)%3],v[(axis+2)%3]);};
            const float x=float(i)/1024,y=x*width;
            for(int side:{-1,0,1}){
                const float offset=side==0?y:std::nextafter(y,side<0?-INFINITY:INFINITY);
                if(offset<0 || offset>width)continue;
                TriangleQuery a{},b{};
                a.triangle.v0=permute({0,0,0});a.triangle.v1=permute({1,0,0});a.triangle.v2=permute({1,width,0});
                b.triangle.v0=permute({0,0,0});b.triangle.v1=permute({1,width,0});b.triangle.v2=permute({0,width,0});
                if(sign<0){std::swap(a.triangle.v1,a.triangle.v2);std::swap(b.triangle.v1,b.triangle.v2);}
                a.ray=b.ray={permute({x,offset,float(sign)}),permute({0,0,float(-sign)})};
                queries.push_back(a);queries.push_back(b);
            }
        }
        DeviceArray<TriangleQuery> inputs(queries);DeviceArray<TrianglePrecisionResult> output(queries.size());
        queryTrianglePrecision<<<(int(queries.size())+127)/128,128>>>(inputs.data,output.data,int(queries.size()));
        cudaCheck(cudaGetLastError());const auto results=output.read();
        for(size_t i=0;i<results.size();i+=2){
            require(results[i].t>0 || results[i+1].t>0,"Both adjacent triangles rejected an in-quad ray");
            for(size_t j=i;j<i+2;++j)if(results[j].t>0)near(results[j].t,1,0);
        }
    });
    test("recorded showcase rays agree across brute force, AABB, median and SAH trees",[&]{
        Scene scene((root/"scenes/shatterseal_moonlit_hall_refined.json").string());
        std::vector<PathSegment> rays;
        for(const auto& row:data["cases"])rays.push_back(path(vec(row["world_origin"]),vec(row["world_direction"])));
        const auto baseline=intersect(scene.geoms,scene.triangles,rays,{4,32,BVHSplitMethod::Median,16});
        const auto sah=intersect(scene.geoms,scene.triangles,rays,{2,32,BVHSplitMethod::BinnedSAH,16});
        for(size_t i=0;i<rays.size();++i){near(baseline[i].t,sah[i].t,0);nearVector(baseline[i].surfaceNormal,sah[i].surfaceNormal,0);}
    });
}
