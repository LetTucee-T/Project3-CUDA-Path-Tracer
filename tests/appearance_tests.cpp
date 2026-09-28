#include "scene.h"
#include "appearance.h"
#include "cameraOrbit.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void check(bool v, const char* why) { if (!v) throw std::runtime_error(why); }
bool near(float a, float b) { return std::abs(a-b)<2e-5f; }
void put(const std::filesystem::path& p, const std::string& data) { std::ofstream(p,std::ios::binary)<<data; }
int main(int argc, char** argv) {
    try {
        check(argc==2,"scratch directory required");
        const std::filesystem::path dir(argv[1]); std::filesystem::create_directories(dir);
        // Deliberately permute all three index domains; use a concave polygon.
        put(dir/"shape.obj", "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 1 1 0\nv 0 2 0\n"
            "vt 1 1\nvt 0 0\nvt 0 1\nvt .5 .5\nvt 1 0\n"
            "vn 0 .6 .8\nvn 0 0 1\nvn .6 0 .8\n"
            "f 1/2/2 2/5/3 3/1/1 4/4/2 5/3/3\n");
        // STB supports binary PPM too; no external image fixture needed.
        std::string ppm="P6\n2 2\n255\n";
        const unsigned char rgb[]={255,0,0, 0,255,0, 0,0,255, 128,128,128};
        ppm.append(reinterpret_cast<const char*>(rgb),12);put(dir/"color.ppm",ppm);
        nlohmann::json j={
            {"Materials",{{"m",{{"TYPE","CoatedDiffuse"},{"RGB",{1,1,1}},
                                  {"COAT_WEIGHT",.12},{"BASE_COLOR_TEXTURE","color.ppm"}}}}},
            {"Objects",{{{"TYPE","mesh"},{"FILE","shape.obj"},{"MATERIAL","m"},
                          {"TRANS",{0,0,0}},{"ROTAT",{0,0,0}},{"SCALE",{1,1,1}},{"SMOOTH_NORMALS",true}}}},
            {"Camera",{{"RES",{8,8}},{"FOVY",30},{"ITERATIONS",1},{"DEPTH",4},{"FILE","test"},
                       {"EYE",{0,2,6}},{"LOOKAT",{0,0,0}},{"UP",{0,1,0}}}}};
        put(dir/"scene.json",j.dump());Scene scene((dir/"scene.json").string());
        check(scene.triangles.size()==3 && scene.surfaces.size()==3,"concave triangulation lost attributes");
        for(size_t i=0;i<scene.triangles.size();++i) {
            const auto& t=scene.triangles[i];const auto& s=scene.surfaces[i];
            const glm::vec3 positions[]={t.v0,t.v1,t.v2};
            for(int k=0;k<3;++k){
                check(near(s.uvs[k].x,positions[k].x*.5f)&&near(s.uvs[k].y,positions[k].y*.5f),"UV indices not tied to polygon corners");
                const glm::vec3 expected=positions[k].x==2 && positions[k].y==2?glm::vec3(0,.6,.8):
                    ((positions[k].x==2&&positions[k].y==0)||(positions[k].x==0&&positions[k].y==2))?glm::vec3(.6,0,.8):glm::vec3(0,0,1);
                check(glm::length(expected-s.normals[k])<2e-5f,"normal indices not preserved");
            }
        }
        const auto& info=scene.textures[0];const auto* pixels=scene.texturePixels.data();
        check(near(pixels[3].x,.2158605f),"sRGB input decode");
        check(glm::length(sampleBaseColor(info,pixels,{.25f,.25f})-glm::vec3(0,0,1))<1e-6,"OBJ V orientation");
        check(glm::length(sampleBaseColor(info,pixels,{1.25f,-.75f})-glm::vec3(0,0,1))<1e-6,"repeat addressing");
        check(near(sampleBaseColor(info,pixels,{.5f,.5f}).x,(1+.2158605f)/4),"bilinear interpolation in linear space");
        check(near(displayColor({1,1,1},true,0).x,encodeSRGB(.5f)),"display mapping");
        check(near(displayColor({2,2,2},false,3).x,2),"legacy linear output altered");
        for(float x:{-2.f,2.f})for(float y:{-3.f,3.f})for(float z:{-4.f,4.f}){
            Camera cam{};cam.lookAt={1,2,3};cam.position=cam.lookAt+glm::vec3(x,y,z);cam.right={1,0,0};
            const auto eye=cam.position;float phi,theta,r;
            cameraOrbit::angles(cam,phi,theta,r);cameraOrbit::apply(cam,phi,theta,r);
            check(glm::length(cam.position-eye)<3e-6f,"camera orbit flips eye quadrant/elevation");
            check(near(glm::length(cam.up),1)&&near(glm::length(cam.right),1)&&std::abs(glm::dot(cam.view,cam.up))<1e-6,"camera axes");
        }
        auto expectFailure=[&](nlohmann::json bad,const std::string& reason){
            put(dir/"bad.json",bad.dump());bool failed=false;
            try{Scene rejected((dir/"bad.json").string());}catch(const std::exception& e){failed=std::string(e.what()).find(reason)!=std::string::npos;}
            check(failed,"invalid appearance data accepted / incorrect diagnostic");
        };
        auto bad=j;bad["Materials"]["m"]["BASE_COLOR_TEXTURE"]="missing.png";expectFailure(bad,"Unable to load");
        bad=j;bad["Materials"]["m"]["COAT_WEIGHT"]=1.01;expectFailure(bad,"COAT_WEIGHT");
        put(dir/"missing_uv.obj","v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        bad=j;bad["Objects"][0]["FILE"]="missing_uv.obj";expectFailure(bad,"requires valid OBJ UVs");
        std::cout<<"PASS: corner indices, polygon attributes, color space, filtering, camera quadrants and invalid inputs\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
