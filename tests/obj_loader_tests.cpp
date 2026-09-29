#include "objLoader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Scratch directory required");
        const auto scratch = fs::path(argv[1]) / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        fs::create_directories(scratch);
        int passed = 0;
        const std::string vertices = "v 0 0 0\nv 2 0 0\nv 0 2 0\n";
        auto parse = [&](const std::string& contents) {
            const auto path = scratch / (std::to_string(passed) + ".obj");
            { std::ofstream out(path, std::ios::binary); out << contents; }
            return obj::read(path.u8string());
        };
        auto test = [&](const char* name, const std::function<void()>& run) {
            run(); ++passed; std::cout << "[PASS] " << name << '\n';
        };
        test("all_corner_forms_and_global_indices", [&] {
            const auto mesh = parse(vertices + "vt 0 1\nvt 1 0\nvt .5 .5\nvn 0 0 1\n"
                "g one\nf 1 2 3\no two\nf 1/3 2/1 3/2\ng three\nf 1//1 2//1 3//1\n"
                "mtllib absent.mtl\nusemtl material\ns 1\nf +1/3/1 +2/1/1 +3/2/1\n");
            require(mesh.faces.size() == 4, "groups dropped faces");
            require(mesh.faces[0].corners[0].uv == -1 && mesh.faces[0].corners[0].normal == -1, "missing indices lost");
            require(mesh.faces[1].corners[0].uv == 2 && mesh.faces[1].corners[1].vertex == 1, "UVs coupled to vertices");
            require(mesh.faces[2].corners[0].uv == -1 && mesh.faces[2].corners[0].normal == 0, "empty middle index shifted fields");
            require(mesh.faces[3].corners[2].vertex == 2 && mesh.faces[3].corners[2].uv == 1, "material/group reset indices");
        });
        test("negative_indices_use_each_array_at_face", [&] {
            const auto mesh = parse(vertices + "vt 0 0\nvt 1 0\nvn 0 0 1\n"
                "f -3/-1/-1 -2/-2/-1 -1/-1/-1\nv 9 9 9\nvt 9 9\nvn 1 0 0\n");
            const auto& face = mesh.faces.front();
            require(face.corners[0].vertex == 0 && face.corners[0].uv == 1 && face.corners[0].normal == 0,
                "negative indices were resolved against final array sizes");
            require(face.corners[2].vertex == 2, "negative vertex index incorrect");
        });
        test("bom_crlf_comments_continuation_scientific_numbers", [&] {
            const auto mesh = parse("\xEF\xBB\xBF# exported mesh\r\n\tv +0 0 -0\r\nv 2e+0 0 0 # x\r\n"
                "v 0 .2E1 0\r\nvt .5\r\nvt .5 .75 0\r\nf 1/1 2/2 \\\r\n 3/1 # end\r\n");
            require(mesh.positions[1][0] == 2 && mesh.positions[2][1] == 2, "decimal parsing changed values");
            require(std::signbit(mesh.positions[0][2]), "signed zero not preserved");
            require(mesh.texcoords[0][1] == 0 && mesh.faces[0].corners.size() == 3, "continuation/optional texture coordinate failed");
        });
        test("positive_forward_references", [&] {
            const auto mesh = parse("f 1 2 3\n" + vertices);
            require(mesh.faces[0].corners[2].vertex == 2, "valid positive reference rejected");
        });
        test("triangle_corner_order_unchanged", [&] {
            const auto mesh = parse(vertices + "f 3 1 2\n");
            const auto result = obj::triangulate(mesh, mesh.faces[0], "triangle.obj");
            require(result.size() == 1 && result[0] == std::array<std::size_t, 3>{0, 1, 2}, "triangle order changed");
        });
        test("existing_quad_diagonals_and_corner_order", [&] {
            const auto mesh = parse("v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\nf 1 2 3 4\nf 4 3 2 1\n");
            // Golden corner triples from the pre-migration cube loader. Keeping
            // cyclic order also keeps intersection/barycentric arithmetic stable.
            const std::vector<std::array<std::size_t, 3>> ccw{{2, 3, 0}, {0, 1, 2}};
            const std::vector<std::array<std::size_t, 3>> cw{{1, 3, 0}, {3, 1, 2}};
            require(obj::triangulate(mesh, mesh.faces[0], "quad.obj") == ccw, "CCW quad compatibility changed");
            require(obj::triangulate(mesh, mesh.faces[1], "quad.obj") == cw, "CW quad compatibility changed");
        });
        test("collinear_boundary_corners_retained", [&] {
            const auto mesh = parse("v 0 0 0\nv 1 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\nf 1 2 3 4 5\n");
            const auto result = obj::triangulate(mesh, mesh.faces[0], "collinear.obj");
            std::set<std::size_t> used;
            for (const auto& t : result) for (auto i : t) used.insert(i);
            require(result.size() == 3 && used.size() == 5, "collinear attribute corner discarded");
        });
        test("all_collinear_polygon_is_degenerate", [&] {
            const auto mesh = parse("v 0 0 0\nv 1 0 0\nv 2 0 0\nv 3 0 0\nf 1 2 3 4\n");
            require(obj::triangulate(mesh, mesh.faces[0], "line.obj").empty(), "zero-area polygon created triangles");
        });
        test("unicode_filename", [&] {
            const auto path = scratch / fs::u8path(u8"\u7f51\u683c.obj");
            { std::ofstream out(path); out << vertices << "f 1 2 3\n"; }
            require(obj::read(path.u8string()).faces.size() == 1, "UTF-8 filename failed");
        });
        const std::vector<std::string> invalid = {
            "f 0 2 3\n", "f -4 -2 -1\n", "f 1 2 9\n", "f 1 2\n", "f 1/ 2 3\n",
            "f 1// 2 3\n", "f 1/1/ 2 3\n", "f 1/1/1/1 2 3\n", "f 1/a 2 3\n",
            "f 1/1 2 3\n", "f 1//1 2 3\n", "f 1.0 2 3\n", "f +-1 2 3\n",
            "f 999999999999999999999999 2 3\n", "f -9223372036854775808 2 3\n",
            "v nan 0 0\n", "v inf 0 0\n", "v 1e400 0 0\n", "v 1e39 0 0\n",
            "v 1junk 0 0\n", "v +-1 0 0\n", "v 1 2\n", "vt nan 0\n", "vn 0 1\n",
            "f 1 2 \\\n", "curv 0 1 1 2\n"
        };
        test("invalid_records_report_filename_and_line", [&] {
            for (const auto& bad : invalid) {
                bool rejected = false;
                try { parse(vertices + bad); }
                catch (const std::exception& e) { const std::string message = e.what(); rejected = message.find(".obj:4:") != std::string::npos; }
                require(rejected, ("invalid input accepted or wrong diagnostic: " + bad).c_str());
            }
        });
        test("self_intersecting_and_nonplanar_polygons_rejected", [&] {
            const std::vector<std::string> polygons = {
                "v 0 0 0\nv 2 2 0\nv 0 2 0\nv 2 0 0\nf 1 2 3 4\n",
                "v 0 0 0\nv 2 0 0\nv 2 2 1\nv 0 2 0\nf 1 2 3 4\n"
            };
            for (const auto& text : polygons) {
                const auto mesh = parse(text);
                bool rejected = false;
                try { obj::triangulate(mesh, mesh.faces[0], "invalid.obj"); }
                catch (const std::exception& e) { rejected = std::string(e.what()).find("invalid.obj:5:") != std::string::npos; }
                require(rejected, "invalid polygon silently triangulated");
            }
        });
        std::cout << passed << " OBJ parser checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
