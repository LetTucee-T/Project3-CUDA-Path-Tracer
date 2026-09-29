#include "objLoader.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace obj {
namespace {
[[noreturn]] void fail(const std::string& file, std::size_t line, const std::string& reason) {
    throw std::runtime_error("OBJ " + file + ":" + std::to_string(line) + ": " + reason);
}

float number(std::string_view token, const std::string& file, std::size_t line) {
    // Parse through double, then round once to the renderer's float storage.
    // from_chars is locale independent and also checks for trailing garbage.
    if (!token.empty() && token.front() == '+') {
        token.remove_prefix(1);
        if (!token.empty() && (token.front() == '+' || token.front() == '-')) fail(file, line, "invalid numeric sign");
    }
    double value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()
        || !std::isfinite(value) || std::abs(value) > (std::numeric_limits<float>::max)())
        fail(file, line, "invalid or non-finite number");
    return static_cast<float>(value);
}

int index(std::string_view token, std::size_t count, const std::string& file, std::size_t line) {
    if (!token.empty() && token.front() == '+') {
        token.remove_prefix(1);
        if (!token.empty() && (token.front() == '+' || token.front() == '-')) fail(file, line, "invalid index sign");
    }
    long long value = 0;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value == 0)
        fail(file, line, "invalid index (indices are nonzero integers)");
    // Negative references are relative to each attribute array AT THIS FACE,
    // not the final array size. Positive references are checked after reading.
    const long long resolved = value < 0 ? static_cast<long long>(count) + value : value - 1;
    if (resolved < 0 || resolved >= (std::numeric_limits<int>::max)())
        fail(file, line, "index out of range");
    return static_cast<int>(resolved);
}

Index corner(std::string_view token, const Mesh& mesh, const std::string& file, std::size_t line) {
    Index result;
    const auto slash = token.find('/');
    result.vertex = index(token.substr(0, slash), mesh.positions.size(), file, line);
    if (slash == std::string_view::npos) return result;
    const auto next = token.find('/', slash + 1);
    if (next == std::string_view::npos) {
        result.uv = index(token.substr(slash + 1), mesh.texcoords.size(), file, line);
    } else {
        if (token.find('/', next + 1) != std::string_view::npos)
            fail(file, line, "face corner has too many slash-separated indices");
        if (next != slash + 1)
            result.uv = index(token.substr(slash + 1, next - slash - 1), mesh.texcoords.size(), file, line);
        result.normal = index(token.substr(next + 1), mesh.normals.size(), file, line);
    }
    return result;
}

using Point = std::array<double, 2>;
constexpr double epsilon = 64.0 * std::numeric_limits<double>::epsilon();
double turn(const Point& a, const Point& b, const Point& c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
bool onSegment(const Point& a, const Point& b, const Point& p) {
    return std::abs(turn(a, b, p)) <= epsilon
        && p[0] >= (std::min)(a[0], b[0]) - epsilon && p[0] <= (std::max)(a[0], b[0]) + epsilon
        && p[1] >= (std::min)(a[1], b[1]) - epsilon && p[1] <= (std::max)(a[1], b[1]) + epsilon;
}
bool crossing(const Point& a, const Point& b, const Point& c, const Point& d) {
    const double abC = turn(a, b, c), abD = turn(a, b, d);
    const double cdA = turn(c, d, a), cdB = turn(c, d, b);
    if (((abC > epsilon && abD < -epsilon) || (abC < -epsilon && abD > epsilon))
        && ((cdA > epsilon && cdB < -epsilon) || (cdA < -epsilon && cdB > epsilon))) return true;
    return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}
}

Mesh read(const std::string& filename) {
    std::ifstream input(std::filesystem::u8path(filename));
    if (!input) throw std::runtime_error("Unable to open OBJ file: " + filename);
    Mesh mesh;
    std::string line, logical;
    std::size_t lineNumber = 0, firstLine = 1;
    bool continued = false;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        if (!continued) { firstLine = lineNumber; logical.clear(); }
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        const auto end = line.find_last_not_of(" \t\r");
        continued = end != std::string::npos && line[end] == '\\';
        if (continued) line.resize(end);
        logical += line;
        if (continued) { logical += ' '; continue; }
        std::istringstream fields(logical);
        fields.imbue(std::locale::classic());
        std::string tag, token;
        if (!(fields >> tag)) continue;
        if (tag == "v" || tag == "vt" || tag == "vn") {
            std::vector<float> values;
            while (fields >> token) values.push_back(number(token, filename, firstLine));
            if (tag == "v") {
                if (values.size() != 3 && values.size() != 4 && values.size() != 6 && values.size() != 7)
                    fail(filename, firstLine, "vertex requires xyz (optional weight/colors are not used)");
                mesh.positions.push_back({values[0], values[1], values[2]});
            } else if (tag == "vt") {
                if (values.empty() || values.size() > 3) fail(filename, firstLine, "texture coordinate requires 1-3 numbers");
                mesh.texcoords.push_back({values[0], values.size() > 1 ? values[1] : 0.0f});
            } else {
                if (values.size() != 3) fail(filename, firstLine, "normal requires xyz");
                mesh.normals.push_back({values[0], values[1], values[2]});
            }
            const auto limit = static_cast<std::size_t>((std::numeric_limits<int>::max)());
            if (mesh.positions.size() > limit || mesh.texcoords.size() > limit || mesh.normals.size() > limit)
                fail(filename, firstLine, "attribute count exceeds int indexing");
        } else if (tag == "f") {
            Face face;
            face.line = firstLine;
            while (fields >> token) face.corners.push_back(corner(token, mesh, filename, firstLine));
            if (face.corners.size() < 3) fail(filename, firstLine, "face requires at least three corners");
            mesh.faces.push_back(std::move(face));
        } else if (tag != "o" && tag != "g" && tag != "s" && tag != "mtllib" && tag != "usemtl") {
            fail(filename, firstLine, "unsupported record '" + tag + "' (export polygon meshes)");
        }
    }
    if (input.bad()) throw std::runtime_error("Unable to read OBJ file: " + filename);
    if (continued) fail(filename, firstLine, "unfinished line continuation");
    for (const auto& face : mesh.faces) for (const auto& i : face.corners) {
        if (static_cast<std::size_t>(i.vertex) >= mesh.positions.size()
            || (i.uv >= 0 && static_cast<std::size_t>(i.uv) >= mesh.texcoords.size())
            || (i.normal >= 0 && static_cast<std::size_t>(i.normal) >= mesh.normals.size()))
            fail(filename, face.line, "face index out of range");
    }
    return mesh;
}

std::vector<std::array<std::size_t, 3>> triangulate(
    const Mesh& mesh, const Face& face, const std::string& filename) {
    const std::size_t count = face.corners.size();
    if (count < 3) fail(filename, face.line, "face requires at least three corners");
    if (count == 3) return {{{0, 1, 2}}};

    // Work in a translated, uniformly scaled frame. This keeps geometric tests
    // useful for both tiny assets and large coordinates without changing floats
    // stored in the actual mesh or interpolated corner attributes.
    const auto& origin = mesh.positions[face.corners[0].vertex];
    std::vector<std::array<double, 3>> points(count);
    double scale = 0;
    for (std::size_t i = 0; i < count; ++i) for (int axis = 0; axis < 3; ++axis) {
        points[i][axis] = double(mesh.positions[face.corners[i].vertex][axis]) - origin[axis];
        scale = (std::max)(scale, std::abs(points[i][axis]));
    }
    if (scale == 0) return {};
    for (auto& p : points) for (auto& x : p) x /= scale;
    std::array<double, 3> normal{}, fallback{};
    double largest = 0;
    for (std::size_t i = 1; i + 1 < count; ++i) {
        const auto& a = points[i]; const auto& b = points[i + 1];
        const std::array<double, 3> n{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
        double magnitude = 0;
        for (int axis = 0; axis < 3; ++axis) { normal[axis] += n[axis]; magnitude += n[axis]*n[axis]; }
        if (magnitude > largest) { largest = magnitude; fallback = n; }
    }
    if (largest == 0) return {};
    double length = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (length == 0) { normal = fallback; length = std::sqrt(largest); }
    int dropped = 0;
    for (int axis = 1; axis < 3; ++axis) if (std::abs(normal[axis]) > std::abs(normal[dropped])) dropped = axis;
    std::vector<Point> projected(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& p = points[i];
        const double planeDistance = (p[0]*normal[0] + p[1]*normal[1] + p[2]*normal[2]) / length;
        if (std::abs(planeDistance) > 32.0 * std::numeric_limits<float>::epsilon())
            fail(filename, face.line, "nonplanar polygon; triangulate before export");
        projected[i] = {p[(dropped + 1) % 3], p[(dropped + 2) % 3]};
    }
    // Reject intersecting edges before clipping, including a bow-tie whose
    // signed area cancels to zero. Adjacent edges share an allowed endpoint.
    for (std::size_t i = 0; i < count; ++i) for (std::size_t j = i + 1; j < count; ++j) {
        const auto nextI = (i + 1) % count, nextJ = (j + 1) % count;
        if (nextI == j || nextJ == i) continue;
        if (crossing(projected[i], projected[nextI], projected[j], projected[nextJ]))
            fail(filename, face.line, "self-intersecting or repeated polygon boundary");
    }
    double area = 0;
    for (std::size_t i = 1; i + 1 < count; ++i) area += turn(projected[0], projected[i], projected[i + 1]);
    if (std::abs(area) <= epsilon) fail(filename, face.line, "polygon is too thin to triangulate reliably");
    std::vector<std::size_t> ring(count);
    std::iota(ring.begin(), ring.end(), 0);
    const bool reverse = area < 0;
    if (reverse) std::reverse(ring.begin(), ring.end());
    std::vector<std::array<std::size_t, 3>> result;
    result.reserve(count - 2);
    // A fixed scan order also preserves the diagonal/corner order of the
    // project's existing quad assets. Never reorder already triangular faces.
    std::size_t cursor = ring.size() - 1;
    while (ring.size() > 2) {
        bool clipped = false;
        for (std::size_t tried = 0; tried < ring.size(); ++tried) {
            const auto a = ring[(cursor + ring.size() - 1) % ring.size()];
            const auto b = ring[cursor]; const auto c = ring[(cursor + 1) % ring.size()];
            bool ear = turn(projected[a], projected[b], projected[c]) > epsilon;
            if (ear) for (const auto other : ring) {
                if (other == a || other == b || other == c) continue;
                if (turn(projected[a], projected[b], projected[other]) >= -epsilon
                    && turn(projected[b], projected[c], projected[other]) >= -epsilon
                    && turn(projected[c], projected[a], projected[other]) >= -epsilon) { ear = false; break; }
            }
            if (ear) {
                result.push_back(reverse ? std::array<std::size_t, 3>{a, c, b} : std::array<std::size_t, 3>{a, b, c});
                ring.erase(ring.begin() + cursor);
                if (!ring.empty()) cursor = (cursor + 1) % ring.size();
                clipped = true;
                break;
            }
            cursor = (cursor + 1) % ring.size();
        }
        if (!clipped) fail(filename, face.line, "unable to triangulate polygon without losing corners");
    }
    return result;
}
}
