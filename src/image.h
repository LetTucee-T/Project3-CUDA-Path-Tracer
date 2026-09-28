#pragma once

#include <glm/glm.hpp>

#include <string>

class Image
{
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename, bool displayTransform = false, float exposure = 0.0f);
    void saveHDR(const std::string &baseFilename);
};
