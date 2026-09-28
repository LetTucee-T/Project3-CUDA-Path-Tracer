#include "camera_sampling_test_cases.h"

int main()
{
    return cameraTest::runSuite([](const std::vector<cameraTest::Query>& queries) {
        std::vector<cameraTest::Result> results;
        for (const auto& query : queries) results.push_back(cameraTest::evaluate(query));
        return results;
    });
}
