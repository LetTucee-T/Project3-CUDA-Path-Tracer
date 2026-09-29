cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/cameraSampling.h
    src/renderControls.h
    src/bvh.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/objLoader.h
    src/sceneStructs.h
    src/utilities.h
)

set(sources
    src/bvh.cpp
    src/main.cpp
    src/renderControls.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/objLoader.cpp
    src/utilities.cpp
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    #stream_compaction  # TODO: uncomment if using your stream compaction
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})


# CPU scene-loading checks do not create a CUDA context or an OpenGL window.
include(CTest)
if(BUILD_TESTING)
    add_executable(obj_loader_tests tests/obj_loader_tests.cpp src/objLoader.cpp)
    target_include_directories(obj_loader_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME obj_loader_cpu COMMAND obj_loader_tests "${CMAKE_CURRENT_BINARY_DIR}/Testing/obj_loader")

    add_executable(camera_sampling_tests tests/camera_sampling_tests.cpp)
    target_include_directories(camera_sampling_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_test(NAME camera_sampling_cpu COMMAND camera_sampling_tests)

    add_executable(camera_sampling_gpu_tests tests/camera_sampling_gpu_tests.cu)
    target_include_directories(camera_sampling_gpu_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(camera_sampling_gpu_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)
    add_test(NAME camera_sampling_gpu COMMAND camera_sampling_gpu_tests)
    set_tests_properties(camera_sampling_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

    add_executable(bvh_builder_tests
        tests/bvh_builder_tests.cpp
        src/bvh.cpp
    )
    target_include_directories(bvh_builder_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    add_test(NAME bvh_builder_cpu COMMAND bvh_builder_tests)
    add_executable(scene_loader_tests
        tests/scene_loader_tests.cpp
        src/bvh.cpp
        src/scene.cpp
        src/objLoader.cpp
        src/stb.cpp
        src/utilities.cpp
    )
    target_include_directories(scene_loader_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    add_test(NAME scene_loader_cpu
        COMMAND scene_loader_tests
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/Testing/mesh_cpu"
    )
    # Exercise the production CUDA kernels without creating an OpenGL window.
    add_executable(mesh_intersection_tests
        tests/mesh_intersection_tests.cu
        src/bvh.cpp
        src/pathtrace.cu
        src/intersections.cu
        src/interactions.cu
        src/scene.cpp
        src/objLoader.cpp
        src/stb.cpp
        src/utilities.cpp
    )
    target_include_directories(mesh_intersection_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    set_target_properties(mesh_intersection_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)
    target_compile_definitions(mesh_intersection_tests PRIVATE PATHTRACE_TESTING)
    target_compile_options(mesh_intersection_tests PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
    add_test(NAME mesh_intersection_gpu
        COMMAND mesh_intersection_tests "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(mesh_intersection_gpu PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 120)

    add_executable(thin_lens_render_tests
        tests/thin_lens_render_tests.cu
        src/bvh.cpp src/pathtrace.cu src/intersections.cu src/interactions.cu
        src/scene.cpp src/objLoader.cpp src/utilities.cpp src/stb.cpp)
    target_include_directories(thin_lens_render_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(thin_lens_render_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)
    target_compile_options(thin_lens_render_tests PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
    add_test(NAME thin_lens_render_gpu
        COMMAND thin_lens_render_tests "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(thin_lens_render_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)

    # Exercise the actual ImGui controls without an OS window or GL context.
    add_executable(render_controls_tests
        tests/render_controls_tests.cpp src/renderControls.cpp
        src/ImGui/imgui.cpp src/ImGui/imgui_draw.cpp
        src/ImGui/imgui_tables.cpp src/ImGui/imgui_widgets.cpp)
    target_include_directories(render_controls_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_test(NAME render_controls_cpu COMMAND render_controls_tests)

    add_test(NAME scene_loader_validate_cli
        COMMAND cis565_path_tracer
            "${CMAKE_CURRENT_SOURCE_DIR}/scenes/mesh_cpu_validation.json"
            --validate-scene
    )
endif()

# Reproducible scene captures with the production renderer, without an OpenGL window.
add_executable(scene_render tools/render_scene.cpp src/bvh.cpp src/pathtrace.cu
    src/intersections.cu src/interactions.cu src/scene.cpp src/objLoader.cpp src/utilities.cpp src/image.cpp src/stb.cpp)
target_include_directories(scene_render PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(scene_render PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)

if(BUILD_TESTING)
    add_executable(appearance_tests tests/appearance_tests.cpp src/scene.cpp src/objLoader.cpp src/bvh.cpp src/utilities.cpp src/stb.cpp)
    target_include_directories(appearance_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_test(NAME appearance_cpu COMMAND appearance_tests "${CMAKE_CURRENT_BINARY_DIR}/Testing/appearance")
endif()
