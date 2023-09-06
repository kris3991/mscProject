# cmake_minimum_required(VERSION 3.12)
# project(CUDA_Vector_Addition)

# set (CMAKE_CXX_STANDARD 17)

# # Find CUDA
# find_package(CUDA REQUIRED)

# # Add the CUDA_NVCC_FLAGS for setting GPU architecture (optional)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35)

# # Specify the source files (both the CUDA source file and header files)
# file(GLOB SOURCES
# src/vector_addition.cu
# src/main.cpp
# )
# file(GLOB HEADERS include/*.h)
# file(GLOB CUDA_HEADERS include/*.cuh)

# # Create an executable target
# cuda_add_executable(vector_addition ${SOURCES} ${HEADERS} ${CUDA_HEADERS})

# # Link CUDA libraries
# target_link_libraries(vector_addition ${CUDA_LIBRARIES})

# # Include the 'include' directory as a target include directory (if needed)
# # target_include_directories(vector_addition PRIVATE ${CMAKE_SOURCE_DIR}/include)

cmake_minimum_required(VERSION 3.12)
project(MScProject)

# Find CUDA
find_package(CUDA REQUIRED)

# Add the CUDA_NVCC_FLAGS for setting GPU architecture (optional)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

#common vars
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set (CMAKE_CXX_STANDARD 17)

#setup Imgui
set(IMGUIDIR "external/imgui")
include_directories(${IMGUIDIR} ${IMGUIDIR}/backends)
aux_source_directory("${IMGUIDIR}/" IMGUICORE)
set(IMGUIBACKENDS "${IMGUIDIR}/backends/imgui_impl_opengl3.cpp" 
  "${IMGUIDIR}/backends/imgui_impl_glfw.cpp"
  "${IMGUIDIR}/misc/cpp/imgui_stdlib.cpp"
)


# Specify the source files (only the CUDA source file)
set(SOURCES 
src/main.cpp
#test file to verify build
src/vector_addition.cu
#file management.
src/fileManager.cpp
#initilise
src/initialise.cpp
${IMGUICORE} 
${IMGUIBACKENDS}
#tiny file loader for dialogs.
external/tinyfiledialogs/tinyfiledialogs.c
#triangle class
src/triangle.cpp
#renderer class
src/renderer.cpp
#shaders.
src/shaderManager.cpp
)



# List all header files in the 'include' directory
file(GLOB HEADER_FILES ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/include/*.cuh)

# Create an executable target
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADER_FILES})

#add glfw
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(USE_MSVC_RUNTIME_LIBRARY_DLL OFF CACHE BOOL "" FORCE)
add_subdirectory("external/glfw-3.3.6")

#add glew
set(BUILD_UTILS OFF CACHE BOOL "" FORCE)
add_subdirectory("external/glew-2.1.0/build/cmake")

#add glm
add_subdirectory("external/glm")


# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} glfw opengl32 glew_s glm)

# Include the 'include' directory as a target include directory
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/external/glew-2.1.0/include)

# Create a custom target to display each header file individually in IDE
foreach(HEADER ${HEADER_FILES})
    get_filename_component(HEADER_NAME ${HEADER} NAME)
    source_group("Header Files" FILES ${HEADER})
endforeach()


add_custom_target(copy_shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources ${CMAKE_BINARY_DIR}/resources
)


add_dependencies(${PROJECT_NAME} copy_shaders)

    
    


# )

# add_custom_target(${PROJECT_NAME}
#     COMMAND vector_addition  # Replace with the test command you want to run
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     COMMENT "Running tests..."

#     BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/template

#         #install glfw
#         COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR}/external/glfw-3.3.6
#         --config $<$<CONFIG:Debug>:Debug> $<$<CONFIG:RelWithDebInfo>:RelWithDebInfo> 
#         $<$<CONFIG:Release>:Release>
#         --prefix ${CMAKE_CURRENT_SOURCE_DIR}/template/external/glfw-3.3.6

# )

# # Make the "run_tests" target depend on the "vector_addition" target
# add_dependencies(${PROJECT_NAME} vector_addition)