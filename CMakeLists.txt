cmake_minimum_required(VERSION 3.12)
project(project_name)

set(CMAKE_CXX_STANDARD 17)

# 将 Python 头文件和库路径添加到 CMake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 添加子目录 src
add_subdirectory(src)