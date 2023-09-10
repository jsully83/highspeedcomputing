cmake_minimum_required(VERSION 3.23.2)
project(Edges VERSION 0.0.1 LANGUAGES CXX CUDA)

add_subdirectory(src)

# install
install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR})