# local libs
add_library(cpuconv cputools.cpp)
target_include_directories(cpuconv PUBLIC ${PROJECT_SOURCE_DIR}/include)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")
MESSAGE(STATUS "cpuconv")

add_executable(${PROJECT_NAME} main.cpp cputools.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} cpuconv)