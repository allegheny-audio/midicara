cmake_minimum_required (VERSION 3.2)
project (TryDLib)

find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED gtk+-3.0)
include_directories(${GTK_INCLUDE_DIRS})

add_executable(feed feed.cpp)
target_link_libraries(feed dlib::dlib ${OpenCV_LIBS} ${GTK_LIBRARIES})
