cmake_minimum_required(VERSION 3.8)
project(ATGlib)
 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
 

aux_source_directory(${CMAKE_SOURCE_DIR} DIR)

 
 

include_directories(./ATGlib/include)

 
 
link_directories(./ATGlib/lib)

 
 
 
 

add_library(LibATG  autograd.cpp
        operators.cpp
        optimizer.cpp
        variable.cpp)

 
 
#生成可执行文件
add_executable(Test   ${SOURCE_FILES} )
#比如：add_executable(hello_world    ${SOURCE_FILES})
 
 
target_link_libraries(LibATG  fmtd)