cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
file(GLOB SOURCE_FILES *.cu)

foreach(SOURCE_PATH ${SOURCE_FILES})

    get_filename_component(EXECUTABLE_NAME ${SOURCE_PATH} NAME_WE)

    
    add_executable(${EXECUTABLE_NAME} ${SOURCE_PATH})

    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES "60")

    #install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)

endforeach(SOURCE_PATH ${SOURCE_FILES})

