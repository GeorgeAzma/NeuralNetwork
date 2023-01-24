cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(vendor)

file(GLOB FILES "src/*.cpp")
add_executable(NeuralNetwork ${FILES})

# Copy data folder where exe file is
add_custom_command(TARGET NeuralNetwork POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/data
    $<TARGET_FILE_DIR:NeuralNetwork>/data)