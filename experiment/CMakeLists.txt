project(LOAD_HIGHS LANGUAGES CXX)

set(HIGHS_DIR lib/cmake/highs)

find_package(HIGHS REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main highs::highs)
