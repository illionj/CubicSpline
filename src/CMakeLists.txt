file(GLOB_RECURSE FILES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc" "*.cpp" "*.h")
if(SRC STREQUAL "")
    message(FATAL_ERROR "No source files (.cc/.cpp) found for cubic_spline!")
endif()

add_library(cubic_spline  SHARED ${FILES})
target_include_directories(cubic_spline 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# target_include_directories(cubic_spline
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
#     $<INSTALL_INTERFACE:include>
# )

# 编库时告诉头文件 “我在导出”
target_compile_definitions(cubic_spline PRIVATE TS_BUILD_DLL)

target_link_libraries(cubic_spline eigen)
