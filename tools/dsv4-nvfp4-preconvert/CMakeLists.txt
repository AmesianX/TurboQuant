set(TARGET llama-dsv4-nvfp4-preconvert)
add_executable(${TARGET} dsv4-nvfp4-preconvert.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
# dsv4_moe_grouped_convert_layer lives in libggml-cuda; link it directly.
if (TARGET ggml-cuda)
    target_link_libraries(${TARGET} PRIVATE ggml-cuda)
endif()
# for "ggml-cuda/dsv4-moe-grouped-blob.h"
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
