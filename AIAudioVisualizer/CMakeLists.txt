# CMakeList.txt : CMake project for AIAudioVisualizer
#
cmake_minimum_required (VERSION 3.8)

#Add SDL2
list(APPEND CMAKE_PREFIX_PATH /SDLsdk/cmake)
set(SDL2_PATH /SDLsdk)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
message(STATUS "<------------SDL libs: ${SDL2_LIBRARIES}")

#Add Vulkan
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
#set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
#set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)


#Add Xtensor
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

#Add portAudio
list(APPEND CMAKE_PREFIX_PATH /portaudioinstall/lib/cmake/portaudio)
#find_package(portaudio REQUIRED)
set(PORTAUDIO_PATH /portaudioinstall)
message(STATUS "<------------portaudio: ${PORTAUDIO_PATH}")

# Add source to this project's executable.
add_executable (AIAudioVisualizer "AIAudioVisualizer.cpp" "AIAudioVisualizer.h" "vk_mem_alloc.h" "vk_engine.cpp" "vk_engine.h" "vk_types.h" "vk_mesh.h" "vk_mesh.cpp" "vk_initializers.cpp" "vk_initializers.h" "VkBootstrap.h" "VkBootstrap.cpp" "VkBootstrapDispatch.h" "tiny_obj_loader.h" "imconfig.h" "imgui_internal.h" "imgui_draw.cpp" "imgui_tables.cpp" "imgui_widgets.cpp" "imstb_textedit.h" "imstb_rectpack.h" "imstb_truetype.h" "imgui.h" "imgui.cpp" "imgui_impl_sdl.h" "imgui_impl_sdl.cpp" "imgui_impl_vulkan.h" "imgui_impl_vulkan.cpp" "imgui_demo.cpp" "AudioFile.h" "Conversion.cpp" "DFT.cpp" "Domain.cpp" "FFT_fftpack.cpp" "Print.cpp" "Recorder.cpp" "Scheduler.cpp" "Soundfile.cpp" "Timer.cpp" "arr.cpp" "fftpack++1.cpp" "fftpack++2.cpp" "scl.cpp" "imgui_ai.h" "portAudioStructs.h")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})#add all files from the current source dir
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Gamma) #include Gamma headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/lib) #include tensorflow headers
include_directories(${xtensor_INCLUDE_DIRS}) #include xtensor headers
include_directories(${xtl_INCLUDE_DIRS}) #include xtensor headers
include_directories(${PORTAUDIO_PATH}/include) #include portaudio headers

target_link_libraries(AIAudioVisualizer PRIVATE ${SDL2_LIBRARIES})
target_link_libraries(AIAudioVisualizer PRIVATE Vulkan::Vulkan)
target_link_libraries(AIAudioVisualizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/tensorflow.lib)
target_link_libraries(AIAudioVisualizer PRIVATE xtl)
target_link_libraries(AIAudioVisualizer PRIVATE xtensor)
target_link_libraries(AIAudioVisualizer PRIVATE ${PORTAUDIO_PATH}/lib/portaudio_x64.lib)

#==============================================================================
# COMPILE SHADERS
#
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
  set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator.exe")
endif()

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "shaders/*.frag"
    "shaders/*.vert"
    )

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/shaders/"
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

add_dependencies(AIAudioVisualizer Shaders)

add_custom_command(TARGET AIAudioVisualizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:AIAudioVisualizer>/shaders/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_BINARY_DIR}/shaders"
        "$<TARGET_FILE_DIR:AIAudioVisualizer>/shaders"
        )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AIAudioVisualizer PROPERTY CXX_STANDARD 20)
endif()

