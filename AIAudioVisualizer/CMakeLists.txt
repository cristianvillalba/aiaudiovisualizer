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
find_package(Vulkan REQUIRED)

#Add Xtensor
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

#Add portAudio
list(APPEND CMAKE_PREFIX_PATH /portaudioinstall/lib/cmake/portaudio)
#find_package(portaudio REQUIRED)
set(PORTAUDIO_PATH /portaudioinstall)
message(STATUS "<------------portaudio: ${PORTAUDIO_PATH}")

# Add source to this project's executable.
add_executable (AIAudioVisualizer "AIAudioVisualizer.cpp" "AIAudioVisualizer.h" "vk_mem_alloc.h" "vk_engine.cpp" "vk_engine.h" "vk_types.h" "vk_mesh.h" "vk_mesh.cpp" "vk_initializers.cpp" "vk_initializers.h" "VkBootstrap.h" "VkBootstrap.cpp" "VkBootstrapDispatch.h" "tiny_obj_loader.h" "imconfig.h" "imgui_internal.h" "imgui_draw.cpp" "imgui_tables.cpp" "imgui_widgets.cpp" "imstb_textedit.h" "imstb_rectpack.h" "imstb_truetype.h" "imgui.h" "imgui.cpp" "imgui_impl_sdl.h" "imgui_impl_sdl.cpp" "imgui_impl_vulkan.h" "imgui_impl_vulkan.cpp" "imgui_demo.cpp" "AudioFile.h" "Conversion.cpp" "DFT.cpp" "Domain.cpp" "FFT_fftpack.cpp" "Print.cpp" "Recorder.cpp" "Scheduler.cpp" "Soundfile.cpp" "Timer.cpp" "arr.cpp" "fftpack++1.cpp" "fftpack++2.cpp" "scl.cpp")

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


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET AIAudioVisualizer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
