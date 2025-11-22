cmake_minimum_required(VERSION 3.14)
# Название проекта берем из имени папки
project(aicomrad-whisper-gpu-daemon CXX)

set(CMAKE_CXX_STANDARD 17)

# --- 1. Включение модуля FetchContent ---
include(FetchContent)

# --- 2. Автоматическая загрузка httplib ---
FetchContent_Declare(
    cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    # Для стабильности можно использовать конкретный тег, например, v0.14.0
    GIT_TAG master
)
FetchContent_MakeAvailable(cpp_httplib)

# --- 3. Автоматическая загрузка whisper.cpp и настройка GPU ---
FetchContent_Declare(
    whisper_cpp
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG master
)
# Устанавливаем флаг для GPU (NVIDIA) ДО добавления подпроекта
set(GGML_CUDA ON CACHE BOOL "Enable CUDA support for GPU acceleration") # <--- Используйте GGML_CUDA
#set(WHISPER_CUBLAS ON CACHE BOOL "Enable cuBLAS (CUDA) support for GPU acceleration")
FetchContent_MakeAvailable(whisper_cpp)

# --- 4. Компиляция исполняемого файла ---
# Добавляем все исходные файлы
add_executable(whisper_service 
    main.cpp
    src/config.cpp
    src/audio_processor.cpp
    src/whisper_engine.cpp
    src/json_utils.cpp
)

# Добавляем инклюды для обеих зависимостей
target_include_directories(whisper_service PRIVATE
    ${cpp_httplib_SOURCE_DIR} # Путь к cpp-httplib/httplib.h
    ${whisper_cpp_SOURCE_DIR} # Путь к whisper.cpp/whisper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Наши заголовочные файлы
)

# Связываем исполняемый файл с библиотекой whisper
target_link_libraries(whisper_service
    # Основная библиотека whisper.cpp (цель называется 'whisper')
    whisper
)
