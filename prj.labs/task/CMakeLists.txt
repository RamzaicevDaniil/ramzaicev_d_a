project(task)

add_executable(task task.cpp task.report.md.in.txt)
target_include_directories(task PRIVATE ../../json-develop)
target_link_libraries(task ${OpenCV_LIBS})

file(READ task.cpp cpp_source)
configure_file(task.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/task.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/task.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")