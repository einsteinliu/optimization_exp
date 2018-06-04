find_package(OpenCV 3.0 QUIET)

ADD_EXECUTABLE(multi_sensor_fusion
  multi_sensor_fusion.cpp
)

INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(multi_sensor_fusion core solver_cholmod types_sba types_slam3d_addons types_slam3d)


