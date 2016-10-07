PROJECT(cnn:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(nt-parser nt-parser.cc eval.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser cnn ${Boost_LIBRARIES} z)

ADD_EXECUTABLE(nt-parser-gen nt-parser-gen.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser-gen cnn ${Boost_LIBRARIES} z)

