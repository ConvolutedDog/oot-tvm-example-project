# This bash script formats code using clang-format

THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

clang-format -style=file -i ${THIS_DIR}/src/*.cc
clang-format -style=file -i ${THIS_DIR}/include/*.h

clang-format -style=file -i ${THIS_DIR}/tests/cpp/include/*.h
clang-format -style=file -i ${THIS_DIR}/tests/cpp/src/*.cc
