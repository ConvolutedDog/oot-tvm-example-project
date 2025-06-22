# This bash script formats C++ code using clang-format

THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

find ${THIS_DIR}/.. -name "*.h" -exec clang-format -style=file -i {} \;
find ${THIS_DIR}/.. -name "*.cc" -exec clang-format -style=file -i {} \;
