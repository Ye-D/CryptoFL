
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl_bind.h"
#include "pybind11/include/pybind11/stl.h"

#include <vector>

#include "cryptofl_aby.h"


namespace py = pybind11;
using namespace std;

typedef uint8_t uInt;
typedef int8_t Int;
typedef uint32_t uInt32;

PYBIND11_MAKE_OPAQUE(std::vector<Int>);
PYBIND11_MAKE_OPAQUE(std::vector<uInt>);
PYBIND11_MAKE_OPAQUE(std::vector<uInt32>);

PYBIND11_MODULE(cryptofl, m) {

    py::bind_vector<std::vector<Int>>(m, "VectorInt", py::buffer_protocol());
    py::bind_vector<std::vector<uInt>>(m, "VectoruInt", py::buffer_protocol());
    py::bind_vector<std::vector<uInt32>>(m, "VectoruInt32", py::buffer_protocol());

    m.doc() = "C++ module for cryptofl_lib";

    m.def("init_cryptofl_aby", &init_cryptofl_aby, "init_cryptofl_aby");

    m.def("shutdown_cryptofl_aby", &shutdown_cryptofl_aby, "shutdown_cryptofl_aby");


}
