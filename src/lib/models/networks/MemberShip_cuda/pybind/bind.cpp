#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"
#include "src/common.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  std::string name = std::string("Foo");
  py::class_<Foo>(m, name.c_str())
      .def(py::init<>())
      .def("setKey", &Foo::setKey)
      .def("getKey", &Foo::getKey)
      .def("__repr__", [](const Foo &a) { return a.toString(); });

  m.def("MemberShip_Forward_Wrapper", &MemberShip_Forward_Wrapper<float>);
  m.def("MemberShip_Input_Backward_Wrapper", &MemberShip_Input_Backward_Wrapper<float>);
  m.def("MemberShip_Center_Backward_Wrapper", &MemberShip_Center_Backward_Wrapper<float>);
  m.def("MemberShip_Lamda_Backward_Wrapper", &MemberShip_Lamda_Backward_Wrapper<float>);

}
