%begin %{
#include <cmath>
#include <iostream>
%}
%module pyswarm

%include "std_array.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "PyConstructs.h"
#include "PyComputeSystem.h"
#include "PyHierarchy.h"
%}

// Handle STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%template(StdVeci) std::vector<int>;
%template(StdVecf) std::vector<float>;
%template(StdVecInt3) std::vector<pyswarm::PyInt3>;
%template(StdVecLayerDesc) std::vector<pyswarm::PyLayerDesc>;

%rename("%(strip:[Py])s") ""; // Remove Py prefix that was added to avoid naming collisions

%include "PyConstructs.h"
%include "PyComputeSystem.h"
%include "PyHierarchy.h"