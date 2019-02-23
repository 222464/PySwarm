#pragma once

namespace pyswarm {
    class PyInt3 {
    private:
    public:
        int x, y, z;

        PyInt3() 
        : x(0), y(0), z(0)
        {}

        PyInt3(int X, int Y, int Z)
        : x(X), y(Y), z(Z)
        {}
    };
}