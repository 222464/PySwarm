#pragma once

namespace pyswarm {
    class PyInt3 {
    private:
    public:
        int x, y, z;

        PyInt3() 
        : x(0), y(0), z(0)
        {}

        PyInt3(int x, int y, int z)
        : x(x), y(y), z(z)
        {}
    };
}