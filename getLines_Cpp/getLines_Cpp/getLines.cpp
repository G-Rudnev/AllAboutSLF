#include "Python.h"
#include "numpy/arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions
#pragma warning(disable:4244)		//������� �� float � int
#include <iostream>
#include <cmath>

using namespace std;

PyArrayObject* pyPnts = nullptr;
PyArrayObject* pylines = nullptr;

double** pnts = nullptr;
double** lines = nullptr;

void lineApproxLMS(double* kbq, double** pnts, int fr, int to) {

}

void lineApproxAveraging(double* kbq, double** pnts, int fr, int to) {

}

PyObject* init(PyObject*, PyObject* o) {
    //����������� ������� Python pnts, lines � ������� pyPnts, pyLines
    //�������� ������ ��� ������� double** pnts, lines
    //��������� N �� Python
    //��������� 1 - ���� �����, 0 - ���� ���
    return PyLong_FromLong(1);
}

PyObject* getLines(PyObject*, PyObject* o) {
    //�������� ���������� pyPnts � pnts, �������, �������� ���������� lines � pyLines, ���������� Nlines
    return PyLong_FromLong(1);
}

static PyMethodDef getLines_methods[] = {
    { "init", (PyCFunction)init, METH_VARARGS, ""},
    { "getLines", (PyCFunction)getLines, METH_VARARGS, ""},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef getLines_getLines = {
    PyModuleDef_HEAD_INIT,
    "getLinesCpp",                        // Module name to use with Python import statements
    "SLF processing acceleration",  // Module description
    0,
    getLines_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_forLidar() {
    return PyModule_Create(&getLines_getLines);
}

PyMODINIT_FUNC PyFini_forLidar() {
    return PyModule_Create(&getLines_getLines);
}