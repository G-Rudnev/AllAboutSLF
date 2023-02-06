#define PY_SSIZE_T_CLEAN
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions

#include <windows.h>

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace std;

struct Device {

    size_t id = -1;

    thread thr; //�����, ������� ����� �������� �� ��������� �������

    mutex mxCall;   //������� �� ������� � ���������� ������� � �������
    mutex mxProcess;    //������� �� �������� ���������� ��������

    condition_variable cvCall; //��������������� cv
    condition_variable cvProcess;

    bool process = true;    //���� ���������� ��������

    bool alive = true;  //���� ����� ������ thr

    queue<void (Device::*) ()> foosQueue;   //������� ������� �������

    size_t Npnts = 0;  //in variable
    size_t Nlines = 0; //out variable

    double par0 = 0.0;  //���������
//  ...

    PyArrayObject* pyPntsXY = nullptr;  //��������� �� ������ ������� ������
    PyArrayObject* pyPntsPhi = nullptr;
    PyArrayObject* pyLinesXY = nullptr;

    double** pntsXY = nullptr;  //��������������� ������� �������� ������
    double* pntsPhi = nullptr;
    double** linesXY = nullptr;

    Device() = default;

    Device (PyArrayObject* const pyPntsXY, PyArrayObject* const pyPntsPhi, PyArrayObject* const pyLinesXY, size_t Npnts, PyObject* pyParams, size_t id) : 
        pyPntsXY(pyPntsXY), pyPntsPhi(pyPntsPhi), pyLinesXY(pyLinesXY), Npnts(Npnts), id(id)
    {

        if (!setParams(pyParams)) {
            throw new logic_error("");  //������ ��� �� ����������, ����� �� �������������� � �������
        }

        pntsXY = new double* [2];
        linesXY = new double* [2];
        for (int i = 0; i < 2; i++) {
            pntsXY[i] = new double[Npnts];
            linesXY[i] = new double[Npnts];
        }
        pntsPhi = new double[Npnts];

        thr = thread([this]     //��������� �����
            {
                while (this->alive) {

                    for (;;) {

                        unique_lock lk0(this->mxCall);
                        this->cvCall.wait(lk0, [this] {return !this->alive || this->foosQueue.size(); });   //���� ������ ����� pyFun
                        
                        if (this->alive) {

                            auto foo = this->foosQueue.front(); //����� ������� � ���� �������

                            if (foo != nullptr)
                                (this->*foo) ();    //�������� �

                            this->foosQueue.pop();  //������� �� �������

                            if (this->foosQueue.empty()) {
                                unique_lock lk1(this->mxProcess);   
                                this->process = true;   //������ ��������� ��� cv ��������, ������� pyFun<nullptr>, �.�. synchronize() �� ������, ������ �����������
                                lk1.unlock();
                                cvProcess.notify_one(); //���������� cv � ���������� ��������

                                cvCall.notify_one();
                                lk0.unlock();
                                break;
                            }
                        }
                        else {
                            cvCall.notify_one();
                            lk0.unlock();
                            break;
                        }
                        cvCall.notify_one();
                        lk0.unlock();
                    }
                }
            }
        );
    }

    ~Device() {
        //�� ����� ������ ������������, �� ����
        //������ ������ ������ ������ ������� � �� ������������, ������� ����� ���� ��������� �����-������ ���������� �������.
        //��� �� ������ ��� �����, ��� ��� �������������� ������������� ������� ��� ����� ������ ���������, ��, ��� ����� ���.
        //��� �� �����, ���� ������������� quit() ���� ���������� ����������.

        cout << "Device " << id << " destructor starts" << endl;

        unique_lock lk1(mxCall);    //���� ���������� ������ pyFun

        unique_lock lk0(mxProcess);
        cvProcess.wait(lk0, [this] {return this->process; }); //���� ���������� ��������� ������� �������
        lk0.unlock();

        alive = false;  //��������� ����� ������
        lk1.unlock();
        cvCall.notify_all();

        thr.join();

        cout << "Device " << id << " destructor ends normally" << endl;
    }

    bool setParams(PyObject* pyParams) {

        //����� � getParams() �������, �� ���� ��� �������������

        if (!PyTuple_Check(pyParams)) {
            cerr << "Parameters should be in a tuple" << endl;
            return false;
        }

        if (PyTuple_GET_SIZE(pyParams) != 1) {  //���-�� ����������, �� �������� ������
            cerr << "There should be 1 parameter in a tuple" << endl;
            return false;
        }

        par0 = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 0));  //����� ������� ����������, ������ ����� ���� ���������

        if (PyErr_Occurred()) {
            cerr << "Wrong type inside the parameters" << endl;
            PyErr_Clear();
            return false;
        }

        return true;
    }

    inline void pushPnts() {

        //���������� �� ����������� ���������� PUSH ������� � ��������� ��������, ����� ����������� � �� ���� �����,
        //��������� �� ���� ������ (pXY_0r, ..) ����� � � ������ ��������, ����������� � ���������� ������������ ��������� � ������.
        //��� ����������� ����� � ����, � � ��������� �������� ����� ����������� ������ - ����� �������� �� ���������� ��� ������� �������.

        //cout << "pushPnts() on id " << id << endl;

        double* pXY_0r = (double*)PyArray_DATA(pyPntsXY);   //0-� ���
        double* pXY_1r = pXY_0r + Npnts;                    //1-� ���
        double* pPhi = (double*)PyArray_DATA(pyPntsPhi);

        for (size_t i = 0; i < Npnts; i++) {
            pntsXY[0][i] = *(pXY_0r + i);
            pntsXY[1][i] = *(pXY_1r + i);
            pntsPhi[i] = *(pPhi + i);
        }

        //� ���������� �������� (���������)
        //for (size_t i = 0; i < Npnts; i++) {
        //    pntsXY[0][i] = *(double*)PyArray_GETPTR2(pyPntsXY, 0, i);
        //    pntsXY[1][i] = *(double*)PyArray_GETPTR2(pyPntsXY, 1, i);
        //    pntsPhi[i] = *(double*)PyArray_GETPTR1(pyPntsPhi, i);
        //}
    }

    void calcLines() {
        //����� ��� ����������� Nlines, ����� ����� ��������� pyParams.
        //�������� ���� �� ��������� - ������� �� ������� ��� �� ����� �������.
 
        //cout << "pullLinesXY() on id " << id << endl;

        Nlines = Npnts;
        double* pLines_0r = (double*)PyArray_DATA(pyLinesXY);
        double* pLines_1r = pLines_0r + Nlines;

        for (size_t i = 0; i < Nlines; i++) {
            *(pLines_0r + i) = exp(pntsXY[0][i] * pntsXY[0][i] * pntsPhi[i]);
            *(pLines_1r + i) = exp(pntsXY[1][i] * pntsXY[1][i] * pntsPhi[i]);
        }

        //� ���������� �������� (���������)
        //for (size_t i = 0; i < Nlines; i++) {
        //    *(double*)PyArray_GETPTR2(pyLinesXY, 0, i) = pntsXY[0][i] * pntsXY[0][i] * pntsPhi[i];
        //    *(double*)PyArray_GETPTR2(pyLinesXY, 1, i) = pntsXY[1][i] * pntsXY[1][i] * pntsPhi[i];
        //}
    }
};

//�������� ���������� ��������� ��������� (���������� �� ���)
vector<unique_ptr<Device>> devices; //� unique_ptr ����� ����������� ���������� �� ���������� ���������� ����������

template<void (Device::* F) ()>
PyObject* pyFun(PyObject*, PyObject* o) {

    if (PyLong_Check(o)) {
        int id = (int)PyLong_AsLong(o);
        if (id >= 0 && id < devices.size()) {

            Device* dev = devices[id].get();

            if (F != nullptr) {
                unique_lock lk(dev->mxCall);
                dev->process = false;   //������ ���� ����������� ��������
                dev->foosQueue.push(F);     //��������� ������� � �������
                lk.unlock();
                dev->cvCall.notify_one();   //����������, ����������� ������� � wait ����������� size() jxthtlb
            }
            else {
                unique_lock lk(dev->mxProcess); //�������������, ����� �� ���� ���������� �������� �� �����
                dev->cvProcess.wait(lk, [dev] {return dev->process; });
                lk.unlock();
            }

            return PyLong_FromLong(0);
        }
        cerr << "Incorrect id " << id << " in " << devices.size() << " devices" << endl;
        return PyLong_FromLong(-1);
    }

    cerr << "Incorrect args" << endl;
    return PyLong_FromLong(-1);
}

PyObject* init(PyObject*, PyObject* o) {

    //����� ���������� ��������� �������, ����� ���������� � � quit() � �� reinit() - ��� ������� �������, ��� � ����������� �����������.
    //����� - ��������, �������� � �������� �������, ���� ��� ��
    //����������� � ����� id

    if (PyTuple_GET_SIZE(o) == 5) {

        PyArrayObject* const pyPntsXY_ = (PyArrayObject*)PyTuple_GetItem(o, 0);
        PyArrayObject* const pyPntsPhi_ = (PyArrayObject*)PyTuple_GetItem(o, 1);
        PyArrayObject* const pyLinesXY_ = (PyArrayObject*)PyTuple_GetItem(o, 2);

        if (!PyLong_Check(PyTuple_GetItem(o, 3))) {
            cerr << "Bad N of points" << endl;
            return PyLong_FromLong(-1);
        }

        size_t N = PyLong_AsLongLong(PyTuple_GetItem(o, 3));
        PyObject* pyParams = PyTuple_GetItem(o, 4);

        if (PyArray_NDIM(pyPntsXY_) != 2 &&
            PyArray_NDIM(pyPntsPhi_) != 1 &&
            PyArray_NDIM(pyLinesXY_) != 2) {
            cerr << "Wrong data dimensions" << endl;
            return PyLong_FromLong(-1);
        }

        if (!PyTuple_Check(pyParams)) {
            cerr << "Wrong parameters arg - must be a tuple" << endl;
            return PyLong_FromLong(-1);
        }

        if (PyErr_Occurred()) {
            cerr << "Bad arguments" << endl;
            PyErr_Clear();
            return PyLong_FromLong(-1);
        }

        try {
            devices.push_back(unique_ptr<Device>(new Device(pyPntsXY_, pyPntsPhi_, pyLinesXY_, N, pyParams, devices.size())));
            return PyLong_FromLongLong(devices.size() - 1);
        }
        catch (exception const*) {
            return PyLong_FromLong(-1);
        }
    }

    cerr << "Incorrect args number" << endl;
    return PyLong_FromLong(-1);
}

static PyMethodDef lidarVector_methods[] = {
    { "init", (PyCFunction)init, METH_VARARGS, ""},
    { "pushPnts", (PyCFunction)pyFun<&Device::pushPnts>, METH_O, ""},
    { "calcLines", (PyCFunction)pyFun<&Device::calcLines>, METH_O, ""},
    { "synchronize", (PyCFunction)pyFun<nullptr>, METH_O, ""},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef lidarVector_module = {
    PyModuleDef_HEAD_INIT,
    "lidarVector",                        // Module name to use with Python import statements
    "Lidar vectorization activity",     // Module description
    0,
    lidarVector_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_lidarVector() {
    return PyModule_Create(&lidarVector_module);
}