#define PY_SSIZE_T_CLEAN
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy\\arrayobject.h"

#pragma warning(disable:4996)		//unsafe functions

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace std;

typedef struct {
    double x;
    double y;
    long shift;
} getPnt_t;

struct Device {

    queue<void (Device::*) (PyObject* o)> foosQueue;   //очередь функций объекта
    queue<PyObject*> argsQueue;   //очередь аргументов функций

    thread thr; //поток, который будет отвечать за обработку объекта

    condition_variable cvCall; //соответствующие cv
    condition_variable cvProcess;

    mutex mxCall;   //мьютекс по вызовам и добавлению функций в очередь
    mutex mxProcess;    //мьютекс по контролю завершения расчетов

    bool process = true;    //флаг завершения расчетов

    PyArrayObject* pyPolyline;  //указатели на нужные объекты питона
    PyArrayObject* pyCheckList;

    PyArrayObject* pyNlines;

    double* pPoly_x = nullptr;
    double* pPoly_y = nullptr;
    double* pCheckList_i = nullptr;
    double* pCheckList_t = nullptr;
    double* pCheckList_isGap = nullptr;

    double par0;

    long* pNlines = nullptr;
    long maxNlines;

    long id;

    Device() = default;

    Device(PyArrayObject* const pyPolyline, PyArrayObject* const pyCheckList, PyArrayObject* const pyNlines,
        PyObject* pyParams, long id) :
        pyPolyline(pyPolyline), pyCheckList(pyCheckList), pyNlines(pyNlines),
        id(id)
    {

        if (!setParams(pyParams)) {
            throw new logic_error("");
        }

        //связывание с объектами Py

        pNlines = (long*)PyArray_DATA(pyNlines);

        pPoly_x = (double*)PyArray_DATA(pyPolyline);   //0-й ряд 
        pPoly_y = pPoly_x + maxNlines;               //1-й ряд

        pCheckList_i = (double*)PyArray_DATA(pyCheckList);
        pCheckList_t = pCheckList_i + maxNlines;
        pCheckList_isGap = pCheckList_t + maxNlines;

        thr = thread([this]     //запускаем поток
            {
                while (true) {
                    unique_lock lk0(mxCall);
                    cvCall.wait(lk0, [this] {return foosQueue.size(); });   //ждун вызова через pyFun

                    auto foo = foosQueue.front(); //берем функцию с края очереди
                    auto args = argsQueue.front(); //берем аргументы с края очереди

                    if (foo != nullptr && args != nullptr)
                        (this->*foo) (args);    //вызываем функцию с соотв. аргументом

                    foosQueue.pop();  //удаляем из очереди функцию
                    argsQueue.pop();  //удаляем из очереди аргументы

                    if (foosQueue.empty()) {
                        unique_lock lk1(mxProcess);
                        process = true;   //меняем состояние для cv процесса, функция pyFun<nullptr>, т.е. synchronize() из питона, сможет завершиться
                        lk1.unlock();
                        cvProcess.notify_one(); //уведомляем cv о завершении расчетов
                    }

                    lk0.unlock();
                }
            }
        );
    }

    ~Device() {
        //НЕ ОСОБО НУЖНОЕ УДОВОЛЬСТВИЕ, НО ЕСТЬ
        //Питона крашит модули весьма кусочно и не консистентно, поэтому может даже подвесить какую-нибудь работающую функцию.
        //Нам не сильно это важно, так как предполагается существование объекта все время работы программы, ну, или почти все.
        //Тем не менее, если разрабатывать quit() этот деструктор пригодится.

        cout << "Device " << id << " destructor starts" << endl;

        unique_lock lk0(mxProcess);
        cvProcess.wait(lk0, [this] { return process; }); //ждет завершения расчетной очереди функций
        lk0.unlock();

        thr.join(); //оказывается, компилятор и сам завершит поток в бесконечном цикле при разрушении cv

        cout << "Device " << id << " destructor ends normally" << endl;
    }

    getPnt_t getPnt(long i, long increment = 1)
    {
        long shift = 0;
        long n;

        while (true)
        {

            n = (i + shift) % *pNlines;
            n < 0 ? n += *pNlines : n;

            if (abs(pPoly_x[n]) > 1e-5 || abs(pPoly_y[n]) > 1e-5)
                return { pPoly_x[n], pPoly_y[n], shift };

            shift += increment;
        }
    }

    bool setParams(PyObject* pyParams) {

        //Можно и getParams() сделать, но пока это необязательно

        if (!PyTuple_Check(pyParams)) {
            cerr << "Parameters should be in a tuple" << endl;
            return false;
        }

        if (PyTuple_GET_SIZE(pyParams) != 1) {  //кол-во параметров, не забываем менять
            cerr << "There should be 1 parameters in a tuple" << endl; // 5-й - кортеж координат монтажного положения устройства
            return false;
        }

        maxNlines = PyLong_AsLong(PyTuple_GetItem(pyParams, 0));

        if (PyErr_Occurred()) {
            cerr << "Wrong type inside the parameters" << endl;
            PyErr_Clear();
            return false;
        }

        return true;
    }

    void closest_pnt(PyObject* args)
    {
        /*
        *   args[0] - id
            pnt : np.ndarray,
            vx_from : long,
            vx_to : long,
            minDist : float,
            onlyGaps : bool = False,
            args[6] - returns_tuple (ссылка на кортеж "возвращаемых" значений)

        */

        // Сначала идёт извлечение всех аргументов
        if (PyTuple_GET_SIZE(args) != 7) {  //кол-во параметров, не забываем менять
            cerr << "There should be 7 parameters in the args tuple" << endl;
            return;
        }


        double* pnt = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(args, 1));
        long vx_from = PyLong_AsLong(PyTuple_GetItem(args, 2));
        long vx_to = PyLong_AsLong(PyTuple_GetItem(args, 3));
        double minDist = (double)PyFloat_AsDouble(PyTuple_GetItem(args, 4));
        bool onlyGaps = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 5));
        PyObject* returns_tuple = PyTuple_GetItem(args, 6);

        long seg_i = -1;

        // достаём нули из питона (в кортеже лежит np.zeros(2))
        double* py_closest = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(returns_tuple, 1));

        double* closestPnt_ = new double[2];
        closestPnt_[0] = 0.0;
        closestPnt_[1] = 0.0;

        vector<double> seg_r = { 0.0, 0.0 };
        vector<double> v = { 0.0, 0.0 };
        vector<double> r = { 0.0, 0.0 };

        double val;
        double t;
        double dist;

        while (vx_from < vx_to)
        {
            getPnt_t p1 = getPnt(vx_from + 1);
            if (!onlyGaps)
            {
                if (p1.shift != 0)
                {
                    vx_from += (p1.shift + 1);
                    continue;
                }
            }

            else if (p1.shift == 0)
            {
                vx_from++;
                continue;
            }

            getPnt_t p0 = getPnt(vx_from);

            seg_r[0] = p1.x - p0.x;
            seg_r[1] = p1.y - p0.y;

            v[0] = pnt[0] - p0.x;
            v[1] = pnt[1] - p0.y;

            val = inner_product(begin(seg_r), end(seg_r), begin(seg_r), 0.0);
            if (val < 1e-4)
                t = 0.0;
            else
            {
                t = inner_product(begin(v), end(v), begin(seg_r), 0.0) / val;
                if (t < 0.0)
                    t = 0.0;
                else if (t > 1.0)
                    t = 1.0;
            }

            closestPnt_[0] = p0.x + t * seg_r[0];
            closestPnt_[1] = p0.y + t * seg_r[1];
            r[0] = pnt[0] - closestPnt_[0];
            r[1] = pnt[1] - closestPnt_[1];

            dist = sqrt(r[0] * r[0] + r[1] * r[1]);
            if (dist < minDist - 1e-9)
            {
                seg_i = vx_from + p1.shift;
                py_closest[0] = closestPnt_[0];
                py_closest[1] = closestPnt_[1];
                minDist = dist;
            }

            vx_from += (p1.shift + 1);
        }

        // заполнение кортежа выходными данными
        PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(seg_i));
        // координаты либо уже положены в цикле, либо так и остались нулями
        /*py_closest[0] = closestPnt_[0];
        py_closest[1] = closestPnt_[1];*/

        PyTuple_SET_ITEM(returns_tuple, 2, PyFloat_FromDouble(minDist));

    }

    void closest_pnt00(PyObject* args)
    {
        /*
          *   args[0] - id
              vx_from : long,
              vx_to : long,
              minDist : float,
              onlyGaps : bool = False,
              args[5] - returns_tuple (ссылка на кортеж "возвращаемых" значений)

          */

          // Сначала идёт извлечение всех аргументов
        if (PyTuple_GET_SIZE(args) != 6) {  //кол-во параметров, не забываем менять
            cerr << "There should be 6 parameters in the args tuple" << endl;
            return;
        }

        long vx_from = PyLong_AsLong(PyTuple_GetItem(args, 1));
        long vx_to = PyLong_AsLong(PyTuple_GetItem(args, 2));
        double minDist = (double)PyFloat_AsDouble(PyTuple_GetItem(args, 3));
        bool onlyGaps = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 4));
        PyObject* returns_tuple = PyTuple_GetItem(args, 5);

        long seg_i = -1;

        // достаём нули из питона (в кортеже лежит np.zeros(2))
        double* py_closest = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GET_ITEM(returns_tuple, 1));

        double* closestPnt_ = new double[2];
        closestPnt_[0] = 0.0;
        closestPnt_[1] = 0.0;

        vector<double> seg_r = { 0.0, 0.0 };

        double val;
        double t;
        double dist;

        while (vx_from < vx_to)
        {
            getPnt_t p1 = getPnt(vx_from + 1);
            if (!onlyGaps)
            {
                if (p1.shift != 0)
                {
                    vx_from += (p1.shift + 1);
                    continue;
                }
            }

            else if (p1.shift == 0)
            {
                vx_from++;
                continue;
            }

            getPnt_t p0 = getPnt(vx_from);

            seg_r[0] = p1.x - p0.x;
            seg_r[1] = p1.y - p0.y;

            val = inner_product(begin(seg_r), end(seg_r), begin(seg_r), 0.0);
            if (val < 1e-4)
                t = 0.0;
            else
            {
                t = (-p0.x * seg_r[0] - p0.y * seg_r[1]) / val;
                if (t < 0.0)
                    t = 0.0;
                else if (t > 1.0)
                    t = 1.0;
            }

            closestPnt_[0] = p0.x + t * seg_r[0];
            closestPnt_[1] = p0.y + t * seg_r[1];

            dist = sqrt(closestPnt_[0] * closestPnt_[0] + closestPnt_[1] * closestPnt_[1]);
            if (dist < minDist - 1e-9)
            {
                seg_i = vx_from + p1.shift;
                py_closest[0] = closestPnt_[0];
                py_closest[1] = closestPnt_[1];
                minDist = dist;
            }

            vx_from += (p1.shift + 1);
        }


        // заполнение кортежа выходными данными
        PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(seg_i));

        /*py_closest[0] = closestPnt_[0];
        py_closest[1] = closestPnt_[1];*/

        PyTuple_SET_ITEM(returns_tuple, 2, PyFloat_FromDouble(minDist));
    }

    void check_segment_intersections(PyObject* args)
    {
        /*
        *   args[0] - id
            p0 : np.ndarray,
            p1 : np.ndarray,
            vx_from : long,
            vx_to : long,
            checkAll : bool = False,
            num : long = 0,
            ignoreGaps : bool = False
            args[8] - returns_tuple (ссылка на кортеж "возвращаемых" значений)
        */

        // Сначала идёт извлечение всех аргументов
        if (PyTuple_GET_SIZE(args) != 9) {  //кол-во параметров, не забываем менять
            cerr << "There should be 9 parameters in the args tuple" << endl;
            return;
        }

        double* p0 = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(args, 1));
        double* p1 = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(args, 2));
        long vx_from = PyLong_AsLong(PyTuple_GetItem(args, 3));
        long vx_to = PyLong_AsLong(PyTuple_GetItem(args, 4));
        bool checkAll = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 5));
        long num = PyLong_AsLong(PyTuple_GetItem(args, 6));
        bool ignoreGaps = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 7));
        PyObject* returns_tuple = PyTuple_GetItem(args, 8);

        double* px = new double[2];
        vector<double> seg_r = { (p1[0] - p0[0]) / 2.0, (p1[1] - p0[1]) / 2.0 };
        vector<double> seg_c = { p0[0] + seg_r[0], p0[1] + seg_r[1] };
        double segLen = 2.0 * sqrt(seg_r[0] * seg_r[0] + seg_r[1] * seg_r[1]);

        vector<double> lines_seg_r = { 0.0, 0.0 };

        vector<double> L0 = { 0.0, 0.0 };
        vector<double> L1 = { 0.0, 0.0 };

        vector<double> T = { 0.0, 0.0 };

        if (checkAll)
        {
            px[0] = 0.0;
            px[1] = 0.0;
        }

        if (abs(seg_r[0]) > 1e-6)
        {
            L0[0] = seg_r[1] / seg_r[0];
            L0[1] = -1.0;
        }

        else
        {
            L0[0] = -1.0;
            L0[1] = 0.0;
        }

        while (vx_from < vx_to)
        {
            getPnt_t p3 = getPnt(vx_from + 1);

            if (ignoreGaps && p3.shift != 0)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            getPnt_t p2 = getPnt(vx_from);

            lines_seg_r[0] = (p3.x - p2.x) / 2.0;
            lines_seg_r[1] = (p3.y - p2.y) / 2.0;

            T[0] = p2.x + lines_seg_r[0] - seg_c[0];
            T[1] = p2.y + lines_seg_r[1] - seg_c[1];

            // L0 along segment normal

            if (abs(inner_product(begin(T), end(T), begin(L0), 0.0)) - abs(inner_product(begin(lines_seg_r), end(lines_seg_r), begin(L0), 0.0)) > -1e-12)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            // L1 along lines_segment normal

            if (abs(lines_seg_r[0]) > 1e-6)
            {
                L1[0] = lines_seg_r[1] / lines_seg_r[0];
                L1[1] = -1.0;
            }

            else
            {
                L1[0] = -1.0;
                L1[1] = 0.0;
            }

            if (abs(inner_product(begin(T), end(T), begin(L1), 0.0)) - abs(inner_product(begin(seg_r), end(seg_r), begin(L1), 0.0)) > -1e-12)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            if (!checkAll)
            {
                PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(1));
                return;
            }

            if (abs(L0[1]) > 1e-6)
            {
                double k1 = -L0[0] / L0[1]; // line 1
                if (abs(L1[1]) > 1e-6)
                {
                    double k2 = -L1[0] / L1[1];
                    double b1 = p0[1] - k1 * p0[0];
                    double b2 = p2.y - k2 * p2.x;
                    px[0] = (b2 - b1) / (k1 - k2);
                    px[1] = k1 * px[0] + b1;
                }

                else
                {
                    px[0] = getPnt(vx_from).x;
                    px[1] = k1 * (p2.x - p0[0]) + p0[1];
                }
            }

            else
            {
                px[0] = p0[0]; // b1
                px[1] = -L1[0] / L1[1] * (p0[0] - p2.x) + p2.y; // k2 * b1 + b2
            }

            pCheckList_i[num] = (double)vx_from;
            pCheckList_t[num] = (double)sqrt((px[0] - p0[0]) * (px[0] - p0[0]) + (px[1] - p0[1]) * (px[1] - p0[1])) / segLen;
            pCheckList_isGap[num] = (double)(p3.shift != 0);

            num++;
            vx_from += (p3.shift + 1);

        }

        // заполнение кортежа выходными данными (num)
        PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(num));
    }

    void check_segment_intersections00(PyObject* args)
    {
        /*
        *   args[0] - id
            p1 : np.ndarray,
            vx_from : long,
            vx_to : long,
            checkAll : bool = False,
            num : long = 0,
            ignoreGaps : bool = False
            args[7] - returns_tuple (ссылка на кортеж "возвращаемых" значений)
        */

        // Сначала идёт извлечение всех аргументов
        if (PyTuple_GET_SIZE(args) != 8) {  //кол-во параметров, не забываем менять
            cerr << "There should be 8 parameters in the args tuple" << endl;
            return;
        }

        double* p1 = (double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(args, 1));
        long vx_from = PyLong_AsLong(PyTuple_GetItem(args, 2));
        long vx_to = PyLong_AsLong(PyTuple_GetItem(args, 3));
        bool checkAll = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 4));
        long num = PyLong_AsLong(PyTuple_GetItem(args, 5));
        bool ignoreGaps = (bool)PyObject_IsTrue(PyTuple_GetItem(args, 6));
        PyObject* returns_tuple = PyTuple_GetItem(args, 7);

        double* px = new double[2];
        vector<double> seg_r = { p1[0] / 2.0, p1[1] / 2.0 };
        double segLen = 2.0 * sqrt(seg_r[0] * seg_r[0] + seg_r[1] * seg_r[1]);

        vector<double> lines_seg_r = { 0.0, 0.0 };

        vector<double> L0 = { 0.0, 0.0 };
        vector<double> L1 = { 0.0, 0.0 };

        vector<double> T = { 0.0, 0.0 };

        if (checkAll)
        {
            px[0] = 0.0;
            px[1] = 0.0;
        }

        if (abs(seg_r[0]) > 1e-6)
        {
            L0[0] = seg_r[1] / seg_r[0];
            L0[1] = -1.0;
        }

        else
        {
            L0[0] = -1.0;
            L0[1] = 0.0;
        }

        while (vx_from < vx_to)
        {
            getPnt_t p3 = getPnt(vx_from + 1);

            if (ignoreGaps && p3.shift != 0)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            getPnt_t p2 = getPnt(vx_from);

            lines_seg_r[0] = (p3.x - p2.x) / 2.0;
            lines_seg_r[1] = (p3.y - p2.y) / 2.0;

            T[0] = p2.x + lines_seg_r[0] - seg_r[0];
            T[1] = p2.y + lines_seg_r[1] - seg_r[1];

            // L0 along segment normal

            if (abs(inner_product(begin(T), end(T), begin(L0), 0.0)) - abs(inner_product(begin(lines_seg_r), end(lines_seg_r), begin(L0), 0.0)) > -1e-12)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            // L1 along lines_segment normal

            if (abs(lines_seg_r[0]) > 1e-6)
            {
                L1[0] = lines_seg_r[1] / lines_seg_r[0];
                L1[1] = -1.0;
            }

            else
            {
                L1[0] = -1.0;
                L1[1] = 0.0;
            }

            if (abs(inner_product(begin(T), end(T), begin(L1), 0.0)) - abs(inner_product(begin(seg_r), end(seg_r), begin(L1), 0.0)) > -1e-12)
            {
                vx_from += (p3.shift + 1);
                continue;
            }

            if (!checkAll)
            {
                PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(1));
                return;
            }

            if (abs(L0[1]) > 1e-6)
            {
                double k1 = -L0[0] / L0[1]; // line 1
                if (abs(L1[1]) > 1e-6)
                {
                    double k2 = -L1[0] / L1[1];
                    px[0] = (p2.y - k2 * p2.x) / (k1 - k2);
                    px[1] = k1 * px[0];
                }

                else
                {
                    px[0] = getPnt(vx_from).x;
                    px[1] = k1 * p2.x;
                }
            }

            else
            {
                px[0] = 0.0;
                px[1] = (L1[0] / L1[1]) * p2.x + p2.y;
            }

            pCheckList_i[num] = (double)vx_from;
            pCheckList_t[num] = (double)sqrt(px[0] * px[0] + px[1] * px[1]) / segLen;
            pCheckList_isGap[num] = (double)(p3.shift != 0);

            num++;
            vx_from += (p3.shift + 1);

        }

        // заполнение кортежа выходными данными (num)
        PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(num));
    }

    void check_if_obb_intersection(PyObject* args)
    {
        /*
            args[0] - id
            obb_L2G: np.ndarray (3x3),
            obb_half_length,
            obb_half_width,
            vx_from : long,
            vx_to : long
            args[6] - returns_tuple (ссылка на кортеж "возвращаемых" значений)
        */

        if (PyTuple_GET_SIZE(args) != 7) {  //кол-во параметров, не забываем менять
            cerr << "There should be 7 parameters in the args tuple" << endl;
            return;
        }

        PyArrayObject* py_obb_L2G = (PyArrayObject*)PyTuple_GetItem(args, 1);
        if (PyArray_SHAPE(py_obb_L2G)[0] != 3) {
            cerr << "The shape of obb_L2G should be 3x3" << endl;
            return;
        }
        double* obb_L2G_0 = (double*)PyArray_DATA(py_obb_L2G);
        double* obb_L2G_1 = obb_L2G_0 + 3;
        double obb_half_length = (double)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
        double obb_half_width = (double)PyFloat_AsDouble(PyTuple_GetItem(args, 3));
        long vx_from = PyLong_AsLong(PyTuple_GetItem(args, 4));
        long vx_to = PyLong_AsLong(PyTuple_GetItem(args, 5));
        PyObject* returns_tuple = PyTuple_GetItem(args, 6);

        vector<double> obb_r1 = { obb_L2G_0[0] * obb_half_length + obb_L2G_0[1] * obb_half_width, obb_L2G_1[0] * obb_half_length + obb_L2G_1[1] * obb_half_width };
        vector<double> obb_r2 = { obb_L2G_0[0] * obb_half_length - obb_L2G_0[1] * obb_half_width, obb_L2G_1[0] * obb_half_length - obb_L2G_1[1] * obb_half_width };

        vector<double> lines_seg_r = { 0.0, 0.0 };

        vector<double> L0 = { 0.0, 0.0 };
        vector<double> L1 = { obb_L2G_0[0], obb_L2G_1[0] };
        vector<double> L2 = { obb_L2G_0[1], obb_L2G_1[1] };

        vector<double> T = { 0.0, 0.0 };

        double abs_T_L, abs_lines_seg_r_L, abs_seg_r_L;
        double abs_obb_r1_L, abs_obb_r2_L;

        while (vx_from < vx_to)
        {
            getPnt_t p1 = getPnt(vx_from + 1);

            if (p1.shift != 0)
            {
                vx_from += (p1.shift + 1);
                continue;
            }

            getPnt_t p0 = getPnt(vx_from);

            lines_seg_r[0] = (p1.x - p0.x) / 2.0;
            lines_seg_r[1] = (p1.y - p0.y) / 2.0;

            T[0] = p0.x + lines_seg_r[0] - obb_L2G_0[2];
            T[1] = p0.y + lines_seg_r[1] - obb_L2G_1[2];

            // L0 along segment
            if (abs(lines_seg_r[0]) > 1e-6)
            {
                L0[0] = lines_seg_r[1] / lines_seg_r[0];
                L0[1] = -1.0;
            }
            else
            {
                L0[0] = -1.0;
                L0[1] = 0.0;
            }

            abs_T_L = abs(inner_product(begin(T), end(T), begin(L0), 0.0));
            abs_obb_r1_L = abs(inner_product(begin(obb_r1), end(obb_r1), begin(L0), 0.0));
            abs_obb_r2_L = abs(inner_product(begin(obb_r2), end(obb_r2), begin(L0), 0.0));

            if (abs_obb_r1_L >= abs_obb_r2_L)
            {
                if (abs_T_L - abs_obb_r1_L > -1e-12)
                {
                    vx_from += 1;
                    continue;
                }
            }

            else if (abs_T_L - abs_obb_r2_L > -1e-12)
            {
                vx_from += 1;
                continue;
            }

            // L1 along length of obb
            abs_T_L = abs(inner_product(begin(T), end(T), begin(L1), 0.0));
            abs_lines_seg_r_L = abs(inner_product(begin(lines_seg_r), end(lines_seg_r), begin(L1), 0.0));
            abs_obb_r1_L = abs(inner_product(begin(obb_r1), end(obb_r1), begin(L1), 0.0));
            abs_obb_r2_L = abs(inner_product(begin(obb_r2), end(obb_r2), begin(L1), 0.0));

            if (abs_obb_r1_L >= abs_obb_r2_L)
            {
                if (abs_T_L - (abs_obb_r1_L + abs_lines_seg_r_L) > -1e-12)
                {
                    vx_from += 1;
                    continue;
                }
            }

            else if (abs_T_L - (abs_obb_r2_L + abs_lines_seg_r_L) > -1e-12)
            {
                vx_from += 1;
                continue;
            }

            // L2 along width of obb
            abs_T_L = abs(inner_product(begin(T), end(T), begin(L2), 0.0));
            abs_lines_seg_r_L = abs(inner_product(begin(lines_seg_r), end(lines_seg_r), begin(L2), 0.0));
            abs_obb_r1_L = abs(inner_product(begin(obb_r1), end(obb_r1), begin(L2), 0.0));
            abs_obb_r2_L = abs(inner_product(begin(obb_r2), end(obb_r2), begin(L2), 0.0));

            if (abs_obb_r1_L >= abs_obb_r2_L)
            {
                if (abs_T_L - (abs_obb_r1_L + abs_lines_seg_r_L) > -1e-12)
                {
                    vx_from += 1;
                    continue;
                }
            }

            else if (abs_T_L - (abs_obb_r2_L + abs_lines_seg_r_L) > -1e-12)
            {
                vx_from += 1;
                continue;
            }

            PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(vx_from));
            return;
        }

        PyTuple_SET_ITEM(returns_tuple, 0, PyLong_FromLong(-1));
        return;
    }

};

//Основной глобальный контейнер устройств (указателей на них)
vector<unique_ptr<Device>> devices; //с unique_ptr будет вывзываться деструктор по завершению отдельного устройства

template<void (Device::* F) (PyObject* o)>
PyObject* pyFun(PyObject*, PyObject* o) {

    if (PyTuple_Check(o)) {
        long id = PyLong_AsLong(PyTuple_GetItem(o, 0));
        if (id >= 0 && id < (long)devices.size()) {

            Device* dev = devices[id].get();

            if (F != nullptr) {
                unique_lock lk(dev->mxCall);
                dev->process = false;   //блочим флаг завершенных расчетов
                dev->foosQueue.push(F);     //добавляем функцию в очередь
                dev->argsQueue.push(o);     //добавляем аргументы в очередь
                lk.unlock();
                dev->cvCall.notify_one();   //уведомляем, проверочная функция в wait запрашивает foosQueue.size()
            }
            else {
                unique_lock lk(dev->mxProcess); //синхронизация, здесь он ждет завершения расчетов по флагу
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

    //После завершения расчетных функций, нужно задуматься и о quit() и об reinit() - для полноты картины, тут и деструкторы понадобятся.
    //Здесь - проверки, проверки и создание объекта, если все ок
    //Возвращаает в питон id

    if (PyTuple_GET_SIZE(o) == 4) {

        PyArrayObject* const pyPolyline_ = (PyArrayObject*)PyTuple_GetItem(o, 0);
        PyArrayObject* const pyCheckList_ = (PyArrayObject*)PyTuple_GetItem(o, 1);

        PyArrayObject* const pyNlines_ = (PyArrayObject*)PyTuple_GetItem(o, 2);

        PyObject* pyParams = PyTuple_GetItem(o, 3);

        if (PyArray_NDIM(pyPolyline_) != 2 &&
            PyArray_NDIM(pyCheckList_) != 2)
        {
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
            devices.push_back(unique_ptr<Device>(new Device(pyPolyline_, pyCheckList_, pyNlines_, pyParams, (long)devices.size())));
            return PyLong_FromLong((long)devices.size() - 1);
        }
        catch (exception const*) {
            return PyLong_FromLong(-1);
        }
    }

    cerr << "Incorrect args number" << endl;
    return PyLong_FromLong(-1);
}

static PyMethodDef polylineCpp_methods[] = {
    { "init", (PyCFunction)init, METH_VARARGS, ""},
    { "closest_pnt", (PyCFunction)pyFun<&Device::closest_pnt>, METH_VARARGS, "Non-blocking!"},
    { "closest_pnt00", (PyCFunction)pyFun<&Device::closest_pnt00>, METH_VARARGS, "Non-blocking!"},
    { "check_segment_intersections", (PyCFunction)pyFun<&Device::check_segment_intersections>, METH_VARARGS, "Non-blocking!"},
    { "check_segment_intersections00", (PyCFunction)pyFun<&Device::check_segment_intersections00>, METH_VARARGS, "Non-blocking!"},
    { "check_if_obb_intersection", (PyCFunction)pyFun<&Device::check_if_obb_intersection>, METH_VARARGS, "Non-blocking!"},
    { "synchronize", (PyCFunction)pyFun<nullptr>, METH_VARARGS, "Waits for all the previous calls to return or returns immediately"},
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef polylineCpp_module = {
    PyModuleDef_HEAD_INIT,
    "polylineCpp",                        // Module name to use with Python import statements
    "Polyline activity",                    // Module description
    0,
    polylineCpp_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_polylineCpp() {
    return PyModule_Create(&polylineCpp_module);
}