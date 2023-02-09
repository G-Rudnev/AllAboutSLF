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

#include "Line.h"

using namespace std;

struct Device {

    size_t id = -1;

    thread thr; //поток, который будет отвечать за обработку объекта

    mutex mxCall;   //мьютекс по вызовам и добавлению функций в очередь
    mutex mxProcess;    //мьютекс по контролю завершения расчетов

    condition_variable cvCall; //соответствующие cv
    condition_variable cvProcess;

    bool process = true;    //флаг завершения расчетов

    bool alive = true;  //флаг жизни потока thr

    queue<void (Device::*) ()> foosQueue;   //очередь функций объекта

    size_t Npnts = 0;  //in variable

    //size_t* Nlines = 0; //out variable

    //double par0 = 0.0;  //параметры
    double deep = 0.0;
    double continuity = 0.0;
    double half_dPhi = 0.0;
    double tolerance = 0.0;

//  ...

    PyArrayObject* pyPntsXY = nullptr;  //указатели на нужные объекты питона
    PyArrayObject* pyPntsPhi = nullptr;
    PyArrayObject* pyLinesXY = nullptr;
    PyArrayObject* pyNlines = nullptr;

    //double** pntsXY = nullptr;  //соответствующие массивы плюсовой памяти
    //double* pntsPhi = nullptr;
    //double** linesXY = nullptr;


    Device() = default;

    Device (PyArrayObject* const pyPntsXY, PyArrayObject* const pyPntsPhi, PyArrayObject* const pyLinesXY, PyArrayObject* const pyNlines, size_t Npnts, PyObject* pyParams, size_t id) : 
        pyPntsXY(pyPntsXY), pyPntsPhi(pyPntsPhi), pyLinesXY(pyLinesXY), pyNlines(pyNlines), Npnts(Npnts), id(id),
        deep(5.0), continuity(0.6), half_dPhi(0.3 * 0.0174532925199432957692369), tolerance(0.1)
    {

        if (!setParams(pyParams)) {
            throw new logic_error("");  //память еще не выделилась, можно не заморачиваться с чисткой
        }

        //setParams(pyParams);

        //pntsXY = new double* [2];
        //linesXY = new double* [2];
        //for (int i = 0; i < 2; i++) {
        //    pntsXY[i] = new double[Npnts];
        //    //linesXY[i] = new double[Npnts]; ////////////////////////////////////////////////////////////////////
        //}


        //pntsPhi = new double[Npnts];

        thr = thread([this]     //запускаем поток
            {
                while (this->alive) {

                    for (;;) {

                        unique_lock lk0(this->mxCall);
                        this->cvCall.wait(lk0, [this] {return !this->alive || this->foosQueue.size(); });   //ждун вызова через pyFun
                        
                        if (this->alive) {

                            auto foo = this->foosQueue.front(); //берем функцию с края очереди

                            if (foo != nullptr)
                                (this->*foo) ();    //вызываем её

                            this->foosQueue.pop();  //удаляем из очереди

                            if (this->foosQueue.empty()) {
                                unique_lock lk1(this->mxProcess);   
                                this->process = true;   //меняем состояние для cv процесса, функция pyFun<nullptr>, т.е. synchronize() из питона, сможет завершиться
                                lk1.unlock();
                                cvProcess.notify_one(); //уведомляем cv о завершении расчетов

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
        //НЕ ОСОБО НУЖНОЕ УДОВОЛЬСТВИЕ, НО ЕСТЬ
        //Питона крашит модули весьма кусочно и не консистентно, поэтому может даже подвесить какую-нибудь работающую функцию.
        //Нам не сильно это важно, так как предполагается существование объекта все время работы программы, ну, или почти все.
        //Тем не менее, если разрабатывать quit() этот деструктор пригодится.

        cout << "Device " << id << " destructor starts" << endl;

        unique_lock lk1(mxCall);    //ждет завершения вызова pyFun

        unique_lock lk0(mxProcess);
        cvProcess.wait(lk0, [this] {return this->process; }); //ждет завершения расчетной очереди функций
        lk0.unlock();

        alive = false;  //выключает петлю потока
        lk1.unlock();
        cvCall.notify_all();

        thr.join();

        cout << "Device " << id << " destructor ends normally" << endl;
    }

    bool setParams(PyObject* pyParams) {

        //Можно и getParams() сделать, но пока это необязательно

        if (!PyTuple_Check(pyParams)) {
            cerr << "Parameters should be in a tuple" << endl;
            return false;
        }

        if (PyTuple_GET_SIZE(pyParams) != 4) {  //кол-во параметров, не забываем менять
            cerr << "There should be 4 parameters in a tuple" << endl; // 5-й - ссылка на единичный список Nlines
            return false;
        }

        //par0 = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 0));  //здесь столько параметров, колько нужно надо прописать
        deep = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 0));
        continuity = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 1));
        half_dPhi = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 2)) * 0.0174532925199432957692369;
        tolerance = PyFloat_AsDouble(PyTuple_GetItem(pyParams, 3));


        if (PyErr_Occurred()) {
            cerr << "Wrong type inside the parameters" << endl;
            PyErr_Clear();
            return false;
        }

        return true;
    }

    //inline void pushPnts() {

    //    //СОВЕРШЕННО НЕ ОБЯЗАТЕЛЬНО ПОЛЬЗОВАТЬ PUSH ФУКНЦИЮ И РАСЧЕТНУЮ ОТДЕЛЬНО, МОЖНО ИСХИТРИТЬСЯ И НА ОДНУ ОБЩУЮ,
    //    //УКАЗАТЕЛИ НА САМИ ДАННЫЕ (pXY_0r, ..) МОЖНО И В ОБЪЕКТ ДОБАВИТЬ, ОБЕСПЕЧИВАЯ В ДАЛЬНЕЙШЕМ НЕИЗМЕННОСТЬ АЛЛОКАЦИИ В ПИТОНЕ.
    //    //БЕЗ КОПИРОВАНИЯ ЗДЕСЬ В ПУШЕ, А С РАСЧЕТАМИ НАПРЯМУЮ БУДЕТ МАКСИМАЛЬНО БЫСТРО - НУЖНО СМОТРЕТЬ ПО РЕАЛИЗАЦИИ КАК ВЫХОДИТ УДОБНЕЕ.

    //    //cout << "pushPnts() on id " << id << endl;

    //    double* pXY_0r = (double*)PyArray_DATA(pyPntsXY);   //0-й ряд
    //    double* pXY_1r = pXY_0r + Npnts;                    //1-й ряд
    //    double* pPhi = (double*)PyArray_DATA(pyPntsPhi);

    //    for (size_t i = 0; i < Npnts; i++) {
    //        pntsXY[0][i] = *(pXY_0r + i);
    //        pntsXY[1][i] = *(pXY_1r + i);
    //        pntsPhi[i] = *(pPhi + i);
    //    }

    //    //С ежекратным запросом (медленнее)
    //    //for (size_t i = 0; i < Npnts; i++) {
    //    //    pntsXY[0][i] = *(double*)PyArray_GETPTR2(pyPntsXY, 0, i);
    //    //    pntsXY[1][i] = *(double*)PyArray_GETPTR2(pyPntsXY, 1, i);
    //    //    pntsPhi[i] = *(double*)PyArray_GETPTR1(pyPntsPhi, i);
    //    //}
    //}

    void calcLines() {
        //Нужно еще выбрасывать Nlines, можно через связанный pyParams.
        //Расчетов мало на прототипе - выигрыш по времени ещё не такой большой.
 
        //cout << "pullLinesXY() on id " << id << endl;

        // Это я просто скопировал из pushPnts
        // Получение точек из Python сюда
        double* pXY_x = (double*)PyArray_DATA(pyPntsXY);   //0-й ряд 
        double* pXY_y = pXY_x + Npnts;                    //1-й ряд
        double* pPhi = (double*)PyArray_DATA(pyPntsPhi);

        size_t* pNlines = (size_t*)PyArray_DATA(pyNlines);

        /*for (size_t i = 0; i < Npnts; i++)  //////////////////////////////////////////////////////////////////////////////////////
        {
            pntsXY[0][i] = *(pXY_x + i);
            pntsXY[1][i] = *(pXY_y + i);
            pntsPhi[i] = *(pPhi + i);
        }*/

        double* pLines_0r = (double*)PyArray_DATA(pyLinesXY);
        double* pLines_1r = pLines_0r + Npnts;

        size_t fr = 0;
        size_t to = 0;
        size_t ex_fr = 0;
        size_t ex_to = 0;

        Line* line = new Line();
        Line* prev_line = new Line();
        Line* ex_line = new Line();

        bool extra = false;

        double pEx_pnt_x = 0.0; // = new double* [2]; //{0.0, 0.0};
        double pEx_pnt_y = 0.0;

        *(pNlines) = 0;

        long* edges; // для результата setWithLMS - смещения от краёв отрезка
        vector<double>* segPntsXY; // точки сформированного отрезка

        while (fr < Npnts) 
        {
            // формирование сегмента
            if (!extra)
            {
                if ((pXY_x[fr] == 0.0) && (pXY_y[fr] == 0.0))
                {
                    line->isGap = true;
                    to = fr + 1;
                    while ((to < Npnts) && ((pXY_x[to] == 0.0) && pXY_y[to] == 0.0))
                        to++;
                }
                
                else
                {
                    line->isGap = false;
                    line->setAsTangentWithOnePnt(new double[] { pXY_x[fr], pXY_y[fr] });
                    to = fr + 1;

                    if ((to < Npnts) && (pXY_x[to] != 0.0 || pXY_y[to] != 0.0) && (sqrt(pow(pXY_x[to] - pXY_x[fr], 2) + pow(pXY_y[to] - pXY_y[fr], 2)) <= continuity))
                    {
                        line->setWithTwoPnts(new double[] { pXY_x[fr], pXY_y[fr] }, new double[] { pXY_x[to], pXY_y[to] });
                        to++;
                        while ((to < Npnts) && (pXY_x[to] != 0.0 || pXY_y[to] != 0) && (line->getDistanceToPnt(new double[] { pXY_x[to], pXY_y[to] }) <= tolerance))
                        {
                            to++;
                            if ((to - fr) % 2 == 0)
                            {
                                size_t mid_to = fr + (size_t)(floor((to - fr) / 2.0));
                                line->setWithTwoPnts(new double[] { pXY_x[fr], pXY_y[fr] }, new double[] { pXY_x[mid_to], pXY_y[mid_to] });
                            }
                        }
                    }

                    if (to - fr > 2)
                    {
                        segPntsXY = new vector<double>[2];
                        for (int i = 0; i < 2; i++)
                            segPntsXY[i] = vector<double> (to - fr);

                        for (size_t i = 0; i < to - fr; i++)
                        {
                            segPntsXY[0][i] = pXY_x[fr + i];
                            segPntsXY[1][i] = pXY_y[fr + i];
                        }

                        edges = line->setWithLMS(segPntsXY);

                        if (edges[0] != 0)
                        {
                            ex_line = line->copy();
                            ex_fr = fr + edges[0];
                            ex_to = to + edges[1];
                            extra = true;
                            if (edges[0] == 1)
                            {
                                line->line[0] = pXY_y[fr] / pXY_x[fr];
                                line->line[1] = 0.0;
                                fr--;
                                to = fr + 1;
                            }

                            else if (edges[0] == 2)
                            {
                                line->setWithTwoPnts(new double[] { pXY_x[fr], pXY_y[fr] }, new double[] { pXY_x[fr + 1], pXY_y[fr + 1] });
                                to = fr + 1;
                            }

                            else
                            {
                                segPntsXY = new vector<double>[2];
                                for (int i = 0; i < 2; i++)
                                    segPntsXY[i] = vector<double> (edges[0]);

                                for (size_t i = 0; i < edges[0]; i++)
                                {
                                    segPntsXY[0][i] = pXY_x[fr + i];
                                    segPntsXY[1][i] = pXY_y[fr + i];
                                }

                                edges = line->setWithLMS(segPntsXY, false);
                                to = fr + edges[0] - 1;
                            }
                        }

                        else
                            to += edges[1];
                    }
                }
            }

            else
            {
                line = ex_line->copy();
                fr = ex_fr;
                to = ex_to;
                extra = false;
            }

            // интеграция сегмента
            if (!(line->isGap))
            {
                if (!prev_line->isGap)
                {
                    prev_line->getIntersection(line, new double* [] { pLines_0r + (*pNlines), pLines_1r + (*pNlines) });
                    double interAngle = atan2(pLines_1r[(*pNlines)], pLines_0r[(*pNlines)]);
                    if (((interAngle > pPhi[fr - 1]) || (interAngle < pPhi[fr])) && ((interAngle > pPhi[fr]) || (interAngle < pPhi[fr - 1])))
                    {
                        prev_line->getProjectionOfPntEx(new double[] { pXY_x[fr - 1], pXY_y[fr - 1] }, new double*[] { pLines_0r + (*pNlines), pLines_1r + (*pNlines) }, half_dPhi, false);
                        (*pNlines) += 1;
                        if (sqrt(pow(pXY_x[fr] - pXY_x[fr - 1], 2) + pow(pXY_y[fr] - pXY_y[fr - 1], 2)) > continuity)
                        {
                            *(pLines_0r + (*pNlines)) = 0.001;
                            *(pLines_1r + (*pNlines)) = 0.001;
                            (*pNlines) += 1;
                        }

                        prev_line->isGap = true;
                    }

                    else
                        (*pNlines) += 1;
                }

                if (prev_line->isGap)
                {
                    line->getProjectionOfPntEx(new double[] { pXY_x[fr], pXY_y[fr] }, new double* [] { pLines_0r + (*pNlines), pLines_1r + (*pNlines) }, half_dPhi, true);
                    (*pNlines) += 1;
                }

                if (to >= Npnts)
                {
                    line->getProjectionOfPntEx(new double[] { pXY_x[to - 1], pXY_y[to - 1] }, new double* [] { pLines_0r + (*pNlines), pLines_1r + (*pNlines) }, half_dPhi, false);
                    (*pNlines) += 1;
                }
            }

            else // сюда на моих данных не заходит - пока не получилось протестировать, возможно, имеет ошибки
            {
                if (fr == 0)
                {
                    if (to < Npnts)
                    {
                        ex_line->line[0] = tan(pPhi[0]);
                        ex_line->line[1] = 0.0;
                        ex_line->getProjectionOfPnt(new double[] { pXY_x[to], pXY_y[to] }, new double* [] { pLines_0r, pLines_1r});
                        *(pLines_0r + 1) = 0.0;
                        *(pLines_1r + 1) = 0.0;
                        (*pNlines) = 2;
                    }

                    else
                    {
                        *pLines_0r = deep * cos(pPhi[0]);
                        *pLines_1r = deep * sin(pPhi[0]);
                        *(pLines_0r + 1) = 0.0;
                        *(pLines_1r + 1) = 0.0;
                        *(pLines_0r + 2) = deep * cos(pPhi[Npnts - 1]);
                        *(pLines_1r + 2) = deep * sin(pPhi[Npnts - 1]);
                        (*pNlines) = 3;
                    }
                }

                else
                {
                    prev_line->getProjectionOfPntEx(new double[] { pXY_x[fr - 1], pXY_y[fr - 1] }, new double* [] { pLines_0r + (*pNlines), pLines_1r + (*pNlines) }, half_dPhi, false);
                    (*pNlines)++;

                    if (to >= Npnts)
                    {
                        *(pLines_0r + (*pNlines)) = 0.0;
                        *(pLines_1r + (*pNlines)) = 0.0;
                        (*pNlines) += 1;

                        ex_line->line[0] = tan(pPhi[Npnts - 1]);
                        ex_line->line[1] = 0.0;
                        ex_line->getProjectionOfPnt(new double[] { pXY_x[fr - 1], pXY_y[fr - 1] }, new double* [] { pLines_0r + (*pNlines), pLines_1r + (*pNlines)});
                        (*pNlines) += 1;
                    }

                    else if (sqrt(pow(pXY_x[to] - pXY_x[fr - 1], 2) + pow(pXY_y[to] - pXY_y[fr - 1], 2)) > continuity)
                    {
                        ex_line->line[0] = tan(pPhi[fr - 1]);
                        ex_line->line[1] = 0.0;
                        ex_line->getProjectionOfPnt(new double[] { pXY_x[to], pXY_y[to] }, new double* [] {&pEx_pnt_x, &pEx_pnt_y});//new double* [] { ex_pnt[0], ex_pnt[1]});
                        
                        if ((sqrt(pow(pEx_pnt_x, 2) + pow(pEx_pnt_y, 2)) > sqrt(pow(pXY_x[fr - 1], 2) + pow(pXY_y[fr - 1], 2))) && (pEx_pnt_x * pXY_x[fr - 1] > 0.0 or pEx_pnt_y * pXY_y[fr - 1] > 0.0))
                        {
                            *(pLines_0r + (*pNlines)) = 0.001;
                            *(pLines_1r + (*pNlines)) = 0.001;
                            (*pNlines) += 1;

                            *(pLines_0r + (*pNlines)) = pEx_pnt_x;
                            *(pLines_1r + (*pNlines)) = pEx_pnt_y;
                            (*pNlines) += 1;

                            *(pLines_0r + (*pNlines)) = 0.0;
                            *(pLines_1r + (*pNlines)) = 0.0;
                            (*pNlines) += 1;
                        }

                        else
                        {
                            ex_line->line[0] = tan(pPhi[to]);
                            ex_line->line[1] = 0.0;
                            ex_line->getProjectionOfPnt(new double[] { pXY_x[fr - 1], pXY_y[fr - 1] }, new double* [] { &pEx_pnt_x, &pEx_pnt_y });

                            if ((sqrt(pow(pEx_pnt_x, 2) + pow(pEx_pnt_y, 2)) > sqrt(pow(pXY_x[to], 2) + pow(pXY_y[to], 2))) && (pEx_pnt_x * pXY_x[to] > 0.0 or pEx_pnt_y * pXY_y[to] > 0.0))
                            {
                                *(pLines_0r + (*pNlines)) = 0.0;
                                *(pLines_1r + (*pNlines)) = 0.0;
                                (*pNlines) += 1;

                                *(pLines_0r + (*pNlines)) = pEx_pnt_x;
                                *(pLines_1r + (*pNlines)) = pEx_pnt_y;
                                (*pNlines) += 1;

                                *(pLines_0r + (*pNlines)) = 0.001;
                                *(pLines_1r + (*pNlines)) = 0.001;
                                (*pNlines) += 1;
                            }

                            else
                            {
                                *(pLines_0r + (*pNlines)) = 0.0;
                                *(pLines_1r + (*pNlines)) = 0.0;
                                (*pNlines) += 1;
                            }
                        }
                    }
                }
            }

            prev_line = line->copy();
            fr = to;
        }

        //с ежекратным запросом (медленнее)
        //for (size_t i = 0; i < Nlines; i++) {
        //    *(double*)PyArray_GETPTR2(pyLinesXY, 0, i) = pntsXY[0][i] * pntsXY[0][i] * pntsPhi[i];
        //    *(double*)PyArray_GETPTR2(pyLinesXY, 1, i) = pntsXY[1][i] * pntsXY[1][i] * pntsPhi[i];
        //}
    }

};


//Основной глобальный контейнер устройств (указателей на них)
vector<unique_ptr<Device>> devices; //с unique_ptr будет вывзываться деструктор по завершению отдельного устройства

template<void (Device::* F) ()>
PyObject* pyFun(PyObject*, PyObject* o) {

    if (PyLong_Check(o)) {
        int id = (int)PyLong_AsLong(o);
        if (id >= 0 && id < devices.size()) {

            Device* dev = devices[id].get();

            if (F != nullptr) {
                unique_lock lk(dev->mxCall);
                dev->process = false;   //блочим флаг завершенных расчетов
                dev->foosQueue.push(F);     //добавляем функцию в очередь
                lk.unlock();
                dev->cvCall.notify_one();   //уведомляем, проверочная функция в wait запрашивает size() jxthtlb
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

    if (PyTuple_GET_SIZE(o) == 6) {

        PyArrayObject* const pyPntsXY_ = (PyArrayObject*)PyTuple_GetItem(o, 0);
        PyArrayObject* const pyPntsPhi_ = (PyArrayObject*)PyTuple_GetItem(o, 1);
        PyArrayObject* const pyLinesXY_ = (PyArrayObject*)PyTuple_GetItem(o, 2);
        PyArrayObject* const pyNlines_ = (PyArrayObject*)PyTuple_GetItem(o, 3);


        if (!PyLong_Check(PyTuple_GetItem(o, 4))) {
            cerr << "Bad N of points" << endl;
            return PyLong_FromLong(-1);
        }

        size_t N = PyLong_AsLongLong(PyTuple_GetItem(o, 4));
        PyObject* pyParams = PyTuple_GetItem(o, 5);

        if (PyArray_NDIM(pyPntsXY_) != 2 &&
            PyArray_NDIM(pyPntsPhi_) != 1 &&
            PyArray_NDIM(pyLinesXY_) != 2) //&& PyArray_NDIM(pyNlines_) != 1) 
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
            devices.push_back(unique_ptr<Device>(new Device(pyPntsXY_, pyPntsPhi_, pyLinesXY_, pyNlines_, N, pyParams, devices.size())));
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
    //{ "pushPnts", (PyCFunction)pyFun<&Device::pushPnts>, METH_O, ""},
    { "calcLines", (PyCFunction)pyFun<&Device::calcLines>, METH_O, ""},
    //{ "getNlines", (PyCFunction)pyFun<&Device::getNlines>, METH_O, "" },
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