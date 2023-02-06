#pragma once
#include <vector>


class Line {
public:
	double* line; // парметры А и С
	bool isGap;
	bool isSingle;
	double q0k; // порог ошибки
	double _q; // ошибка
	double* _sums; // вычисляемые суммы

	Line(double q0k=0.005); // конструктор
	
	static double* LMS(std::vector<double> X, std::vector<double> Y, size_t fr, size_t to, double* sums); // МНК-аппроксимация прямой, возвращает A, C, q 
	Line* copy(); // возвращает копию объекта Line
	void setAsTangentWithOnePnt(double* p); // задаёт A, C по одной точке
	void setWithTwoPnts(double* p1, double* p2); // задаёт A, C по двум точкам
	size_t* setWithLMS(std::vector<double>* pnts, bool best = true); // возвращает смещения среза от краёв ?
	double getDistanceToPnt(double* p, bool sgnd=false); // возвращает расстояние от точки до прямой
	void getProjectionOfPnt(double* p, double** pout); // ???
	void getProjectionOfPntEx(double* p, double** pout, double half_dPhi, bool direction); // ???
	void getIntersection(Line* line, double** pout); // вычисляет координаты точки пересечения двух прямых
	void lol();

};

