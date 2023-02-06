#pragma once
#include <vector>


class Line {
public:
	double* line; // �������� � � �
	bool isGap;
	bool isSingle;
	double q0k; // ����� ������
	double _q; // ������
	double* _sums; // ����������� �����

	Line(double q0k=0.005); // �����������
	
	static double* LMS(std::vector<double> X, std::vector<double> Y, size_t fr, size_t to, double* sums); // ���-������������� ������, ���������� A, C, q 
	Line* copy(); // ���������� ����� ������� Line
	void setAsTangentWithOnePnt(double* p); // ����� A, C �� ����� �����
	void setWithTwoPnts(double* p1, double* p2); // ����� A, C �� ���� ������
	size_t* setWithLMS(std::vector<double>* pnts, bool best = true); // ���������� �������� ����� �� ���� ?
	double getDistanceToPnt(double* p, bool sgnd=false); // ���������� ���������� �� ����� �� ������
	void getProjectionOfPnt(double* p, double** pout); // ???
	void getProjectionOfPntEx(double* p, double** pout, double half_dPhi, bool direction); // ???
	void getIntersection(Line* line, double** pout); // ��������� ���������� ����� ����������� ���� ������
	void lol();

};

