#include "line.h"

#include <iostream>
//#include <limits>

#include <cmath>
#include <vector>
#include <numeric>

using namespace std;

Line::Line(double qOk) :
	line(new double[2]{ 0.0, 0.0 }),
	isGap(true),
	isSingle(true),
	qOk(qOk),
	_q(qOk),
	_sums(new double[5]{ 0.0, 0.0, 0.0, 0.0, 0.0 })
{}

double* Line::LMS(vector<double> X, vector<double> Y, long fr, long to, double* sums)
{
	/*
	sums0 = sum(X)
	sums1 = sum(Y)
	sums2 = sums(X^2)
	sums3 = sums(Y^2)
	sums4 = sums(X * Y)
	*/

	long N = to - fr;
	double phi = sums[4] - (sums[0] * sums[1]) / N;

	double distsSum1 = 0.0;
	double distsSum2 = 0.0;
	double A1, A2, C1, C2;

	if (abs(phi) > 1e-6)
	{
		double theta = (sums[3] - sums[2]) / phi + (sums[0] * sums[0] - sums[1] * sums[1]) / (phi * N);
		double D = theta * theta + 4.0;

		A1 = (theta + sqrt(D)) / 2.0;
		A2 = (theta - sqrt(D)) / 2.0;

		C1 = (sums[1] - sums[0] * A1) / N;
		C2 = (sums[1] - sums[0] * A2) / N;

		for (long i = fr; i < to; i++)
		{
			distsSum1 += abs(X[i] * A1 - Y[i] + C1) / sqrt(A1 * A1 + 1.0);
			distsSum2 += abs(X[i] * A2 - Y[i] + C2) / sqrt(A2 * A2 + 1.0);
		}

	}

	else
	{
		A1 = INFINITY;
		A2 = 0.0;

		C1 = sums[0] / N;
		C2 = sums[1] / N;

		for (long i = fr; i < to; i++)
		{
			distsSum1 += abs(X[i] - C1);
			distsSum2 += abs(X[i] - C2);
		}
	}

	if (distsSum1 < distsSum2)
		return new double[3]{ A1, C1, distsSum1 / N };
	else
		return new double[3]{ A2, C2, distsSum2 / N };
}

Line* Line::copy()
{
	Line* cp_line = new Line();
	for (long i = 0; i < 2; i++)
		cp_line->line[i] = this->line[i];

	cp_line->line[0] = this->line[0];
	cp_line->line[1] = this->line[1];
	cp_line->isGap = this->isGap;
	cp_line->isSingle = this->isSingle;
	cp_line->qOk = this->qOk;
	cp_line->_q = this->_q;

	/*for (long i = 0; i < 5; i++)
		cp_line->_sums[i] = this->_sums[i];*/

	return cp_line;
}

void Line::setAsTangentWithOnePnt(double* p)
{
	// p[0] = x, p[1] = y

	if (abs(p[1]) > 1e-6)
	{
		this->line[0] = -p[0] / p[1];
		this->line[1] = (p[0] * p[0] + p[1] * p[1]) / p[1];
	}

	else
	{
		this->line[0] = INFINITY;
		this->line[1] = p[0];
	}
	this->isSingle = true;
}

void Line::setWithTwoPnts(double* p1, double* p2)
{
	double dx = p2[0] - p1[0];
	if (abs(dx) > 1e-6)
	{
		this->line[0] = (p2[1] - p1[1]) / dx;
		this->line[1] = p1[1] - this->line[0] * p1[0];
	}

	else
	{
		this->line[0] = INFINITY;
		this->line[1] = p1[0]; // без минуса удобнее
	}
	this->isSingle = false;
}

long* Line::setWithLMS(vector<double>* pnts, bool best)
{
	vector<double> X = pnts[0];
	vector<double> Y = pnts[1];
	long segN = (long)X.size();

	// sums calculation
	this->_sums[0] = accumulate(begin(X), end(X), 0.0);
	this->_sums[1] = accumulate(begin(Y), end(Y), 0.0);
	this->_sums[2] = inner_product(begin(X), end(X), begin(X), 0.0);
	this->_sums[3] = inner_product(begin(Y), end(Y), begin(Y), 0.0);
	this->_sums[4] = inner_product(begin(X), end(X), begin(Y), 0.0);

	double* resLMS = new double[3];

	resLMS = Line::LMS(X, Y, 0, segN, this->_sums); // returns A, C, q
	this->line[0] = resLMS[0];
	this->line[1] = resLMS[1];
	this->_q = resLMS[2];

	this->isSingle = false;

	if (best)
	{
		long beg = 0;
		long end = segN;

		double A, C, q;

		bool direction = true;
		while (this->_q >= this->qOk)
		{
			if (direction)
			{
				end--;
				this->_sums[0] -= X[end];
				this->_sums[1] -= Y[end];
				this->_sums[2] -= X[end] * X[end];
				this->_sums[3] -= Y[end] * Y[end];
				this->_sums[4] -= X[end] * Y[end];
			}

			else
			{
				this->_sums[0] -= X[beg];
				this->_sums[1] -= Y[beg];
				this->_sums[2] -= X[beg] * X[beg];
				this->_sums[3] -= Y[beg] * Y[beg];
				this->_sums[4] -= X[beg] * Y[beg];
				beg++;
			}

			resLMS = Line::LMS(X, Y, beg, end, this->_sums); // returns A, C, q
			A = resLMS[0];
			C = resLMS[1];
			q = resLMS[2];

			if (q > this->_q)
			{
				if (direction)
				{
					end++;
					direction = false;
				}

				else
				{
					beg--;
					break;
				}
			}

			else
			{
				this->line[0] = A;
				this->line[1] = C;
				this->_q = q;
			}

		}

		return new long[2]{ (long)beg, (long)(end - segN) };
	}

	else
		return new long[2]{ 0, 0 };
}

double Line::getDistanceToPnt(double* p, bool sgnd)
{
	if (sgnd)
	{
		if (!isinf(this->line[0]))
			return (this->line[0] * p[0] - p[1] + this->line[1]) / sqrt(this->line[0] * this->line[0] + 1.0);
		else
			return (p[0] - this->line[1]);
	}

	else
	{
		if (!isinf(this->line[0]))
			return abs(this->line[0] * p[0] - p[1] + this->line[1]) / sqrt(this->line[0] * this->line[0] + 1.0);
		else
			return abs(p[0] - this->line[1]);
	}
}

void Line::getProjectionOfPnt(double* p, double** pout)
{
	if (!isinf(this->line[0]))
	{
		*pout[0] = (p[0] + this->line[0] * p[1] - this->line[0] * this->line[1]) / (this->line[0] * this->line[0] + 1.0);
		*pout[1] = this->line[0] * *pout[0] + this->line[1];
	}
	else
	{
		*pout[0] = this->line[1];
		*pout[1] = p[1];
	}
}

void Line::getProjectionOfPntEx(double* p, double** pout, double half_dPhi, bool direction)
{
	if (!this->isSingle)
	{
		if (!isinf(this->line[0]))
		{
			*pout[0] = (p[0] + this->line[0] * p[1] - this->line[0] * this->line[1]) / (this->line[0] * this->line[0] + 1.0);
			*pout[1] = this->line[0] * *pout[0] + this->line[1];
		}

		else
		{
			*pout[0] = this->line[1];
			*pout[1] = p[1];
		}
	}

	else
	{
		double l = sqrt(p[0] * p[0] + p[1] * p[1]) * tan(half_dPhi); // находим точку на линии, отстоящую вправо от точки касания на угол half_dphi
		double alpha = atan2(p[1], p[0]);  // + pi / 2.0
		if (direction)
		{
			*pout[0] = p[0] - l * sin(alpha); // cos(x) = sin(x + pi / 2), -sin(x) = cos(x + pi / 2)
			*pout[1] = p[1] + l * cos(alpha);
		}

		else
		{
			*pout[0] = p[0] + l * sin(alpha);
			*pout[1] = p[1] - l * cos(alpha);
		}
	}
}

void Line::getIntersection(Line* line, double** pout)
{
	if (abs(this->line[0] - line->line[0]) < 1e-6)
	{
		*pout[0] = INFINITY; //прямые параллельны, в т.ч.и если обе вертикальные
		*pout[1] = INFINITY;
	}

	else
	{
		if (isinf(this->line[0]))
		{
			*pout[0] = this->line[1]; // вертикальная исходная
			*pout[1] = line->line[0] * this->line[1] + line->line[1];
		}

		else if (isinf(line->line[0]))
		{
			*pout[0] = line->line[1]; // вертикальная подставляемая
			*pout[1] = this->line[0] * line->line[1] + this->line[1];
		}

		else
		{
			*pout[0] = (line->line[1] - this->line[1]) / (this->line[0] - line->line[0]);
			*pout[1] = this->line[0] * *pout[0] + this->line[1];
		}
	}
}

void Line::lol()
{
	cout << this->line[0] << " " << this->line[1] << endl;
	this->line[0] = 500;
	cout << this->line[0] << " " << this->line[1] << endl;
}


