#include "Line.h"

#include <iostream>
//#include <limits>

#include <cmath>
#include <vector>
#include <numeric>

using namespace std;

Line::Line(double q0k):
	line(new double[] {0.0, 0.0}),
	isGap(true),
	isSingle(true),
	q0k(q0k),
	_q(q0k),
	_sums(new double[] {0.0, 0.0, 0.0, 0.0, 0.0}) 
{}


double* Line::LMS(vector<double> X, vector<double> Y, size_t fr, size_t to, double* sums) 
{
	/*
	sums0 = sum(X)
	sums1 = sum(Y)
	sums2 = sums(X^2)
	sums3 = sums(Y^2)
	sums4 = sums(X * Y)
	*/

	size_t N = to - fr;
	double phi = sums[4] - (sums[0] * sums[1]) / N;

	if (abs(phi) > 0.000001) 
	{
		double theta = (sums[3] - sums[2]) / phi + (pow(sums[0], 2.0) - pow(sums[1], 2.0)) / (phi * N);
		double D = pow(theta, 2.0) + 4.0;
		
		double A1 = (theta + sqrt(D)) / 2.0;
		double A2 = (theta - sqrt(D)) / 2.0;
		
		double C1 = (sums[1] - sums[0] * A1) / N;
		double C2 = (sums[1] - sums[0] * A2) / N;

		double distsSum1 = 0.0;
		double distsSum2 = 0.0;

		for (size_t i = fr; i < to; i++) 
		{
			distsSum1 += abs(X[i] * A1 - Y[i] + C1) / sqrt(pow(A1, 2.0) + 1.0);
			distsSum2 += abs(X[i] * A2 - Y[i] + C2) / sqrt(pow(A2, 2.0) + 1.0);
		}

		if (distsSum1 < distsSum2)
			return new double[3]{ A1, C1, distsSum1 / N };
		else
			return new double[3]{ A2, C2, distsSum2 / N };

	}

	/*size_t N = (size_t) to - (size_t) fr;
	double* p1 = new double[] {X[0], Y[0]};
	double* p2 = new double[] {X[N - 1], Y[N - 1]};
	this->setWithTwoPnts(p1, p2);

	return new double[3] {this->line[0], this->line[1], 0.0};*/

	return new double[3];
}


Line* Line::copy() 
{
	Line* cp_line = new Line();
	for (int i = 0; i < 2; i++) 
		cp_line->line[i] = this->line[i];

	cp_line->line[0] = this->line[0];
	cp_line->line[1] = this->line[1];
	cp_line->isGap = this->isGap;
	cp_line->isSingle = this->isSingle;
	cp_line->q0k = this->q0k;
	cp_line->_q = this->_q;
	
	for (int i = 0; i < 5; i++) 
		cp_line->_sums[i] = this->_sums[i];

	return cp_line;
}


void Line::setAsTangentWithOnePnt(double* p) 
{
	// p[0] = x, p[1] = y

	if (p[1] != 0.0) 
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
	if (dx != 0.0) 
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


//size_t* Line::setWithLMS(double** pnts, bool best) {
//	
//	//size_t N =  sizeof(*pnts) / sizeof(**pnts);
//	// cout << sizeof(**pnts) << endl;
//	size_t N = 0;
//	return new size_t[] {0, N};
//}


size_t* Line::setWithLMS(vector<double>* pnts, bool best) 
{
	vector<double> X = pnts[0];
	vector<double> Y = pnts[1];
	size_t N = X.size();

	// sums calculation
	this->_sums[0] = accumulate(begin(X), end(X), 0.0);
	this->_sums[1] = accumulate(begin(Y), end(Y), 0.0);
	this->_sums[2] = inner_product(begin(X), end(X), begin(X), 0.0);
	this->_sums[3] = inner_product(begin(Y), end(Y), begin(Y), 0.0);
	this->_sums[4] = inner_product(begin(X), end(X), begin(Y), 0.0);

	double* resLMS = new double[3];

	resLMS = Line::LMS(X, Y, 0, N, this->_sums); // returns A, C, q
	this->line[0] = resLMS[0];
	this->line[1] = resLMS[1];
	this->_q = resLMS[2];
	this->isSingle = false;

	if (best)
	{
		size_t beg = 0;
		size_t end = N;

		double A, C, q;

		bool direction = true;
		while (this->_q >= this->q0k)
		{
			if (direction)
			{
				end--;
				//cout << "end = " << end << endl;
				this->_sums[0] -= X[end];
				this->_sums[1] -= Y[end];
				this->_sums[2] -= pow(X[end], 2.0);
				this->_sums[3] -= pow(Y[end], 2.0);
				this->_sums[4] -= X[end] * Y[end];

				/**(this->_sums) -= X[end];
				*(this->_sums + 1) -= Y[end];
				*(this->_sums + 2) -= pow(X[end], 2.0);
				*(this->_sums + 3) -= pow(Y[end], 2.0);
				*(this->_sums + 4) -= X[end] * Y[end];*/
			}

			else
			{
				//beg++;
				this->_sums[0] += X[beg];
				this->_sums[1] += Y[beg];
				this->_sums[2] += pow(X[beg], 2.0);
				this->_sums[3] += pow(Y[beg], 2.0);
				this->_sums[4] += X[beg] * Y[beg];
				beg++;
			}

			resLMS = Line::LMS(X, Y, 0, N, this->_sums); // returns A, C, q
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

		return new size_t[] { beg, end - N };
	}

	else
		return new size_t[] { 0, 0 };
}


double Line::getDistanceToPnt(double* p, bool sgnd) 
{
	if (sgnd) 
	{
		if (!isinf(this->line[0]))
			return (this->line[0] * p[0] - p[1] + this->line[1]) / sqrt(pow(this->line[0], 2) + 1);
		else
			return (p[0] - this->line[1]);
	} 
	
	else 
	{
		if (!isinf(this->line[0]))
			return abs(this->line[0] * p[0] - p[1] + this->line[1]) / sqrt(pow(this->line[0], 2) + 1);
		else
			return abs(p[0] - this->line[1]);
	}
}


void Line::getProjectionOfPnt(double* p, double** pout) 
{
	if (!isinf(this->line[0]))
	{
		*pout[0] = (p[0] + this->line[0] * p[1] - this->line[0] * this->line[1]) / (pow(this->line[0], 2) + 1.0);
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
			*pout[0] = (p[0] + this->line[0] * p[1] - this->line[0] * this->line[1]) / (pow(this->line[0], 2) + 1.0);
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
		double l = sqrt(pow(p[0], 2) + pow(p[1], 2)) * tan(half_dPhi); // находим точку на линии, отстоящую вправо от точки касания на угол half_dphi
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
	if (this->line[0] == line->line[0])
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


