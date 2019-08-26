#include <iostream>
#include "windows.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

template <typename Derived>
VectorXf solveSVD(const MatrixBase<Derived>& A, const VectorXf & b) {

	VectorXf x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
	cout << "The least-squares using SVD solution is:\n" << x << endl;
	return x;
}
template <typename Derived>
VectorXf solveQRdecomposition(const MatrixBase<Derived>& A, const VectorXf & b) {

	VectorXf x = A.colPivHouseholderQr().solve(b);
	cout << "The solution using the QR decomposition is:\n" << x << endl;
	return x;
}
template <typename Derived>
VectorXf solveNormal(const MatrixBase<Derived>& A, const VectorXf & b) {
	/*
	Finding the least squares solution of Ax = b is equivalent to solving the normal equation ATAx = ATb.
	*/
	VectorXf x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
	cout << "The solution using normal equations is:\n" << x << endl;
	return x;
}

MatrixXf reshape(VectorXf & x, int height, int width) {
	return Map<MatrixXf>(x.data(), height, width);
}
void saveImage(MatrixXf image, string path, bool visualize) {
	
	Mat dst;
	eigen2cv(image,dst);
	cout << "opencv matrix:\n" << dst << endl;
	
	cv::imwrite(path + "imgOut.bmp", dst);
	if (visualize) {
		imshow("Hi", dst);
		cvWaitKey();
	}
}

int main() {

	MatrixXf A = MatrixXf::Random(100, 100);
	cout << "Here is the matrix A:\n" << A << endl;
	VectorXf b = VectorXf::Random(100);
	cout << "Here is the right hand side b:\n" << b << endl;

	VectorXf x = solveSVD(A, b);
	cout << "(main) The least-squares using SVD solution is:\n" << x << endl;
	VectorXf y = solveQRdecomposition(A, b);
	cout << "(main)The solution using the QR decomposition is:\n" << y << endl;
	VectorXf z = solveNormal(A, b);
	cout << "(main)The solution using normal equations is:\n" << x << endl;
	cout << "Reshape: " << reshape(x, 10, 10) << endl;
	saveImage(reshape(x, 10, 10), "", true);

	Sleep(50000);
	return 0;
}