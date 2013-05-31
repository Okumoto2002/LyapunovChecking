//
//  main.cpp
//  SunSpotNumberViaMarkovNetwork
//
//  Created by Masoud Mirmomeni on 10/19/12.
//  Copyright (c) 2012 Masoud Mirmomeni. All rights reserved.
//

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "SunSpot.h"
#include <string>
#include <cstring>
#include <sstream>
#include <array>
#include<math.h>

using namespace boost::numeric::ublas;
using namespace std;

namespace bnu=boost::numeric::ublas;

typedef boost::numeric::ublas::matrix<int> matrix_type; //!< Type for matrix that will store raw sunspot numbers.
typedef boost::numeric::ublas::matrix<double> matrix_type_estimated;
typedef boost::numeric::ublas::matrix_row<matrix_type> row_type; //!< Row type for the matrix.
typedef boost::numeric::ublas::vector<int> vector_type; //!< Type for a vector of sunspot numbers.
typedef boost::numeric::ublas::vector<double> vector_type_distance; //!< Type for a vector of sunspot numbers.

vector_type           _IntegerInput;
matrix_type           _input; //!< All historical sunspot number data used during fitness evaluation (inputs to MKV network).
matrix_type           _observed; //!< Observed (real) historical sunspot numbers.
vector_type           _IntegerObserved;
vector_type_distance  _IntegerObservedED;
matrix_type_estimated _Training;

int MatrixSize;



const int MAX_ED = 7;
const int MAX_NONLINEARITY = 4;
int ParameterOrder [MAX_NONLINEARITY][MAX_ED] = {{1,2,3,4,5,6,7},{8,10,13,17,22,28,35},{36,39,45,55,70,91,119},{120,124,134,154,189,245,329}};


int factorial (int i)
{
    int a = 1;
    if (i<=1)
        return 1;
    
    for (int j = 2; j<=i ; j++)
        a*=j;
    
    return a;
    
}


template<class T>
bool InvertMatrix(const matrix<T>& input, matrix<T>& inverse)
{
	typedef permutation_matrix<std::size_t> pmatrix;
    
	// create a working copy of the input
	matrix<T> A(input);
    
	// create a permutation matrix for the LU-factorization
	pmatrix pm(A.size1());
    
	// perform LU-factorization
	int res = lu_factorize(A, pm);
	if (res != 0)
		return false;
    
	// create identity matrix of "inverse"
	inverse.assign(identity_matrix<T> (A.size1()));
    
	// backsubstitute to get the inverse
	lu_substitute(A, pm, inverse);
    
	return true;
}



// Estimating the embedding dimension.
template <typename Embedding, typename Nonlinearity>
unsigned embedding_dimension(Embedding& d , Nonlinearity& n) {
    
    std::string filename="/Users/mirmomeny/Desktop/Checking/MarkovNetworkBinarykData.txt";
    
    std::ifstream MyFile (filename.c_str());
    
    
    std::string Line;
    MatrixSize=0;
    
    if (MyFile.is_open())
    {
        for (int i = 1; i <= 4; i++)
        {
            getline (MyFile,Line);
        }
        MyFile>>MatrixSize;
    }
    else
    {
        std::cerr<<"Sorry, this file cannot be read."<<std::endl;
        return 0;
    }
    
    _IntegerInput.resize(MatrixSize - 1);
    _IntegerObservedED.resize(MatrixSize - 1);
    int TempSSN = 0;
    MyFile >>TempSSN;
    
    for (int i = 0; i < MatrixSize - 1; i++)
    {
        _IntegerInput(i) = TempSSN;
        MyFile >>TempSSN;
        _IntegerObservedED(i) = TempSSN;
        
    }
    
    double MaxSSN = 0;
    
    for (int i = 0; i < MatrixSize - 1; i++)
    {
        
        if (MaxSSN < _IntegerObservedED(i))
            MaxSSN = _IntegerObservedED(i);
            
    }
    
    
   
     
     for (int i = 0; i < MatrixSize - 1; i++)
    {
        _IntegerObservedED(i) = _IntegerObservedED(i) / MaxSSN;
    }
   
    
    matrix_type_estimated _Parameters;
    matrix_type           _TrainingEstimationMatrix;
    matrix_type_estimated _IntegerEstimatedED;
    vector_type_distance  _TrainError;
    matrix_type_estimated _TrainVector;
    vector_type_distance  _TrainErrorSorted;
    
    _TrainError.resize(MAX_ED * MAX_NONLINEARITY);
    _TrainErrorSorted.resize(MAX_ED * MAX_NONLINEARITY);
    int NumParameters = 0;
    
    NumParameters = factorial(MAX_ED + MAX_NONLINEARITY) / (factorial(MAX_ED) * factorial(MAX_NONLINEARITY));
    _Training.resize(MatrixSize - MAX_ED - 1 , NumParameters);
    _TrainVector.resize(MatrixSize - MAX_ED - 1,1);
    _IntegerEstimatedED.resize(MatrixSize - MAX_ED - 1,1);;
    
    for (int i = 0; i < MatrixSize - MAX_ED - 1; i++)
    {
        _Training(i,0) = 1.0;
        
        for (int j = 1 ; j <= MAX_ED ; j++)
            _Training(i,j) = _IntegerObservedED(i + MAX_ED - j);
        
        _Training(i,8)   = pow(_IntegerObservedED(i + MAX_ED - 1),2);
        
        _Training(i,9)   = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2);
        _Training(i,10)  = pow(_IntegerObservedED(i + MAX_ED - 2),2);
        
        _Training(i,11)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,12)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,13)  = pow(_IntegerObservedED(i + MAX_ED - 3),2);
        
        _Training(i,14)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,15)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,16)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,17)  = pow(_IntegerObservedED(i + MAX_ED - 4),2);
        
        _Training(i,18)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,19)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,20)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,21)  = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,22)  = pow(_IntegerObservedED(i + MAX_ED - 5),2);
        
        _Training(i,23)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,24)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,25)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,26)  = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,27)  = _IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,28)  = pow(_IntegerObservedED(i + MAX_ED - 6),2);
        
        _Training(i,29)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,30)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,31)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,32)  = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,33)  = _IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,34)  = _IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,35)  = pow(_IntegerObservedED(i + MAX_ED - 7),2);
        
        
        
        
        
        _Training(i,36)  = pow(_IntegerObservedED(i + MAX_ED - 1),3);
        
        _Training(i,37)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2);
        _Training(i,38)  = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2);
        _Training(i,39)  = pow(_IntegerObservedED(i + MAX_ED - 2),3);
        
        _Training(i,40)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,41)  = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,42)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,43)  = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),2);
        _Training(i,44)  = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2);
        _Training(i,45)  = pow(_IntegerObservedED(i + MAX_ED - 3),3);
        
        _Training(i,46)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,47)  = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,48)  = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,49)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,50)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,51)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,52)  = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,53)  = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,54)  = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,55)  = pow(_IntegerObservedED(i + MAX_ED - 4),3);
        
        _Training(i,56)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,57)  = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,58)  = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,59)  = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,60)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,61)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,62)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,63)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,64)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,65)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,66)  = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,67)  = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,68)  = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,69)  = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,70)  = pow(_IntegerObservedED(i + MAX_ED - 5),3);
        
        _Training(i,71)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,72)  = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,73)  = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,74)  = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,75)  = pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,76)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,77)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,78)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,79)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,80)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,81)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,82)  = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,83)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,84)  = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,85)  = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,86)  = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,87)  = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,88)  = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,89)  = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,90)  = _IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,91)  = pow(_IntegerObservedED(i + MAX_ED - 6),3);
        
        _Training(i,92)  = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,93)  = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,94)  = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,95)  = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,96)  = pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,97)  = pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,98)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,99)  = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,100) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,101) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,102) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,103) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,104) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,105) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,106) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,107) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,108) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,109) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,110) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,111) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,112) = _IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,113) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,114) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,115) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,116) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,117) = _IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,118) = _IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,119) = pow(_IntegerObservedED(i + MAX_ED - 7),3);
        
        
        
        
        
        
        _Training(i,120) = pow(_IntegerObservedED(i + MAX_ED - 1),4);
        
        _Training(i,121) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 2);
        _Training(i,122) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 2),2);
        _Training(i,123) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),3);
        _Training(i,124) = pow(_IntegerObservedED(i + MAX_ED - 2),4);
        
        _Training(i,125) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,126) = pow(_IntegerObservedED(i + MAX_ED - 2),3)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,127) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,128) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3);
        _Training(i,129) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 3),2);
        _Training(i,130) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2);
        _Training(i,131) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*pow(_IntegerObservedED(i + MAX_ED - 3),2);
        _Training(i,132) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),3);
        _Training(i,133) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),3);
        _Training(i,134) = pow(_IntegerObservedED(i + MAX_ED - 3),4);
        
        _Training(i,135) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,136) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,137) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,138) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,139) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,140) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,141) = pow(_IntegerObservedED(i + MAX_ED - 2),3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,142) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,143) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,144) = pow(_IntegerObservedED(i + MAX_ED - 3),3)*_IntegerObservedED(i + MAX_ED - 4);
        _Training(i,145) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,146) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,147) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,148) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,149) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,150) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*pow(_IntegerObservedED(i + MAX_ED - 4),2);
        _Training(i,151) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 4),3);
        _Training(i,152) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),3);
        _Training(i,153) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),3);
        _Training(i,154) = pow(_IntegerObservedED(i + MAX_ED - 4),4);
        
        _Training(i,155) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,156) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,157) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,158) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,159) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,160) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,161) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,162) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,163) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,164) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,165) = pow(_IntegerObservedED(i + MAX_ED - 2),3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,166) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,167) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,168) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,169) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,170) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,171) = pow(_IntegerObservedED(i + MAX_ED - 3),3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,172) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,173) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,174) = pow(_IntegerObservedED(i + MAX_ED - 4),3)*_IntegerObservedED(i + MAX_ED - 5);
        _Training(i,175) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,176) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,177) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,178) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,179) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,180) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,181) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,182) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,183) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,184) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*pow(_IntegerObservedED(i + MAX_ED - 5),2);
        _Training(i,185) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 5),3);
        _Training(i,186) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 5),3);
        _Training(i,187) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 3),3);
        _Training(i,188) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),3);
        _Training(i,189) = pow(_IntegerObservedED(i + MAX_ED - 5),4);
        
        _Training(i,190) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,191) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,192) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,193) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,194) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,195) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,196) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,197) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,198) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,199) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,200) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,201) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,202) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,203) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,204) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,205) = pow(_IntegerObservedED(i + MAX_ED - 2),3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,206) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,207) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,208) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,209) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,210) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,211) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,212) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,213) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,214) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,215) = pow(_IntegerObservedED(i + MAX_ED - 3),3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,216) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,217) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,218) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,219) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,220) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,221) = pow(_IntegerObservedED(i + MAX_ED - 4),3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,222) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,223) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,224) = pow(_IntegerObservedED(i + MAX_ED - 5),3)*_IntegerObservedED(i + MAX_ED - 6);
        _Training(i,225) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,226) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,227) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,228) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,229) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,230) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,231) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,232) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,233) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,234) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,235) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,236) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,237) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,238) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,239) = pow(_IntegerObservedED(i + MAX_ED - 5),2)*pow(_IntegerObservedED(i + MAX_ED - 6),2);
        _Training(i,240) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 6),3);
        _Training(i,241) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 6),3);
        _Training(i,242) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 6),3);
        _Training(i,243) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),3);
        _Training(i,244) = _IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),3);
        _Training(i,245) = pow(_IntegerObservedED(i + MAX_ED - 6),4);
        
        _Training(i,246) = pow(_IntegerObservedED(i + MAX_ED - 1),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,247) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,248) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,249) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,250) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,251) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,252) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,253) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,254) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,255) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,256) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,257) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,258) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,259) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,260) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,261) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,262) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,263) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,264) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,265) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,266) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,267) = pow(_IntegerObservedED(i + MAX_ED - 2),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,268) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,269) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,270) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,271) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,272) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,273) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,274) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,275) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,276) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,277) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,278) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,279) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,280) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,281) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,282) = pow(_IntegerObservedED(i + MAX_ED - 3),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,283) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,284) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,285) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,286) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,287) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,288) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,289) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,290) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,291) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,292) = pow(_IntegerObservedED(i + MAX_ED - 4),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,293) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,294) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,295) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,296) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,297) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,298) = pow(_IntegerObservedED(i + MAX_ED - 5),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,299) = pow(_IntegerObservedED(i + MAX_ED - 5),2)*_IntegerObservedED(i + MAX_ED - 6)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,300) = _IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 6),2)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,301) = pow(_IntegerObservedED(i + MAX_ED - 6),3)*_IntegerObservedED(i + MAX_ED - 7);
        _Training(i,302) = pow(_IntegerObservedED(i + MAX_ED - 1),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,303) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,304) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,305) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,306) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,307) = _IntegerObservedED(i + MAX_ED - 1)*_IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,308) = pow(_IntegerObservedED(i + MAX_ED - 2),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,309) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,310) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,311) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,312) = _IntegerObservedED(i + MAX_ED - 2)*_IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,313) = pow(_IntegerObservedED(i + MAX_ED - 3),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,314) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,315) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,316) = _IntegerObservedED(i + MAX_ED - 3)*_IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,317) = pow(_IntegerObservedED(i + MAX_ED - 4),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,318) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,319) = _IntegerObservedED(i + MAX_ED - 4)*_IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,320) = pow(_IntegerObservedED(i + MAX_ED - 5),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,321) = _IntegerObservedED(i + MAX_ED - 5)*_IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,322) = pow(_IntegerObservedED(i + MAX_ED - 6),2)*pow(_IntegerObservedED(i + MAX_ED - 7),2);
        _Training(i,323) = _IntegerObservedED(i + MAX_ED - 1)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,324) = _IntegerObservedED(i + MAX_ED - 2)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,325) = _IntegerObservedED(i + MAX_ED - 3)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,326) = _IntegerObservedED(i + MAX_ED - 4)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,327) = _IntegerObservedED(i + MAX_ED - 5)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,328) = _IntegerObservedED(i + MAX_ED - 6)*pow(_IntegerObservedED(i + MAX_ED - 7),3);
        _Training(i,329) = pow(_IntegerObservedED(i + MAX_ED - 7),4);
        

        
        _TrainVector(i,0)  = _IntegerObservedED(i + MAX_ED);
        
    }
    
    matrix_type TempMatrix(MatrixSize - MAX_ED - 1 , 1 , 1);
    int ColumnCounter;
    
    for (int j = 1; j <= MAX_NONLINEARITY; j++)
    {
        for (int i = 1; i <= MAX_ED; i++)
        {
            NumParameters = factorial(i + j) / (factorial(i) * factorial(j));
            _Parameters.resize(NumParameters,1);
            _TrainingEstimationMatrix.resize(MatrixSize - MAX_ED - 1 , NumParameters);
            ColumnCounter = 0;
            
            for (int s = 0 ; s < MatrixSize - MAX_ED - 1 ; s++)
                _TrainingEstimationMatrix(s , ColumnCounter) = 1;
            

            for (int p = 0; p < j ; p++)
            {
                for (int k = ParameterOrder[p][0];k <= ParameterOrder[p][i-1];k++)
                {
                    ColumnCounter++;
                    
                    for (int s = 0 ; s < MatrixSize - MAX_ED - 1 ; s++)
                        _TrainingEstimationMatrix(s , ColumnCounter) = _Training (s , k);
                    
                }
            }
            
            matrix_type_estimated _TrainingTranspose = boost::numeric::ublas::trans(_TrainingEstimationMatrix);
            matrix_type_estimated _TrainingSquare    = boost::numeric::ublas::prod(_TrainingTranspose, _TrainingEstimationMatrix);
            matrix_type_estimated _TrainingInverse;
            _TrainingInverse.resize(NumParameters , NumParameters);
            
            InvertMatrix(_TrainingSquare, _TrainingInverse);
            matrix_type_estimated _RightMatrix = boost::numeric::ublas::prod(_TrainingTranspose, _TrainVector);
            _Parameters = boost::numeric::ublas::prod(_TrainingInverse, _RightMatrix);
            
            //for (int s = 0 ; s < NumParameters ; s++)
            //    cout<< _Parameters(s , 0) << endl;
            
            
            /*****************
             Error Calculation
             *****************/
            
            _IntegerEstimatedED = boost::numeric::ublas::prod (_TrainingEstimationMatrix , _Parameters);
            bnu::vector<double> err;
            err.resize(MatrixSize - MAX_ED - 1);
            
            for (int s = 0 ; s < MatrixSize - MAX_ED - 1 ; s++)
                err(s) = _IntegerEstimatedED(s,0) - _TrainVector(s , 0);
         /*   cout << "a"<<(j - 1) * MAX_ED + (i - 1)<<"=["<<endl;
            for (int s = 0 ; s < MatrixSize - MAX_ED - 1 ; s++)
                cout << _IntegerEstimatedED(s,0) << "    "<< _TrainVector(s , 0) << "    " <<err(s) <<endl ;
            
            cout <<"];"<<endl;
          */
            _TrainError((j - 1) * MAX_ED + (i - 1)) = sqrt(1/static_cast<double>(err.size()) * bnu::inner_prod(err,err));
            
        }
    }
    
    
    _TrainErrorSorted = _TrainError;
    
    std::sort (_TrainErrorSorted.begin(), _TrainErrorSorted.end());
    
    double _TempError=10000;
    const double _Threshold = 0.005;
    
    for (int i = 1 ; i < _TrainErrorSorted.size() ; i++)
    {
        if ((_TrainErrorSorted(i) - _TrainErrorSorted(0))/_TrainErrorSorted(0) < _Threshold)
            _TempError = _TrainErrorSorted(i);
    }
    
    if (_TempError==10000)
        _TempError = _TrainErrorSorted(0);
    
    
    unsigned f = 0;
    
    for (unsigned i = 0 ; i < _TrainError.size() ; i++)
    {
        if (_TrainError(i) == _TempError)
        {
            f = i % MAX_ED + 1;
            d = i % MAX_ED + 1;
            n = i / MAX_ED + 1;
            return f;
        }
    }
    
    return f;
}


// QR decomposition.
template <typename MatrixQ, typename MatrixR, typename MatrixA>
bool QR_factorization(MatrixQ& Q , MatrixR& R , MatrixA& JacobianMatrix) {
    
    using namespace boost::numeric::ublas;

    int d = JacobianMatrix.size1();
    
    for (int i = 0; i < d; i++)
    {
        Q(i,0) = JacobianMatrix(i,0);
    }
    
    double _TempNorm = 0.0;
    double _TempProject = 0.0;
    
    for (int i = 0; i < d; i++)
    {
        _TempNorm += Q(i , 0) * Q(i , 0);
    }
    
    for (int i = 0; i < d; i++)
    {
        Q(i,0) = Q(i,0) / std::sqrt(_TempNorm);
    }
    
    for (int i = 0; i < d; i++)
    {
        for (int j = i + 1; j < d; j++)
        {
            R(i,j)=0;
        }
    }
    
    for (int i = 1; i < d; i++)
    {
        
        for (int j = 0; j < d; j++)
        {
            Q(j , i) = JacobianMatrix(j , i);
        }
        
        for (int j = 0; j < i; j++)
        {
            _TempProject = 0.0;
            
            for (int k = 0; k < d; k++)
            {
                _TempProject += JacobianMatrix(k , i) * Q(k , j);
            }

            for (int k = 0; k < d; k++)
            {
                Q(k,i) -= _TempProject * Q(k , j);
            }

        }
        
        _TempNorm = 0;
        
        for (int j = 0; j < d; j++)
        {
            _TempNorm += Q(j , i) * Q(j , i);
        }
        
        for (int j = 0; j < d; j++)
        {
            Q(j,i) = Q(j,i) / std::sqrt(_TempNorm);
        }
        
    }
    
    
    for (int i = 0; i < d; i++)
    {
        for (int j = i; j < d; j++)
        {
            _TempProject  = 0;
            
            for (int k = 0; k < d; k++)
            {
                _TempProject += JacobianMatrix(k , j) * Q(k , i);
            }
            
            R(i, j) = _TempProject;
        }
    }
    
    return true;
}


// Estimating the Lyapunov exponent first approach.
template <typename Embedding, typename Nonlinearity, typename Lyapunov, typename LargestLyapunov>
double lyapunov_estimation(Embedding& d , Nonlinearity& n , Lyapunov& _LEs , LargestLyapunov& _LargestLyapunov) {
    namespace bnu=boost::numeric::ublas;
    // input data can be found here (defined in config file or command line):
    
    /**************
     Initialization
     **************/
    
    for (int k = 0 ; k < d ; k++)
        _LEs(k) = 0;
    
    int NumParameters = factorial(d + n) / (factorial(d) * factorial(n));
    
    bool label;
    matrix_type_estimated _Parameters;
    matrix_type_estimated _Regressor;
    matrix_type_estimated _RegressorTranspose;
    matrix_type_estimated _Variance;
    matrix_type_estimated _TempOne;
    matrix_type_estimated _TempTwo;
    matrix_type_estimated _TempThree;
    matrix_type_estimated _TempFour;
    matrix_type_estimated _Denominator;
    matrix_type_estimated _Identity;
    matrix_type_estimated _InverseDenominator;
    matrix_type_estimated _EvolutionTerm;
    matrix_type_estimated _Jacobian;
    matrix_type_estimated _Q;
    matrix_type_estimated _Q0;
    matrix_type_estimated _TempJacobian;
    matrix_type_estimated _R;
    
    _Jacobian.resize(d,d);
    _TempJacobian.resize(d,d);
    _Q.resize(d,d);
    _Q0.resize(d,d);
    
    for (int i = 0; i <d; i++)
        for (int j = 0; j <d; j++)
        {
            _Q0(i,j) = 0;
            
            if (i == j)
                _Q0(i,j) = 1;
        }
    _R.resize(d,d);
    _Regressor.resize(NumParameters,1);
    
    
    for (int i = 0; i < d - 1; i++)
    {
        for (int j = 0; j < d; j++)
        {
            _Jacobian(i,j)=0;
        }
        _Jacobian(i,i+1)=1;
    }
    
    _Parameters.resize(NumParameters,1);
    _Identity.resize(1,1);
    _Identity(0,0)=1;
    
    matrix_type_estimated _ActualInput;
    matrix_type_estimated _EstimatedInput;
    matrix_type_estimated _Error;
    
    _ActualInput.resize(1,1);
    _EstimatedInput.resize(1,1);
    _Error.resize(1,1);
    
    
    _Regressor.resize(NumParameters,1);
    _Variance.resize(NumParameters,NumParameters);
    
    srand((unsigned)time(0));
    
    for(int i=0; i<NumParameters; i++)
        _Parameters(i,0) = (rand()%10)+1;
    
    for(int i=0; i<NumParameters; i++){
        
        for(int j=0; j<NumParameters; j++)
            _Variance(i,j)=0;
        _Variance(i,i)=1000000;
    }
    
    
    /**********
     Estimation
     **********/
    
    for (int i = 0; i < MatrixSize - MAX_ED - 1; i++)
    {
        
        /********************
         Parameter Estimation
         ********************/
        
        int ColumnCounter = 0;
        _Regressor(0,0)   = 1;
        
        for (int p = 0; p < n ; p++)
        {
            for (int k = ParameterOrder[p][0];k <= ParameterOrder[p][d-1];k++)
            {
                ColumnCounter++;
                _Regressor(ColumnCounter , 0) = _Training(i , k);
            }
        }
        
        
        
        _ActualInput(0,0) = _IntegerObservedED(i + MAX_ED);
        _RegressorTranspose.resize(1,NumParameters);
        _TempOne.resize(1,NumParameters);
        _TempTwo.resize(NumParameters,1);
        _TempThree.resize(NumParameters,NumParameters);
        _TempFour.resize(1,1);
        _Identity.resize(1, 1);
        _Identity(0, 0) = 1;
        _InverseDenominator.resize(1,1);
        _EvolutionTerm.resize(NumParameters,NumParameters);
        
        _RegressorTranspose = boost::numeric::ublas::trans(_Regressor);
        _TempOne            = boost::numeric::ublas::prod(_RegressorTranspose, _Variance);
        _TempTwo            = boost::numeric::ublas::prod(_Variance , _Regressor);
        _TempThree          = boost::numeric::ublas::prod(_TempTwo, _TempOne);
        _TempFour           = boost::numeric::ublas::prod(_RegressorTranspose, _TempTwo);
        _Denominator.resize(1,1);
        _Denominator        = _Identity + _TempFour;
        InvertMatrix(_Denominator, _InverseDenominator);
        _EvolutionTerm = _InverseDenominator(0,0) * _TempThree;
        _Variance -= _EvolutionTerm;
        matrix_type_estimated _TransParameters;
        _TransParameters.resize(1,NumParameters);
        
        _TransParameters = boost::numeric::ublas::trans(_Parameters);
        _EstimatedInput  = boost::numeric::ublas::prod(_TransParameters , _Regressor);
        
        _Error = _ActualInput - _EstimatedInput;
        
        matrix_type_estimated _EvolutionParameterOne;
        _EvolutionParameterOne.resize(NumParameters, 1);
        _EvolutionParameterOne = boost::numeric::ublas::prod(_Variance , _Regressor);
        
        matrix_type_estimated _EvolutionParameter;
        _EvolutionParameter.resize(NumParameters, 1);
        
        _EvolutionParameter = _Error(0,0)*_EvolutionParameterOne;
        _Parameters += _EvolutionParameter;
        
        /*******************
         Jacobian Estimation
         *******************/
        
        switch (n)
        {
            case 1:
                
                switch (d)
            {
                case 1:
                    _Jacobian(0,0) = _Parameters(1,0);
                    break;
                case 2:
                    _Jacobian(1,0) = _Parameters(1,0);
                    _Jacobian(1,1) = _Parameters(2,0);
                    break;
                case 3:
                    _Jacobian(2,0) = _Parameters(1,0);
                    _Jacobian(2,1) = _Parameters(2,0);
                    _Jacobian(2,2) = _Parameters(3,0);
                    break;
                case 4:
                    _Jacobian(3,0) = _Parameters(1,0);
                    _Jacobian(3,1) = _Parameters(2,0);
                    _Jacobian(3,2) = _Parameters(3,0);
                    _Jacobian(3,3) = _Parameters(4,0);
                    break;
                case 5:
                    _Jacobian(4,0) = _Parameters(1,0);
                    _Jacobian(4,1) = _Parameters(2,0);
                    _Jacobian(4,2) = _Parameters(3,0);
                    _Jacobian(4,3) = _Parameters(4,0);
                    _Jacobian(4,4) = _Parameters(5,0);
                    break;
                case 6:
                    _Jacobian(5,0) = _Parameters(1,0);
                    _Jacobian(5,1) = _Parameters(2,0);
                    _Jacobian(5,2) = _Parameters(3,0);
                    _Jacobian(5,3) = _Parameters(4,0);
                    _Jacobian(5,4) = _Parameters(5,0);
                    _Jacobian(5,5) = _Parameters(6,0);
                    break;
                case 7:
                    _Jacobian(6,0) = _Parameters(1,0);
                    _Jacobian(6,1) = _Parameters(2,0);
                    _Jacobian(6,2) = _Parameters(3,0);
                    _Jacobian(6,3) = _Parameters(4,0);
                    _Jacobian(6,4) = _Parameters(5,0);
                    _Jacobian(6,5) = _Parameters(6,0);
                    _Jacobian(6,6) = _Parameters(7,0);
                    break;
            }
                break;
                
            case 2:
                switch (d)
            {
                case 1:
                    _Jacobian(0,0) = _Parameters(1,0) + 2 * _Parameters(2,0) * _Training(i,1);
                    break;
                case 2:
                    _Jacobian(1,0) = _Parameters(1,0) + 2 * _Parameters(3,0) * _Training(i,1) + _Parameters(4,0) * _Training(i,2);
                    _Jacobian(1,1) = _Parameters(2,0) + _Parameters(4,0) * _Training(i,1) + 2 * _Parameters(5,0) * _Training(i,2);
                    break;
                case 3:
                    _Jacobian(2,0) = _Parameters(1,0)+2*_Parameters(4,0)*_Training(i,1)+_Parameters(5,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,3);
                    _Jacobian(2,1) = _Parameters(2,0)+2*_Parameters(6,0)*_Training(i,2)+_Parameters(5,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,3);
                    _Jacobian(2,2) = _Parameters(3,0)+2*_Parameters(9,0)*_Training(i,3)+_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2);
                    break;
                case 4:
                    _Jacobian(3,0) = _Parameters(1,0)+2*_Parameters(5,0)*_Training(i,1)+_Parameters(6,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,3)+_Parameters(11,0)*_Training(i,4);
                    _Jacobian(3,1) = _Parameters(2,0)+2*_Parameters(7,0)*_Training(i,2)+_Parameters(6,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4);
                    _Jacobian(3,2) = _Parameters(3,0)+2*_Parameters(10,0)*_Training(i,3)+_Parameters(8,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,2)+_Parameters(13,0)*_Training(i,4);
                    _Jacobian(3,3) = _Parameters(4,0)+2*_Parameters(14,0)*_Training(i,4)+_Parameters(11,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,2)+_Parameters(13,0)*_Training(i,4);
                    break;
                case 5:
                    _Jacobian(4,0) = _Parameters(1,0)+2*_Parameters(6,0)*_Training(i,1)+_Parameters(7,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4)+_Parameters(16,0)*_Training(i,5);
                    _Jacobian(4,1) = _Parameters(2,0)+2*_Parameters(8,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5);
                    _Jacobian(4,2) = _Parameters(3,0)+2*_Parameters(11,0)*_Training(i,3)+_Parameters(9,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5);
                    _Jacobian(4,3) = _Parameters(4,0)+2*_Parameters(15,0)*_Training(i,4)+_Parameters(12,0)*_Training(i,1)+_Parameters(13,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5);
                    _Jacobian(4,4) = _Parameters(5,0)+2*_Parameters(20,0)*_Training(i,5)+_Parameters(16,0)*_Training(i,1)+_Parameters(17,0)*_Training(i,2)+_Parameters(18,0)*_Training(i,3)+_Parameters(19,0)*_Training(i,4);
                    break;
                case 6:
                    _Jacobian(5,0) = _Parameters(1,0)+2*_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5)+_Parameters(22,0)*_Training(i,6);
                    _Jacobian(5,1) = _Parameters(2,0)+2*_Parameters(9,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6);
                    _Jacobian(5,2) = _Parameters(3,0)+2*_Parameters(12,0)*_Training(i,3)+_Parameters(10,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,2)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6);
                    _Jacobian(5,3) = _Parameters(4,0)+2*_Parameters(16,0)*_Training(i,4)+_Parameters(13,0)*_Training(i,1)+_Parameters(14,0)*_Training(i,2)+_Parameters(15,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6);
                    _Jacobian(5,4) = _Parameters(5,0)+2*_Parameters(21,0)*_Training(i,5)+_Parameters(17,0)*_Training(i,1)+_Parameters(18,0)*_Training(i,2)+_Parameters(19,0)*_Training(i,3)+_Parameters(20,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,6);
                    _Jacobian(5,5) = _Parameters(6,0)+2*_Parameters(27,0)*_Training(i,6)+_Parameters(22,0)*_Training(i,1)+_Parameters(23,0)*_Training(i,2)+_Parameters(24,0)*_Training(i,3)+_Parameters(25,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,5);
                    break;
                case 7:
                    _Jacobian(5,0) = _Parameters(1,0)+2*_Parameters(8,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,2)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6)+_Parameters(29,0)*_Training(i,7);
                    _Jacobian(5,1) = _Parameters(2,0)+2*_Parameters(10,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,3)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6)+_Parameters(30,0)*_Training(i,7);
                    _Jacobian(5,2) = _Parameters(3,0)+2*_Parameters(13,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,2)+_Parameters(16,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6)+_Parameters(31,0)*_Training(i,7);
                    _Jacobian(5,3) = _Parameters(4,0)+2*_Parameters(17,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,1)+_Parameters(15,0)*_Training(i,2)+_Parameters(16,0)*_Training(i,4)+_Parameters(21,0)*_Training(i,5)+_Parameters(26,0)*_Training(i,6)+_Parameters(32,0)*_Training(i,7);
                    _Jacobian(5,4) = _Parameters(5,0)+2*_Parameters(22,0)*_Training(i,5)+_Parameters(18,0)*_Training(i,1)+_Parameters(19,0)*_Training(i,2)+_Parameters(20,0)*_Training(i,3)+_Parameters(21,0)*_Training(i,4)+_Parameters(27,0)*_Training(i,6)+_Parameters(33,0)*_Training(i,7);
                    _Jacobian(5,5) = _Parameters(6,0)+2*_Parameters(28,0)*_Training(i,6)+_Parameters(23,0)*_Training(i,1)+_Parameters(24,0)*_Training(i,2)+_Parameters(25,0)*_Training(i,3)+_Parameters(26,0)*_Training(i,4)+_Parameters(27,0)*_Training(i,5)+_Parameters(34,0)*_Training(i,7);
                    _Jacobian(6,6) = _Parameters(7,0)+2*_Parameters(35,0)*_Training(i,7)+_Parameters(29,0)*_Training(i,1)+_Parameters(30,0)*_Training(i,2)+_Parameters(31,0)*_Training(i,3)+_Parameters(32,0)*_Training(i,4)+_Parameters(33,0)*_Training(i,5)+_Parameters(34,0)*_Training(i,6);
                    break;
            }
                break;
                
            case 3:
                switch (d)
            {
                case 1:
                    _Jacobian(0,0) = _Parameters(1,0)+2*_Parameters(2,0)*_Training(i,1)+3*_Parameters(3,0)*pow(_Training(i,1),2);
                    break;
                case 2:
                    _Jacobian(1,0) = _Parameters(1,0)+2*_Parameters(3,0)*_Training(i,1)+_Parameters(4,0)*_Training(i,2)+3*_Parameters(6,0)*pow(_Training(i,1),2)+2*_Parameters(7,0)*_Training(i,1)*_Training(i,2)+_Parameters(8,0)*pow(_Training(i,2),2);
                    
                    _Jacobian(1,1) = _Parameters(2,0)+2*_Parameters(5,0)*_Training(i,5)+_Parameters(4,0)*_Training(i,1)+3*_Parameters(9,0)*pow(_Training(i,2),2)+2*_Parameters(8,0)*_Training(i,1)*_Training(i,2)+_Parameters(7,0)*pow(_Training(i,1),2);
                    break;
                case 3:
                    
                    _Jacobian(2,0) = _Parameters(1,0)+2*_Parameters(4,0)*_Training(i,1)+_Parameters(5,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,3)+3*_Parameters(10,0)*pow(_Training(i,1),2)+2*_Parameters(11,0)*_Training(i,1)*_Training(i,2)+_Parameters(12,0)*pow(_Training(i,2),2)+2*_Parameters(14,0)*_Training(i,1)*_Training(i,3)+_Parameters(16,0)*_Training(i,2)*_Training(i,3)+_Parameters(17,0)*pow(_Training(i,3),2);
                    
                    _Jacobian(2,1) = _Parameters(2,0)+2*_Parameters(6,0)*_Training(i,2)+_Parameters(5,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,3)+3*_Parameters(13,0)*pow(_Training(i,2),2)+2*_Parameters(12,0)*_Training(i,1)*_Training(i,2)+_Parameters(11,0)*pow(_Training(i,1),2)+2*_Parameters(15,0)*_Training(i,2)*_Training(i,3)+_Parameters(16,0)*_Training(i,1)*_Training(i,3)+_Parameters(18,0)*pow(_Training(i,3),2);
                    
                    _Jacobian(2,2) = _Parameters(3,0)+2*_Parameters(9,0)*_Training(i,3)+_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2)+3*_Parameters(19,0)*pow(_Training(i,3),2)+2*_Parameters(17,0)*_Training(i,1)*_Training(i,3)+_Parameters(14,0)*pow(_Training(i,1),2)+2*_Parameters(18,0)*_Training(i,2)*_Training(i,3)+_Parameters(16,0)*_Training(i,1)*_Training(i,2)+_Parameters(15,0)*pow(_Training(i,2),2);
                    break;
                case 4:
                    _Jacobian(3,0) = _Parameters(1,0)+2*_Parameters(5,0)*_Training(i,1)+_Parameters(6,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,3)+_Parameters(11,0)*_Training(i,4)+3*_Parameters(15,0)*pow(_Training(i,1),2)+2*_Parameters(16,0)*_Training(i,1)*_Training(i,2)+_Parameters(17,0)*pow(_Training(i,2),2)+2*_Parameters(19,0)*_Training(i,1)*_Training(i,3)+_Parameters(21,0)*_Training(i,2)*_Training(i,3)+_Parameters(22,0)*pow(_Training(i,3),2)+2*_Parameters(25,0)*_Training(i,1)*_Training(i,4)+_Parameters(28,0)*_Training(i,2)*_Training(i,4)+_Parameters(29,0)*_Training(i,3)*_Training(i,4)+_Parameters(31,0)*pow(_Training(i,4),2);
                    
                    _Jacobian(3,1) = _Parameters(2,0)+2*_Parameters(7,0)*_Training(i,2)+_Parameters(6,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4)+3*_Parameters(18,0)*pow(_Training(i,2),2)+2*_Parameters(17,0)*_Training(i,1)*_Training(i,2)+_Parameters(16,0)*pow(_Training(i,1),2)+2*_Parameters(20,0)*_Training(i,2)*_Training(i,3)+_Parameters(21,0)*_Training(i,1)*_Training(i,3)+_Parameters(23,0)*pow(_Training(i,3),2)+2*_Parameters(26,0)*_Training(i,2)*_Training(i,4)+_Parameters(28,0)*_Training(i,1)*_Training(i,4)+_Parameters(30,0)*_Training(i,3)*_Training(i,4)+_Parameters(32,0)*pow(_Training(i,4),2);
                    
                    _Jacobian(3,2) = _Parameters(3,0)+2*_Parameters(10,0)*_Training(i,3)+_Parameters(9,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,1)+_Parameters(13,0)*_Training(i,4)+3*_Parameters(24,0)*pow(_Training(i,3),2)+2*_Parameters(22,0)*_Training(i,1)*_Training(i,3)+_Parameters(19,0)*pow(_Training(i,1),2)+2*_Parameters(23,0)*_Training(i,2)*_Training(i,3)+_Parameters(21,0)*_Training(i,1)*_Training(i,2)+_Parameters(20,0)*pow(_Training(i,2),2)+2*_Parameters(27,0)*_Training(i,3)*_Training(i,4)+_Parameters(29,0)*_Training(i,1)*_Training(i,4)+_Parameters(30,0)*_Training(i,2)*_Training(i,4)+_Parameters(33,0)*pow(_Training(i,4),2);
                    
                    _Jacobian(3,3) = _Parameters(4,0)+_Parameters(11,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,2)+_Parameters(13,0)*_Training(i,3)+2*_Parameters(14,0)*_Training(i,4)+_Parameters(25,0)*pow(_Training(i,1),2)+_Parameters(26,0)*pow(_Training(i,2),2)+_Parameters(27,0)*pow(_Training(i,3),2)+_Parameters(28,0)*_Training(i,1)*_Training(i,2)+_Parameters(29,0)*_Training(i,1)*_Training(i,3)+_Parameters(30,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(31,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(32,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(33,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(34,0)*pow(_Training(i,4),2);
                    break;
                case 5:
                    _Jacobian(4,0) = _Parameters(1,0)+2*_Parameters(6,0)*_Training(i,1)+_Parameters(7,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4)+_Parameters(16,0)*_Training(i,5)+3*_Parameters(21,0)*pow(_Training(i,1),2)+2*_Parameters(22,0)*_Training(i,1)*_Training(i,2)+_Parameters(23,0)*pow(_Training(i,2),2)+2*_Parameters(25,0)*_Training(i,1)*_Training(i,3)+_Parameters(27,0)*_Training(i,2)*_Training(i,3)+_Parameters(28,0)*pow(_Training(i,3),2)+2*_Parameters(31,0)*_Training(i,1)*_Training(i,4)+_Parameters(34,0)*_Training(i,2)*_Training(i,4)+_Parameters(35,0)*_Training(i,3)*_Training(i,4)+_Parameters(37,0)*pow(_Training(i,4),2)+2*_Parameters(41,0)*_Training(i,1)*_Training(i,5)+_Parameters(45,0)*_Training(i,2)*_Training(i,5)+_Parameters(46,0)*_Training(i,3)*_Training(i,5)+_Parameters(47,0)*_Training(i,4)*_Training(i,5)+_Parameters(51,0)*pow(_Training(i,5),2);
                    
                    _Jacobian(4,1) = _Parameters(2,0)+2*_Parameters(8,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5)+3*_Parameters(24,0)*pow(_Training(i,2),2)+2*_Parameters(23,0)*_Training(i,1)*_Training(i,2)+_Parameters(22,0)*pow(_Training(i,1),2)+2*_Parameters(26,0)*_Training(i,2)*_Training(i,3)+_Parameters(27,0)*_Training(i,1)*_Training(i,3)+_Parameters(29,0)*pow(_Training(i,3),2)+2*_Parameters(32,0)*_Training(i,2)*_Training(i,4)+_Parameters(34,0)*_Training(i,1)*_Training(i,4)+_Parameters(36,0)*_Training(i,3)*_Training(i,4)+_Parameters(38,0)*pow(_Training(i,4),2)+2*_Parameters(42,0)*_Training(i,2)*_Training(i,5)+_Parameters(45,0)*_Training(i,1)*_Training(i,5)+_Parameters(48,0)*_Training(i,3)*_Training(i,5)+_Parameters(49,0)*_Training(i,4)*_Training(i,5)+_Parameters(52,0)*pow(_Training(i,5),2);
                    
                    _Jacobian(4,2) = _Parameters(3,0)+2*_Parameters(11,0)*_Training(i,3)+_Parameters(9,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+3*_Parameters(30,0)*pow(_Training(i,3),2)+2*_Parameters(28,0)*_Training(i,1)*_Training(i,3)+_Parameters(25,0)*pow(_Training(i,1),2)+2*_Parameters(29,0)*_Training(i,2)*_Training(i,3)+_Parameters(27,0)*_Training(i,1)*_Training(i,2)+_Parameters(26,0)*pow(_Training(i,2),2)+2*_Parameters(33,0)*_Training(i,3)*_Training(i,4)+_Parameters(35,0)*_Training(i,1)*_Training(i,4)+_Parameters(36,0)*_Training(i,2)*_Training(i,4)+_Parameters(39,0)*pow(_Training(i,4),2)+2*_Parameters(43,0)*_Training(i,3)*_Training(i,5)+_Parameters(46,0)*_Training(i,1)*_Training(i,5)+_Parameters(48,0)*_Training(i,2)*_Training(i,5)+_Parameters(50,0)*_Training(i,4)*_Training(i,5)+_Parameters(53,0)*pow(_Training(i,5),2);
                    
                    _Jacobian(4,3) = _Parameters(4,0)+2*_Parameters(15,0)*_Training(i,4)+_Parameters(12,0)*_Training(i,1)+_Parameters(13,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,3)+_Parameters(19,0)*_Training(i,5)+3*_Parameters(40,0)*pow(_Training(i,4),2)+2*_Parameters(37,0)*_Training(i,1)*_Training(i,4)+_Parameters(31,0)*pow(_Training(i,1),2)+2*_Parameters(38,0)*_Training(i,2)*_Training(i,4)+_Parameters(34,0)*_Training(i,1)*_Training(i,2)+_Parameters(32,0)*pow(_Training(i,2),2)+2*_Parameters(39,0)*_Training(i,3)*_Training(i,4)+_Parameters(35,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*_Training(i,2)*_Training(i,3)+_Parameters(33,0)*pow(_Training(i,3),2)+2*_Parameters(44,0)*_Training(i,4)*_Training(i,5)+_Parameters(47,0)*_Training(i,1)*_Training(i,5)+_Parameters(49,0)*_Training(i,2)*_Training(i,5)+_Parameters(50,0)*_Training(i,3)*_Training(i,5)+_Parameters(54,0)*pow(_Training(i,5),2);
                    
                    _Jacobian(4,4) = _Parameters(5,0)+2*_Parameters(20,0)*_Training(i,5)+_Parameters(16,0)*_Training(i,1)+_Parameters(17,0)*_Training(i,2)+_Parameters(18,0)*_Training(i,3)+_Parameters(19,0)*_Training(i,4)+3*_Parameters(55,0)*pow(_Training(i,5),2)+2*_Parameters(51,0)*_Training(i,1)*_Training(i,5)+_Parameters(41,0)*pow(_Training(i,1),2)+2*_Parameters(52,0)*_Training(i,2)*_Training(i,5)+_Parameters(45,0)*_Training(i,1)*_Training(i,2)+_Parameters(42,0)*pow(_Training(i,2),2)+2*_Parameters(53,0)*_Training(i,3)*_Training(i,5)+_Parameters(46,0)*_Training(i,1)*_Training(i,3)+_Parameters(48,0)*_Training(i,2)*_Training(i,3)+_Parameters(43,0)*pow(_Training(i,3),2)+2*_Parameters(54,0)*_Training(i,4)*_Training(i,5)+_Parameters(47,0)*_Training(i,1)*_Training(i,4)+_Parameters(49,0)*_Training(i,2)*_Training(i,4)+_Parameters(50,0)*_Training(i,3)*_Training(i,4)+_Parameters(44,0)*pow(_Training(i,4),2);
                    
                    break;
                case 6:
                    _Jacobian(5,0) = _Parameters(1,0)+2*_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5)+_Parameters(22,0)*_Training(i,6)+3*_Parameters(28,0)*pow(_Training(i,1),2)+2*_Parameters(29,0)*_Training(i,1)*_Training(i,2)+_Parameters(30,0)*pow(_Training(i,2),2)+2*_Parameters(32,0)*_Training(i,1)*_Training(i,2)+_Parameters(34,0)*_Training(i,2)*_Training(i,3)+_Parameters(35,0)*pow(_Training(i,3),2)+2*_Parameters(38,0)*_Training(i,1)*_Training(i,4)+_Parameters(41,0)*_Training(i,2)*_Training(i,4)+_Parameters(42,0)*_Training(i,3)*_Training(i,4)+_Parameters(44,0)*pow(_Training(i,4),2)+2*_Parameters(48,0)*_Training(i,1)*_Training(i,5)+_Parameters(52,0)*_Training(i,2)*_Training(i,5)+_Parameters(53,0)*_Training(i,3)*_Training(i,5)+_Parameters(54,0)*_Training(i,4)*_Training(i,5)+_Parameters(58,0)*pow(_Training(i,5),2)+2*_Parameters(63,0)*_Training(i,1)*_Training(i,6)+_Parameters(68,0)*_Training(i,2)*_Training(i,6)+_Parameters(69,0)*_Training(i,3)*_Training(i,6)+_Parameters(70,0)*_Training(i,4)*_Training(i,6)+_Parameters(71,0)*_Training(i,5)*_Training(i,6)+_Parameters(78,0)*pow(_Training(i,6),2);
                    
                    _Jacobian(5,1) = _Parameters(2,0)+2*_Parameters(9,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6)+3*_Parameters(31,0)*pow(_Training(i,2),2)+2*_Parameters(30,0)*_Training(i,1)*_Training(i,2)+_Parameters(29,0)*pow(_Training(i,1),2)+2*_Parameters(33,0)*_Training(i,2)*_Training(i,3)+_Parameters(34,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*pow(_Training(i,3),2)+2*_Parameters(39,0)*_Training(i,2)*_Training(i,4)+_Parameters(41,0)*_Training(i,1)*_Training(i,4)+_Parameters(43,0)*_Training(i,3)*_Training(i,4)+_Parameters(45,0)*pow(_Training(i,4),2)+2*_Parameters(49,0)*_Training(i,2)*_Training(i,5)+_Parameters(52,0)*_Training(i,1)*_Training(i,5)+_Parameters(55,0)*_Training(i,3)*_Training(i,5)+_Parameters(56,0)*_Training(i,4)*_Training(i,5)+_Parameters(59,0)*pow(_Training(i,5),2)+2*_Parameters(64,0)*_Training(i,2)*_Training(i,6)+_Parameters(68,0)*_Training(i,1)*_Training(i,6)+_Parameters(72,0)*_Training(i,3)*_Training(i,6)+_Parameters(73,0)*_Training(i,4)*_Training(i,6)+_Parameters(74,0)*_Training(i,5)*_Training(i,6)+_Parameters(79,0)*pow(_Training(i,6),2);
                    
                    _Jacobian(5,2) = _Parameters(3,0)+_Parameters(10,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,2)+2*_Parameters(12,0)*_Training(i,3)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6)+_Parameters(32,0)*pow(_Training(i,1),2)+_Parameters(33,0)*pow(_Training(i,2),2)+_Parameters(34,0)*_Training(i,1)*_Training(i,2)+2*_Parameters(35,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*_Training(i,2)*_Training(i,3)+3*_Parameters(37,0)*pow(_Training(i,3),2)+2*_Parameters(40,0)*_Training(i,3)*_Training(i,4)+_Parameters(42,0)*_Training(i,1)*_Training(i,4)+_Parameters(43,0)*_Training(i,2)*_Training(i,4)+_Parameters(46,0)*pow(_Training(i,4),2)+2*_Parameters(50,0)*_Training(i,3)*_Training(i,5)+_Parameters(53,0)*_Training(i,1)*_Training(i,5)+_Parameters(55,0)*_Training(i,2)*_Training(i,5)+_Parameters(57,0)*_Training(i,4)*_Training(i,5)+_Parameters(60,0)*pow(_Training(i,5),2)+2*_Parameters(65,0)*_Training(i,3)*_Training(i,6)+_Parameters(69,0)*_Training(i,1)*_Training(i,6)+_Parameters(72,0)*_Training(i,2)*_Training(i,6)+_Parameters(75,0)*_Training(i,4)*_Training(i,6)+_Parameters(76,0)*_Training(i,5)*_Training(i,6)+_Parameters(80,0)*pow(_Training(i,6),2);
                    
                    _Jacobian(5,3) = _Parameters(4,0)+_Parameters(13,0)*_Training(i,1)+_Parameters(14,0)*_Training(i,2)+_Parameters(15,0)*_Training(i,3)+2*_Parameters(16,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6)+_Parameters(38,0)*pow(_Training(i,1),2)+_Parameters(39,0)*pow(_Training(i,2),2)+_Parameters(40,0)*pow(_Training(i,3),2)+_Parameters(41,0)*_Training(i,1)*_Training(i,2)+_Parameters(42,0)*_Training(i,1)*_Training(i,3)+_Parameters(43,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(44,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(45,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(46,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(47,0)*pow(_Training(i,4),2)+2*_Parameters(51,0)*_Training(i,4)*_Training(i,5)+_Parameters(54,0)*_Training(i,1)*_Training(i,5)+_Parameters(56,0)*_Training(i,2)*_Training(i,5)+_Parameters(57,0)*_Training(i,3)*_Training(i,5)+_Parameters(61,0)*pow(_Training(i,5),2)+2*_Parameters(66,0)*_Training(i,4)*_Training(i,6)+_Parameters(70,0)*_Training(i,1)*_Training(i,6)+_Parameters(73,0)*_Training(i,2)*_Training(i,6)+_Parameters(75,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,5)*_Training(i,6)+_Parameters(81,0)*pow(_Training(i,6),2);
                    
                    _Jacobian(5,4) = _Parameters(5,0)+_Parameters(17,0)*_Training(i,1)+_Parameters(18,0)*_Training(i,2)+_Parameters(19,0)*_Training(i,3)+2*_Parameters(21,0)*_Training(i,5)+_Parameters(20,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,6)+_Parameters(48,0)*pow(_Training(i,1),2)+_Parameters(49,0)*pow(_Training(i,2),2)+_Parameters(50,0)*pow(_Training(i,3),2)+_Parameters(51,0)*pow(_Training(i,4),2)+_Parameters(52,0)*_Training(i,1)*_Training(i,2)+_Parameters(53,0)*_Training(i,1)*_Training(i,3)+_Parameters(54,0)*_Training(i,1)*_Training(i,4)+_Parameters(55,0)*_Training(i,2)*_Training(i,3)+_Parameters(56,0)*_Training(i,2)*_Training(i,4)+_Parameters(57,0)*_Training(i,3)*_Training(i,4)+2*_Parameters(58,0)*_Training(i,1)*_Training(i,5)+2*_Parameters(59,0)*_Training(i,2)*_Training(i,5)+2*_Parameters(60,0)*_Training(i,3)*_Training(i,5)+2*_Parameters(61,0)*_Training(i,4)*_Training(i,5)+3*_Parameters(62,0)*pow(_Training(i,5),2)+2*_Parameters(67,0)*_Training(i,5)*_Training(i,6)+_Parameters(71,0)*_Training(i,1)*_Training(i,6)+_Parameters(74,0)*_Training(i,2)*_Training(i,6)+_Parameters(76,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,4)*_Training(i,6)+_Parameters(82,0)*pow(_Training(i,6),2);
                    
                    _Jacobian(5,5) = _Parameters(6,0)+_Parameters(22,0)*_Training(i,1)+_Parameters(23,0)*_Training(i,2)+_Parameters(24,0)*_Training(i,3)+2*_Parameters(27,0)*_Training(i,6)+_Parameters(25,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,5)+_Parameters(63,0)*pow(_Training(i,1),2)+_Parameters(64,0)*pow(_Training(i,2),2)+_Parameters(65,0)*pow(_Training(i,3),2)+_Parameters(66,0)*pow(_Training(i,4),2)+_Parameters(68,0)*_Training(i,1)*_Training(i,2)+_Parameters(69,0)*_Training(i,1)*_Training(i,3)+_Parameters(70,0)*_Training(i,1)*_Training(i,4)+_Parameters(71,0)*_Training(i,1)*_Training(i,5)+_Parameters(72,0)*_Training(i,2)*_Training(i,3)+_Parameters(73,0)*_Training(i,2)*_Training(i,4)+_Parameters(74,0)*_Training(i,2)*_Training(i,5)+_Parameters(75,0)*_Training(i,3)*_Training(i,4)+_Parameters(76,0)*_Training(i,3)*_Training(i,5)+_Parameters(77,0)*_Training(i,4)*_Training(i,5)+2*_Parameters(78,0)*_Training(i,1)*_Training(i,6)+2*_Parameters(79,0)*_Training(i,2)*_Training(i,6)+2*_Parameters(80,0)*_Training(i,3)*_Training(i,6)+2*_Parameters(81,0)*_Training(i,4)*_Training(i,6)+2*_Parameters(85,0)*_Training(i,5)*_Training(i,6)+2*_Parameters(67,0)*pow(_Training(i,5),2)+3*_Parameters(83,0)*pow(_Training(i,6),2);
                    
                    break;
                case 7:
                    _Jacobian(6,0) = _Parameters(1,0)+2*_Parameters(8,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,2)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6)+_Parameters(29,0)*_Training(i,7)+3*_Parameters(36,0)*pow(_Training(i,1),2)+2*_Parameters(37,0)*_Training(i,1)*_Training(i,2)+_Parameters(38,0)*pow(_Training(i,2),2)+2*_Parameters(40,0)*_Training(i,1)*_Training(i,3)+_Parameters(42,0)*_Training(i,2)*_Training(i,3)+_Parameters(43,0)*pow(_Training(i,3),2)+2*_Parameters(46,0)*_Training(i,1)*_Training(i,4)+_Parameters(49,0)*_Training(i,2)*_Training(i,4)+_Parameters(50,0)*_Training(i,3)*_Training(i,4)+_Parameters(52,0)*pow(_Training(i,4),2)+2*_Parameters(56,0)*_Training(i,1)*_Training(i,5)+_Parameters(60,0)*_Training(i,2)*_Training(i,5)+_Parameters(61,0)*_Training(i,3)*_Training(i,5)+_Parameters(62,0)*_Training(i,4)*_Training(i,5)+_Parameters(66,0)*pow(_Training(i,5),2)+2*_Parameters(71,0)*_Training(i,1)*_Training(i,6)+_Parameters(76,0)*_Training(i,2)*_Training(i,6)+_Parameters(77,0)*_Training(i,3)*_Training(i,6)+_Parameters(78,0)*_Training(i,4)*_Training(i,6)+_Parameters(79,0)*_Training(i,5)*_Training(i,6)+_Parameters(86,0)*pow(_Training(i,6),2)+2*_Parameters(92,0)*_Training(i,1)*_Training(i,7)+_Parameters(98,0)*_Training(i,2)*_Training(i,7)+_Parameters(99,0)*_Training(i,3)*_Training(i,7)+_Parameters(100,0)*_Training(i,4)*_Training(i,7)+_Parameters(101,0)*_Training(i,5)*_Training(i,7)+_Parameters(102,0)*_Training(i,6)*_Training(i,7)+_Parameters(113,0)*pow(_Training(i,7),2);
                    
                    _Jacobian(6,1) = _Parameters(2,0)+2*_Parameters(10,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,3)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6)+_Parameters(30,0)*_Training(i,7)+3*_Parameters(39,0)*pow(_Training(i,2),2)+2*_Parameters(38,0)*_Training(i,1)*_Training(i,2)+_Parameters(37,0)*pow(_Training(i,1),2)+2*_Parameters(41,0)*_Training(i,2)*_Training(i,3)+_Parameters(42,0)*_Training(i,1)*_Training(i,3)+_Parameters(44,0)*pow(_Training(i,3),2)+2*_Parameters(47,0)*_Training(i,2)*_Training(i,4)+_Parameters(49,0)*_Training(i,1)*_Training(i,4)+_Parameters(51,0)*_Training(i,3)*_Training(i,4)+_Parameters(53,0)*pow(_Training(i,4),2)+2*_Parameters(57,0)*_Training(i,2)*_Training(i,5)+_Parameters(60,0)*_Training(i,1)*_Training(i,5)+_Parameters(63,0)*_Training(i,3)*_Training(i,5)+_Parameters(64,0)*_Training(i,4)*_Training(i,5)+_Parameters(67,0)*pow(_Training(i,5),2)+2*_Parameters(72,0)*_Training(i,2)*_Training(i,6)+_Parameters(76,0)*_Training(i,1)*_Training(i,6)+_Parameters(80,0)*_Training(i,3)*_Training(i,6)+_Parameters(81,0)*_Training(i,4)*_Training(i,6)+_Parameters(82,0)*_Training(i,5)*_Training(i,6)+_Parameters(87,0)*pow(_Training(i,6),2)+2*_Parameters(93,0)*_Training(i,2)*_Training(i,7)+_Parameters(98,0)*_Training(i,1)*_Training(i,7)+_Parameters(103,0)*_Training(i,3)*_Training(i,7)+_Parameters(104,0)*_Training(i,4)*_Training(i,7)+_Parameters(105,0)*_Training(i,5)*_Training(i,7)+_Parameters(106,0)*_Training(i,6)*_Training(i,7)+_Parameters(114,0)*pow(_Training(i,7),2);
                    
                    _Jacobian(6,2) = _Parameters(3,0)+2*_Parameters(13,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,2)+_Parameters(11,0)*_Training(i,1)+_Parameters(16,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6)+_Parameters(31,0)*_Training(i,7)+3*_Parameters(45,0)*pow(_Training(i,3),2)+2*_Parameters(44,0)*_Training(i,3)*_Training(i,2)+_Parameters(41,0)*pow(_Training(i,2),2)+2*_Parameters(43,0)*_Training(i,1)*_Training(i,3)+_Parameters(42,0)*_Training(i,1)*_Training(i,2)+_Parameters(40,0)*pow(_Training(i,1),2)+2*_Parameters(48,0)*_Training(i,3)*_Training(i,4)+_Parameters(50,0)*_Training(i,1)*_Training(i,4)+_Parameters(51,0)*_Training(i,2)*_Training(i,4)+_Parameters(54,0)*pow(_Training(i,4),2)+2*_Parameters(58,0)*_Training(i,3)*_Training(i,5)+_Parameters(61,0)*_Training(i,1)*_Training(i,5)+_Parameters(63,0)*_Training(i,2)*_Training(i,5)+_Parameters(65,0)*_Training(i,4)*_Training(i,5)+_Parameters(68,0)*pow(_Training(i,5),2)+2*_Parameters(73,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,1)*_Training(i,6)+_Parameters(80,0)*_Training(i,2)*_Training(i,6)+_Parameters(83,0)*_Training(i,4)*_Training(i,6)+_Parameters(84,0)*_Training(i,5)*_Training(i,6)+_Parameters(88,0)*pow(_Training(i,6),2)+2*_Parameters(94,0)*_Training(i,3)*_Training(i,7)+_Parameters(99,0)*_Training(i,1)*_Training(i,7)+_Parameters(103,0)*_Training(i,2)*_Training(i,7)+_Parameters(107,0)*_Training(i,4)*_Training(i,7)+_Parameters(108,0)*_Training(i,5)*_Training(i,7)+_Parameters(109,0)*_Training(i,6)*_Training(i,7)+_Parameters(115,0)*pow(_Training(i,7),2);
                    
                    
                    _Jacobian(6,3) = _Parameters(4,0)+_Parameters(14,0)*_Training(i,1)+_Parameters(15,0)*_Training(i,2)+_Parameters(16,0)*_Training(i,3)+2*_Parameters(17,0)*_Training(i,4)+_Parameters(21,0)*_Training(i,5)+_Parameters(26,0)*_Training(i,6)+_Parameters(32,0)*_Training(i,7)+_Parameters(46,0)*pow(_Training(i,1),2)+_Parameters(47,0)*pow(_Training(i,2),2)+_Parameters(48,0)*pow(_Training(i,3),2)+_Parameters(49,0)*_Training(i,1)*_Training(i,2)+_Parameters(50,0)*_Training(i,1)*_Training(i,3)+_Parameters(51,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(52,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(53,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(53,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(55,0)*pow(_Training(i,4),2)+2*_Parameters(59,0)*_Training(i,4)*_Training(i,5)+_Parameters(62,0)*_Training(i,1)*_Training(i,5)+_Parameters(64,0)*_Training(i,2)*_Training(i,5)+_Parameters(65,0)*_Training(i,3)*_Training(i,5)+_Parameters(69,0)*pow(_Training(i,5),2)+2*_Parameters(74,0)*_Training(i,4)*_Training(i,6)+_Parameters(78,0)*_Training(i,1)*_Training(i,6)+_Parameters(81,0)*_Training(i,2)*_Training(i,6)+_Parameters(83,0)*_Training(i,3)*_Training(i,6)+_Parameters(85,0)*_Training(i,5)*_Training(i,6)+_Parameters(89,0)*pow(_Training(i,6),2)+2*_Parameters(95,0)*_Training(i,1)*_Training(i,7)+_Parameters(100,0)*_Training(i,1)*_Training(i,7)+_Parameters(104,0)*_Training(i,2)*_Training(i,7)+_Parameters(107,0)*_Training(i,3)*_Training(i,7)+_Parameters(110,0)*_Training(i,5)*_Training(i,7)+_Parameters(111,0)*_Training(i,6)*_Training(i,7)+_Parameters(116,0)*pow(_Training(i,7),2);
                    
                    _Jacobian(6,4) = _Parameters(5,0)+_Parameters(18,0)*_Training(i,1)+_Parameters(19,0)*_Training(i,2)+_Parameters(20,0)*_Training(i,3)+_Parameters(21,0)*_Training(i,4)+2*_Parameters(22,0)*_Training(i,5)+_Parameters(27,0)*_Training(i,6)+_Parameters(33,0)*_Training(i,7)+_Parameters(56,0)*pow(_Training(i,1),2)+_Parameters(57,0)*pow(_Training(i,2),2)+_Parameters(58,0)*pow(_Training(i,3),2)+_Parameters(59,0)*pow(_Training(i,4),2)+_Parameters(60,0)*_Training(i,1)*_Training(i,2)+_Parameters(61,0)*_Training(i,1)*_Training(i,3)+_Parameters(62,0)*_Training(i,1)*_Training(i,4)+_Parameters(63,0)*_Training(i,2)*_Training(i,3)+_Parameters(64,0)*_Training(i,2)*_Training(i,4)+_Parameters(65,0)*_Training(i,3)*_Training(i,4)+2*_Parameters(66,0)*_Training(i,1)*_Training(i,5)+2*_Parameters(67,0)*_Training(i,2)*_Training(i,5)+2*_Parameters(68,0)*_Training(i,3)*_Training(i,5)+2*_Parameters(69,0)*_Training(i,4)*_Training(i,5)+3*_Parameters(70,0)*pow(_Training(i,5),2)+2*_Parameters(75,0)*_Training(i,5)*_Training(i,6)+_Parameters(79,0)*_Training(i,1)*_Training(i,6)+_Parameters(82,0)*_Training(i,2)*_Training(i,6)+_Parameters(84,0)*_Training(i,3)*_Training(i,6)+_Parameters(85,0)*_Training(i,4)*_Training(i,6)+_Parameters(90,0)*pow(_Training(i,6),2)+2*_Parameters(96,0)*_Training(i,5)*_Training(i,7)+_Parameters(101,0)*_Training(i,1)*_Training(i,7)+_Parameters(105,0)*_Training(i,2)*_Training(i,7)+_Parameters(108,0)*_Training(i,3)*_Training(i,7)+_Parameters(110,0)*_Training(i,4)*_Training(i,7)+_Parameters(112,0)*_Training(i,6)*_Training(i,7)+_Parameters(117,0)*pow(_Training(i,7),2);
                    
                    _Jacobian(6,5) = _Parameters(6,0)+_Parameters(23,0)*_Training(i,1)+_Parameters(24,0)*_Training(i,2)+_Parameters(25,0)*_Training(i,3)+_Parameters(26,0)*_Training(i,4)+_Parameters(27,0)*_Training(i,5)+2*_Parameters(28,0)*_Training(i,6)+_Parameters(34,0)*_Training(i,7)+_Parameters(71,0)*pow(_Training(i,1),2)+_Parameters(72,0)*pow(_Training(i,2),2)+_Parameters(73,0)*pow(_Training(i,3),2)+_Parameters(74,0)*pow(_Training(i,4),2)+_Parameters(75,0)*pow(_Training(i,5),2)+_Parameters(76,0)*_Training(i,1)*_Training(i,2)+_Parameters(77,0)*_Training(i,1)*_Training(i,3)+_Parameters(78,0)*_Training(i,1)*_Training(i,4)+_Parameters(79,0)*_Training(i,1)*_Training(i,5)+_Parameters(80,0)*_Training(i,2)*_Training(i,3)+_Parameters(81,0)*_Training(i,2)*_Training(i,4)+_Parameters(82,0)*_Training(i,2)*_Training(i,5)+_Parameters(83,0)*_Training(i,3)*_Training(i,4)+_Parameters(84,0)*_Training(i,3)*_Training(i,5)+_Parameters(85,0)*_Training(i,4)*_Training(i,5)+2*_Parameters(86,0)*_Training(i,1)*_Training(i,6)+2*_Parameters(87,0)*_Training(i,2)*_Training(i,6)+2*_Parameters(88,0)*_Training(i,3)*_Training(i,6)+2*_Parameters(89,0)*_Training(i,4)*_Training(i,6)+2*_Parameters(90,0)*_Training(i,5)*_Training(i,6)+3*_Parameters(91,0)*pow(_Training(i,6),2)+2*_Parameters(97,0)*_Training(i,6)*_Training(i,7)+_Parameters(102,0)*_Training(i,1)*_Training(i,7)+_Parameters(106,0)*_Training(i,2)*_Training(i,7)+_Parameters(109,0)*_Training(i,3)*_Training(i,7)+_Parameters(111,0)*_Training(i,4)*_Training(i,7)+_Parameters(112,0)*_Training(i,5)*_Training(i,7)+_Parameters(118,0)*pow(_Training(i,7),2);
                    
                    _Jacobian(6,6) = _Parameters(7,0)+_Parameters(29,0)*_Training(i,1)+_Parameters(30,0)*_Training(i,2)+_Parameters(31,0)*_Training(i,3)+_Parameters(32,0)*_Training(i,4)+_Parameters(33,0)*_Training(i,5)+_Parameters(34,0)*_Training(i,6)+2*_Parameters(35,0)*_Training(i,7)+_Parameters(92,0)*pow(_Training(i,1),2)+_Parameters(93,0)*pow(_Training(i,2),2)+_Parameters(94,0)*pow(_Training(i,3),2)+_Parameters(95,0)*pow(_Training(i,4),2)+_Parameters(96,0)*pow(_Training(i,5),2)+_Parameters(97,0)*pow(_Training(i,6),2)+_Parameters(98,0)*_Training(i,1)*_Training(i,2)+_Parameters(99,0)*_Training(i,1)*_Training(i,3)+_Parameters(100,0)*_Training(i,1)*_Training(i,4)+_Parameters(101,0)*_Training(i,1)*_Training(i,5)+_Parameters(102,0)*_Training(i,1)*_Training(i,6)+_Parameters(103,0)*_Training(i,2)*_Training(i,3)+_Parameters(104,0)*_Training(i,2)*_Training(i,4)+_Parameters(105,0)*_Training(i,2)*_Training(i,5)+_Parameters(106,0)*_Training(i,2)*_Training(i,6)+_Parameters(107,0)*_Training(i,3)*_Training(i,4)+_Parameters(108,0)*_Training(i,3)*_Training(i,5)+_Parameters(109,0)*_Training(i,3)*_Training(i,6)+_Parameters(110,0)*_Training(i,4)*_Training(i,5)+_Parameters(111,0)*_Training(i,4)*_Training(i,6)+_Parameters(112,0)*_Training(i,5)*_Training(i,6)+2*_Parameters(113,0)*_Training(i,1)*_Training(i,7)+2*_Parameters(114,0)*_Training(i,2)*_Training(i,7)+2*_Parameters(115,0)*_Training(i,3)*_Training(i,7)+2*_Parameters(116,0)*_Training(i,4)*_Training(i,7)+2*_Parameters(117,0)*_Training(i,5)*_Training(i,7)+2*_Parameters(118,0)*_Training(i,6)*_Training(i,7)+3*_Parameters(119,0)*pow(_Training(i,7),2);
                    
                    break;
            }
                break;
                
            case 4:
                switch (d)
            {
                case 1:
                    _Jacobian(0,0) = _Parameters(1,0)+2*_Parameters(2,0)*_Training(i,1)+3*_Parameters(3,0)*pow(_Training(i,1),2)+4*_Parameters(4,0)*pow(_Training(i,1),3);
                    break;
                case 2:
                    _Jacobian(1,0) = _Parameters(1,0)+2*_Parameters(3,0)*_Training(i,1)+_Parameters(4,0)*_Training(i,2)+3*_Parameters(6,0)*pow(_Training(i,1),2)+2*_Parameters(7,0)*_Training(i,1)*_Training(i,2)+_Parameters(8,0)*pow(_Training(i,2),2)+4*_Parameters(10,0)*pow(_Training(i,1),3)+3*_Parameters(11,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(12,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(13,0)*pow(_Training(i,2),3);
                    
                    _Jacobian(1,1) = _Parameters(2,0)+2*_Parameters(5,0)*_Training(i,5)+_Parameters(4,0)*_Training(i,1)+3*_Parameters(9,0)*pow(_Training(i,2),2)+2*_Parameters(8,0)*_Training(i,1)*_Training(i,2)+_Parameters(7,0)*pow(_Training(i,1),2)+4*_Parameters(14,0)*pow(_Training(i,2),3)+3*_Parameters(13,0)*pow(_Training(i,2),2)*_Training(i,1)+2*_Parameters(12,0)*_Training(i,2)*pow(_Training(i,1),2)+_Parameters(11,0)*pow(_Training(i,1),3);
                    break;
                case 3:
                    
                    _Jacobian(2,0) = _Parameters(1,0)+2*_Parameters(4,0)*_Training(i,1)+_Parameters(5,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,3)+3*_Parameters(10,0)*pow(_Training(i,1),2)+2*_Parameters(11,0)*_Training(i,1)*_Training(i,2)+_Parameters(12,0)*pow(_Training(i,2),2)+2*_Parameters(14,0)*_Training(i,1)*_Training(i,3)+_Parameters(16,0)*_Training(i,2)*_Training(i,3)+_Parameters(17,0)*pow(_Training(i,3),2)+4*_Parameters(20,0)*pow(_Training(i,1),3)+3*_Parameters(21,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(22,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(23,0)*pow(_Training(i,2),3)+3*_Parameters(25,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(27,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(28,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(29,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(30,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(32,0)*pow(_Training(i,3),3);
                    
                    
                    _Jacobian(2,1) = _Parameters(2,0)+2*_Parameters(6,0)*_Training(i,2)+_Parameters(5,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,3)+3*_Parameters(13,0)*pow(_Training(i,2),2)+2*_Parameters(12,0)*_Training(i,1)*_Training(i,2)+_Parameters(11,0)*pow(_Training(i,1),2)+2*_Parameters(15,0)*_Training(i,2)*_Training(i,3)+_Parameters(16,0)*_Training(i,1)*_Training(i,3)+_Parameters(18,0)*pow(_Training(i,3),2)+3*_Parameters(21,0)*pow(_Training(i,1),3)+2*_Parameters(22,0)*pow(_Training(i,1),2)*_Training(i,2)+3*_Parameters(23,0)*_Training(i,1)*pow(_Training(i,2),2)+4*_Parameters(24,0)*pow(_Training(i,2),3)+3*_Parameters(26,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(27,0)*_Training(i,1)*_Training(i,3)+2*_Parameters(28,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(30,0)*_Training(i,1)*pow(_Training(i,3),2)+2*_Parameters(31,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(33,0)*pow(_Training(i,3),3);
                    
                    
                    _Jacobian(2,2) = _Parameters(3,0)+2*_Parameters(9,0)*_Training(i,3)+_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2)+3*_Parameters(19,0)*pow(_Training(i,3),2)+2*_Parameters(17,0)*_Training(i,1)*_Training(i,3)+_Parameters(14,0)*pow(_Training(i,1),2)+2*_Parameters(18,0)*_Training(i,2)*_Training(i,3)+_Parameters(16,0)*_Training(i,1)*_Training(i,2)+_Parameters(15,0)*pow(_Training(i,2),2)+_Parameters(25,0)*pow(_Training(i,1),3)+_Parameters(26,0)*pow(_Training(i,2),3)+_Parameters(27,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(28,0)*_Training(i,1)*pow(_Training(i,2),2)+2*_Parameters(29,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(30,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+2*_Parameters(31,0)*pow(_Training(i,2),2)*_Training(i,3)+3*_Parameters(32,0)*_Training(i,1)*pow(_Training(i,3),2)+3*_Parameters(34,0)*_Training(i,2)*pow(_Training(i,3),2)+4*_Parameters(33,0)*pow(_Training(i,3),3);
                    
                    break;
                    
                case 4:
                    
                    _Jacobian(3,0) = _Parameters(1,0)+2*_Parameters(5,0)*_Training(i,1)+_Parameters(6,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,3)+_Parameters(11,0)*_Training(i,4)+3*_Parameters(15,0)*pow(_Training(i,1),2)+2*_Parameters(16,0)*_Training(i,1)*_Training(i,2)+_Parameters(17,0)*pow(_Training(i,2),2)+2*_Parameters(19,0)*_Training(i,1)*_Training(i,3)+_Parameters(21,0)*_Training(i,2)*_Training(i,3)+_Parameters(22,0)*pow(_Training(i,3),2)+2*_Parameters(25,0)*_Training(i,1)*_Training(i,4)+_Parameters(28,0)*_Training(i,2)*_Training(i,4)+_Parameters(29,0)*_Training(i,3)*_Training(i,4)+_Parameters(31,0)*pow(_Training(i,4),2)+4*_Parameters(35,0)*pow(_Training(i,1),3)+3*_Parameters(36,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(37,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(38,0)*pow(_Training(i,2),3)+3*_Parameters(40,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(42,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(43,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(44,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(45,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(47,0)*pow(_Training(i,3),3)+3*_Parameters(50,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(51,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(52,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+2*_Parameters(53,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(54,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(55,0)*pow(_Training(i,3),2)*_Training(i,4)+2*_Parameters(60,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(61,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(62,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(66,0)*pow(_Training(i,4),3);
                    
                    _Jacobian(3,1) = _Parameters(2,0)+2*_Parameters(7,0)*_Training(i,2)+_Parameters(6,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4)+3*_Parameters(18,0)*pow(_Training(i,2),2)+2*_Parameters(17,0)*_Training(i,1)*_Training(i,2)+_Parameters(16,0)*pow(_Training(i,1),2)+2*_Parameters(20,0)*_Training(i,2)*_Training(i,3)+_Parameters(21,0)*_Training(i,1)*_Training(i,3)+_Parameters(23,0)*pow(_Training(i,3),2)+2*_Parameters(26,0)*_Training(i,2)*_Training(i,4)+_Parameters(28,0)*_Training(i,1)*_Training(i,4)+_Parameters(30,0)*_Training(i,3)*_Training(i,4)+_Parameters(32,0)*pow(_Training(i,4),2)+3*_Parameters(36,0)*pow(_Training(i,1),3)+2*_Parameters(37,0)*pow(_Training(i,1),2)*_Training(i,2)+3*_Parameters(38,0)*_Training(i,1)*pow(_Training(i,2),2)+4*_Parameters(39,0)*pow(_Training(i,2),3)+3*_Parameters(41,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(42,0)*_Training(i,1)*_Training(i,3)+2*_Parameters(43,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(45,0)*_Training(i,1)*pow(_Training(i,3),2)+2*_Parameters(46,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(48,0)*pow(_Training(i,3),3)+_Parameters(51,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(53,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(54,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+3*_Parameters(56,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(57,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(58,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(61,0)*_Training(i,1)*pow(_Training(i,4),2)+2*_Parameters(63,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(64,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(67,0)*pow(_Training(i,4),3);
                    
                    
                    
                    _Jacobian(3,2) = _Parameters(3,0)+2*_Parameters(10,0)*_Training(i,3)+_Parameters(9,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,1)+_Parameters(13,0)*_Training(i,4)+3*_Parameters(24,0)*pow(_Training(i,3),2)+2*_Parameters(22,0)*_Training(i,1)*_Training(i,3)+_Parameters(19,0)*pow(_Training(i,1),2)+2*_Parameters(23,0)*_Training(i,2)*_Training(i,3)+_Parameters(21,0)*_Training(i,1)*_Training(i,2)+_Parameters(20,0)*pow(_Training(i,2),2)+2*_Parameters(27,0)*_Training(i,3)*_Training(i,4)+_Parameters(29,0)*_Training(i,1)*_Training(i,4)+_Parameters(30,0)*_Training(i,2)*_Training(i,4)+_Parameters(33,0)*pow(_Training(i,4),2)+_Parameters(40,0)*pow(_Training(i,1),3)+_Parameters(41,0)*pow(_Training(i,2),3)+_Parameters(42,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(43,0)*_Training(i,1)*pow(_Training(i,2),2)+2*_Parameters(44,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(45,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+2*_Parameters(46,0)*pow(_Training(i,2),2)*_Training(i,3)+3*_Parameters(47,0)*_Training(i,1)*pow(_Training(i,3),2)+3*_Parameters(48,0)*_Training(i,2)*pow(_Training(i,3),2)+4*_Parameters(49,0)*pow(_Training(i,3),3)+_Parameters(52,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(54,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(55,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(57,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(58,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+3*_Parameters(59,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(62,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(64,0)*_Training(i,2)*pow(_Training(i,4),2)+2*_Parameters(65,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(68,0)*pow(_Training(i,4),3);
                    
                    
                    
                    _Jacobian(3,3) = _Parameters(4,0)+_Parameters(11,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,2)+_Parameters(13,0)*_Training(i,3)+2*_Parameters(14,0)*_Training(i,4)+_Parameters(25,0)*pow(_Training(i,1),2)+_Parameters(26,0)*pow(_Training(i,2),2)+_Parameters(27,0)*pow(_Training(i,3),2)+_Parameters(28,0)*_Training(i,1)*_Training(i,2)+_Parameters(29,0)*_Training(i,1)*_Training(i,3)+_Parameters(30,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(31,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(32,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(33,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(34,0)*pow(_Training(i,4),2)+_Parameters(50,0)*pow(_Training(i,1),3)+_Parameters(51,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(52,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(53,0)*pow(_Training(i,2),2)*_Training(i,1)+_Parameters(54,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(55,0)*pow(_Training(i,3),2)*_Training(i,1)+_Parameters(56,0)*pow(_Training(i,3),3)+_Parameters(57,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(58,0)*pow(_Training(i,3),2)*_Training(i,2)+_Parameters(59,0)*pow(_Training(i,3),3)+2*_Parameters(60,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(61,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(62,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(63,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(64,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+2*_Parameters(65,0)*pow(_Training(i,3),2)*_Training(i,4)+3*_Parameters(66,0)*pow(_Training(i,4),2)*_Training(i,1)+3*_Parameters(67,0)*pow(_Training(i,4),2)*_Training(i,2)+3*_Parameters(68,0)*pow(_Training(i,4),2)*_Training(i,3)+4*_Parameters(69,0)*pow(_Training(i,4),3);
                    
                    break;
                    
                case 5:
                    
                    _Jacobian(4,0) = _Parameters(1,0)+2*_Parameters(6,0)*_Training(i,1)+_Parameters(7,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,4)+_Parameters(16,0)*_Training(i,5)+3*_Parameters(21,0)*pow(_Training(i,1),2)+2*_Parameters(22,0)*_Training(i,1)*_Training(i,2)+_Parameters(23,0)*pow(_Training(i,2),2)+2*_Parameters(25,0)*_Training(i,1)*_Training(i,3)+_Parameters(27,0)*_Training(i,2)*_Training(i,3)+_Parameters(28,0)*pow(_Training(i,3),2)+2*_Parameters(31,0)*_Training(i,1)*_Training(i,4)+_Parameters(34,0)*_Training(i,2)*_Training(i,4)+_Parameters(35,0)*_Training(i,3)*_Training(i,4)+_Parameters(37,0)*pow(_Training(i,4),2)+2*_Parameters(41,0)*_Training(i,1)*_Training(i,5)+_Parameters(45,0)*_Training(i,2)*_Training(i,5)+_Parameters(46,0)*_Training(i,3)*_Training(i,5)+_Parameters(47,0)*_Training(i,4)*_Training(i,5)+_Parameters(51,0)*pow(_Training(i,5),2)+4*_Parameters(56,0)*pow(_Training(i,1),3)+3*_Parameters(57,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(58,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(59,0)*pow(_Training(i,2),3)+3*_Parameters(61,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(63,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(64,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(65,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(66,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(68,0)*pow(_Training(i,3),3)+3*_Parameters(71,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(72,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(73,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+2*_Parameters(74,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(75,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(76,0)*pow(_Training(i,3),2)*_Training(i,4)+2*_Parameters(81,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(82,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(83,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(87,0)*pow(_Training(i,4),3)+3*_Parameters(91,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(92,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(93,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(94,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(95,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(96,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(97,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(98,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(99,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(100,0)*pow(_Training(i,4),2)*_Training(i,5)+2*_Parameters(111,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(112,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(113,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(114,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(121,0)*pow(_Training(i,5),3);
                    
                    _Jacobian(4,1) = _Parameters(2,0)+2*_Parameters(8,0)*_Training(i,2)+_Parameters(7,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5)+3*_Parameters(24,0)*pow(_Training(i,2),2)+2*_Parameters(23,0)*_Training(i,1)*_Training(i,2)+_Parameters(22,0)*pow(_Training(i,1),2)+2*_Parameters(26,0)*_Training(i,2)*_Training(i,3)+_Parameters(27,0)*_Training(i,1)*_Training(i,3)+_Parameters(29,0)*pow(_Training(i,3),2)+2*_Parameters(32,0)*_Training(i,2)*_Training(i,4)+_Parameters(34,0)*_Training(i,1)*_Training(i,4)+_Parameters(36,0)*_Training(i,3)*_Training(i,4)+_Parameters(38,0)*pow(_Training(i,4),2)+2*_Parameters(42,0)*_Training(i,2)*_Training(i,5)+_Parameters(45,0)*_Training(i,1)*_Training(i,5)+_Parameters(48,0)*_Training(i,3)*_Training(i,5)+_Parameters(49,0)*_Training(i,4)*_Training(i,5)+_Parameters(52,0)*pow(_Training(i,5),2)+3*_Parameters(57,0)*pow(_Training(i,1),3)+2*_Parameters(58,0)*pow(_Training(i,1),2)*_Training(i,2)+3*_Parameters(59,0)*_Training(i,1)*pow(_Training(i,2),2)+4*_Parameters(60,0)*pow(_Training(i,2),3)+3*_Parameters(62,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(63,0)*_Training(i,1)*_Training(i,3)+2*_Parameters(64,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(66,0)*_Training(i,1)*pow(_Training(i,3),2)+2*_Parameters(67,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(69,0)*pow(_Training(i,3),3)+_Parameters(72,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(74,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(75,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+3*_Parameters(77,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(78,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(79,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(82,0)*_Training(i,1)*pow(_Training(i,4),2)+2*_Parameters(84,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(85,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(88,0)*pow(_Training(i,4),3)+_Parameters(92,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(99,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(95,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(96,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(102,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(103,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(104,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+3*_Parameters(101,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(105,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(106,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(114,0)*_Training(i,1)*pow(_Training(i,5),2)+2*_Parameters(115,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(116,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(117,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(122,0)*pow(_Training(i,5),3);
                    
                    
                    _Jacobian(4,2) = _Parameters(3,0)+2*_Parameters(11,0)*_Training(i,3)+_Parameters(9,0)*_Training(i,1)+_Parameters(10,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+3*_Parameters(30,0)*pow(_Training(i,3),2)+2*_Parameters(28,0)*_Training(i,1)*_Training(i,3)+_Parameters(25,0)*pow(_Training(i,1),2)+2*_Parameters(29,0)*_Training(i,2)*_Training(i,3)+_Parameters(27,0)*_Training(i,1)*_Training(i,2)+_Parameters(26,0)*pow(_Training(i,2),2)+2*_Parameters(33,0)*_Training(i,3)*_Training(i,4)+_Parameters(35,0)*_Training(i,1)*_Training(i,4)+_Parameters(36,0)*_Training(i,2)*_Training(i,4)+_Parameters(39,0)*pow(_Training(i,4),2)+2*_Parameters(43,0)*_Training(i,3)*_Training(i,5)+_Parameters(46,0)*_Training(i,1)*_Training(i,5)+_Parameters(48,0)*_Training(i,2)*_Training(i,5)+_Parameters(50,0)*_Training(i,4)*_Training(i,5)+_Parameters(53,0)*pow(_Training(i,5),2)+_Parameters(61,0)*pow(_Training(i,1),3)+_Parameters(62,0)*pow(_Training(i,2),3)+_Parameters(63,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(64,0)*_Training(i,1)*pow(_Training(i,2),2)+2*_Parameters(65,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(66,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+2*_Parameters(67,0)*pow(_Training(i,2),2)*_Training(i,3)+3*_Parameters(68,0)*_Training(i,1)*pow(_Training(i,3),2)+3*_Parameters(69,0)*_Training(i,2)*pow(_Training(i,3),2)+4*_Parameters(70,0)*pow(_Training(i,3),3)+_Parameters(73,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(75,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(76,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(78,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(79,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+3*_Parameters(80,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(83,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(85,0)*_Training(i,2)*pow(_Training(i,4),2)+2*_Parameters(86,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(89,0)*pow(_Training(i,4),3)+_Parameters(93,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(95,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(99,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(97,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(105,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(104,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(108,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(102,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(107,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(109,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(113,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(116,0)*_Training(i,2)*pow(_Training(i,5),2)+2*_Parameters(118,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(119,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(123,0)*pow(_Training(i,5),3);
                    
                    _Jacobian(4,3) = _Parameters(4,0)+2*_Parameters(15,0)*_Training(i,4)+_Parameters(12,0)*_Training(i,1)+_Parameters(13,0)*_Training(i,2)+_Parameters(14,0)*_Training(i,3)+_Parameters(19,0)*_Training(i,5)+3*_Parameters(40,0)*pow(_Training(i,4),2)+2*_Parameters(37,0)*_Training(i,1)*_Training(i,4)+_Parameters(31,0)*pow(_Training(i,1),2)+2*_Parameters(38,0)*_Training(i,2)*_Training(i,4)+_Parameters(34,0)*_Training(i,1)*_Training(i,2)+_Parameters(32,0)*pow(_Training(i,2),2)+2*_Parameters(39,0)*_Training(i,3)*_Training(i,4)+_Parameters(35,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*_Training(i,2)*_Training(i,3)+_Parameters(33,0)*pow(_Training(i,3),2)+2*_Parameters(44,0)*_Training(i,4)*_Training(i,5)+_Parameters(47,0)*_Training(i,1)*_Training(i,5)+_Parameters(49,0)*_Training(i,2)*_Training(i,5)+_Parameters(50,0)*_Training(i,3)*_Training(i,5)+_Parameters(54,0)*pow(_Training(i,5),2)+_Parameters(71,0)*pow(_Training(i,1),3)+_Parameters(72,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(73,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(74,0)*pow(_Training(i,2),2)*_Training(i,1)+_Parameters(75,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(76,0)*pow(_Training(i,3),2)*_Training(i,1)+_Parameters(77,0)*pow(_Training(i,3),3)+_Parameters(78,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(79,0)*pow(_Training(i,3),2)*_Training(i,2)+_Parameters(80,0)*pow(_Training(i,3),3)+2*_Parameters(81,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(82,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(83,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(84,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(85,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+2*_Parameters(86,0)*pow(_Training(i,3),2)*_Training(i,4)+3*_Parameters(87,0)*pow(_Training(i,4),2)*_Training(i,1)+3*_Parameters(88,0)*pow(_Training(i,4),2)*_Training(i,2)+3*_Parameters(89,0)*pow(_Training(i,4),2)*_Training(i,3)+4*_Parameters(90,0)*pow(_Training(i,4),3)+_Parameters(94,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(96,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(97,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(100,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(104,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(106,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(109,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(103,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(108,0)*pow(_Training(i,3),2)*_Training(i,5)+3*_Parameters(110,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(114,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(117,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(119,0)*_Training(i,3)*pow(_Training(i,5),2)+2*_Parameters(120,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(124,0)*pow(_Training(i,5),3);
                    
                    
                    _Jacobian(4,4) = _Parameters(5,0)+2*_Parameters(20,0)*_Training(i,5)+_Parameters(16,0)*_Training(i,1)+_Parameters(17,0)*_Training(i,2)+_Parameters(18,0)*_Training(i,3)+_Parameters(19,0)*_Training(i,4)+3*_Parameters(55,0)*pow(_Training(i,5),2)+2*_Parameters(51,0)*_Training(i,1)*_Training(i,5)+_Parameters(41,0)*pow(_Training(i,1),2)+2*_Parameters(52,0)*_Training(i,2)*_Training(i,5)+_Parameters(45,0)*_Training(i,1)*_Training(i,2)+_Parameters(42,0)*pow(_Training(i,2),2)+2*_Parameters(53,0)*_Training(i,3)*_Training(i,5)+_Parameters(46,0)*_Training(i,1)*_Training(i,3)+_Parameters(48,0)*_Training(i,2)*_Training(i,3)+_Parameters(43,0)*pow(_Training(i,3),2)+2*_Parameters(54,0)*_Training(i,4)*_Training(i,5)+_Parameters(47,0)*_Training(i,1)*_Training(i,4)+_Parameters(49,0)*_Training(i,2)*_Training(i,4)+_Parameters(50,0)*_Training(i,3)*_Training(i,4)+_Parameters(44,0)*pow(_Training(i,4),2)+_Parameters(91,0)*pow(_Training(i,1),3)+_Parameters(92,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(98,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(103,0)*pow(_Training(i,2),3)+_Parameters(93,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(95,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(102,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(99,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(105,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(107,0)*pow(_Training(i,3),3)+_Parameters(94,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(96,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(97,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(103,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(104,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(108,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(100,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(106,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(109,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(110,0)*pow(_Training(i,4),3)+2*_Parameters(111,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(112,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(113,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(114,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(116,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(117,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(119,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+2*_Parameters(115,0)*pow(_Training(i,2),2)*_Training(i,5)+2*_Parameters(118,0)*pow(_Training(i,3),2)*_Training(i,5)+2*_Parameters(120,0)*pow(_Training(i,4),2)*_Training(i,5)+3*_Parameters(121,0)*_Training(i,1)*pow(_Training(i,5),2)+3*_Parameters(122,0)*_Training(i,2)*pow(_Training(i,5),2)+3*_Parameters(123,0)*_Training(i,3)*pow(_Training(i,5),2)+3*_Parameters(124,0)*_Training(i,4)*pow(_Training(i,5),2)+4*_Parameters(125,0)*pow(_Training(i,5),3);
                    
                    
                    
                    break;
                    
                case 6:
                    
                    _Jacobian(5,0) = _Parameters(1,0)+2*_Parameters(7,0)*_Training(i,1)+_Parameters(8,0)*_Training(i,2)+_Parameters(10,0)*_Training(i,3)+_Parameters(13,0)*_Training(i,4)+_Parameters(17,0)*_Training(i,5)+_Parameters(22,0)*_Training(i,6)+3*_Parameters(28,0)*pow(_Training(i,1),2)+2*_Parameters(29,0)*_Training(i,1)*_Training(i,2)+_Parameters(30,0)*pow(_Training(i,2),2)+2*_Parameters(32,0)*_Training(i,1)*_Training(i,2)+_Parameters(34,0)*_Training(i,2)*_Training(i,3)+_Parameters(35,0)*pow(_Training(i,3),2)+2*_Parameters(38,0)*_Training(i,1)*_Training(i,4)+_Parameters(41,0)*_Training(i,2)*_Training(i,4)+_Parameters(42,0)*_Training(i,3)*_Training(i,4)+_Parameters(44,0)*pow(_Training(i,4),2)+2*_Parameters(48,0)*_Training(i,1)*_Training(i,5)+_Parameters(52,0)*_Training(i,2)*_Training(i,5)+_Parameters(53,0)*_Training(i,3)*_Training(i,5)+_Parameters(54,0)*_Training(i,4)*_Training(i,5)+_Parameters(58,0)*pow(_Training(i,5),2)+2*_Parameters(63,0)*_Training(i,1)*_Training(i,6)+_Parameters(68,0)*_Training(i,2)*_Training(i,6)+_Parameters(69,0)*_Training(i,3)*_Training(i,6)+_Parameters(70,0)*_Training(i,4)*_Training(i,6)+_Parameters(71,0)*_Training(i,5)*_Training(i,6)+_Parameters(78,0)*pow(_Training(i,6),2)+4*_Parameters(84,0)*pow(_Training(i,1),3)+3*_Parameters(85,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(86,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(87,0)*pow(_Training(i,2),3)+3*_Parameters(89,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(91,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(92,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(93,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(94,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(96,0)*pow(_Training(i,3),3)+3*_Parameters(99,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(100,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(101,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+2*_Parameters(102,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(103,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(104,0)*pow(_Training(i,3),2)*_Training(i,4)+2*_Parameters(109,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(110,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(111,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(115,0)*pow(_Training(i,4),3)+3*_Parameters(119,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(120,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(121,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(122,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(123,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(124,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(125,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(126,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(127,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(128,0)*pow(_Training(i,4),2)*_Training(i,5)+2*_Parameters(139,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(140,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(141,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(142,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(149,0)*pow(_Training(i,5),3)+3*_Parameters(154,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(155,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(156,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(157,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(158,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(159,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(160,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(161,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(162,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(163,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(164,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(165,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(166,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(167,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(168,0)*pow(_Training(i,5),2)*_Training(i,6)+2*_Parameters(189,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(190,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(191,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(192,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(193,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(203,0)*pow(_Training(i,6),3);
                    
                    _Jacobian(5,1) = _Parameters(2,0)+2*_Parameters(9,0)*_Training(i,2)+_Parameters(8,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6)+3*_Parameters(31,0)*pow(_Training(i,2),2)+2*_Parameters(30,0)*_Training(i,1)*_Training(i,2)+_Parameters(29,0)*pow(_Training(i,1),2)+2*_Parameters(33,0)*_Training(i,2)*_Training(i,3)+_Parameters(34,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*pow(_Training(i,3),2)+2*_Parameters(39,0)*_Training(i,2)*_Training(i,4)+_Parameters(41,0)*_Training(i,1)*_Training(i,4)+_Parameters(43,0)*_Training(i,3)*_Training(i,4)+_Parameters(45,0)*pow(_Training(i,4),2)+2*_Parameters(49,0)*_Training(i,2)*_Training(i,5)+_Parameters(52,0)*_Training(i,1)*_Training(i,5)+_Parameters(55,0)*_Training(i,3)*_Training(i,5)+_Parameters(56,0)*_Training(i,4)*_Training(i,5)+_Parameters(59,0)*pow(_Training(i,5),2)+2*_Parameters(64,0)*_Training(i,2)*_Training(i,6)+_Parameters(68,0)*_Training(i,1)*_Training(i,6)+_Parameters(72,0)*_Training(i,3)*_Training(i,6)+_Parameters(73,0)*_Training(i,4)*_Training(i,6)+_Parameters(74,0)*_Training(i,5)*_Training(i,6)+_Parameters(79,0)*pow(_Training(i,6),2)+3*_Parameters(85,0)*pow(_Training(i,1),3)+2*_Parameters(86,0)*pow(_Training(i,1),2)*_Training(i,2)+3*_Parameters(87,0)*_Training(i,1)*pow(_Training(i,2),2)+4*_Parameters(88,0)*pow(_Training(i,2),3)+3*_Parameters(90,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(91,0)*_Training(i,1)*_Training(i,3)+2*_Parameters(92,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(94,0)*_Training(i,1)*pow(_Training(i,3),2)+2*_Parameters(95,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(97,0)*pow(_Training(i,3),3)+_Parameters(100,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(102,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(103,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+3*_Parameters(105,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(106,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(107,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(110,0)*_Training(i,1)*pow(_Training(i,4),2)+2*_Parameters(112,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(113,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(116,0)*pow(_Training(i,4),3)+_Parameters(120,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(127,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(123,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(124,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(130,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(131,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(132,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+3*_Parameters(129,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(133,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(134,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(142,0)*_Training(i,1)*pow(_Training(i,5),2)+2*_Parameters(143,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(144,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(145,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(150,0)*pow(_Training(i,5),3)+_Parameters(155,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(165,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(159,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(160,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(161,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(170,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(171,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(172,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(173,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(174,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(175,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+3*_Parameters(169,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(176,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(177,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(178,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(190,0)*_Training(i,1)*pow(_Training(i,6),2)+2*_Parameters(194,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(195,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(196,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(197,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(204,0)*pow(_Training(i,6),3);
                    
                    _Jacobian(5,2) = _Parameters(3,0)+_Parameters(10,0)*_Training(i,1)+_Parameters(11,0)*_Training(i,2)+2*_Parameters(12,0)*_Training(i,3)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6)+_Parameters(32,0)*pow(_Training(i,1),2)+_Parameters(33,0)*pow(_Training(i,2),2)+_Parameters(34,0)*_Training(i,1)*_Training(i,2)+2*_Parameters(35,0)*_Training(i,1)*_Training(i,3)+_Parameters(36,0)*_Training(i,2)*_Training(i,3)+3*_Parameters(37,0)*pow(_Training(i,3),2)+2*_Parameters(40,0)*_Training(i,3)*_Training(i,4)+_Parameters(42,0)*_Training(i,1)*_Training(i,4)+_Parameters(43,0)*_Training(i,2)*_Training(i,4)+_Parameters(46,0)*pow(_Training(i,4),2)+2*_Parameters(50,0)*_Training(i,3)*_Training(i,5)+_Parameters(53,0)*_Training(i,1)*_Training(i,5)+_Parameters(55,0)*_Training(i,2)*_Training(i,5)+_Parameters(57,0)*_Training(i,4)*_Training(i,5)+_Parameters(60,0)*pow(_Training(i,5),2)+2*_Parameters(65,0)*_Training(i,3)*_Training(i,6)+_Parameters(69,0)*_Training(i,1)*_Training(i,6)+_Parameters(72,0)*_Training(i,2)*_Training(i,6)+_Parameters(75,0)*_Training(i,4)*_Training(i,6)+_Parameters(76,0)*_Training(i,5)*_Training(i,6)+_Parameters(80,0)*pow(_Training(i,6),2)+_Parameters(89,0)*pow(_Training(i,1),3)+_Parameters(90,0)*pow(_Training(i,2),3)+_Parameters(91,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(92,0)*_Training(i,1)*pow(_Training(i,2),2)+2*_Parameters(93,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(94,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+2*_Parameters(95,0)*pow(_Training(i,2),2)*_Training(i,3)+3*_Parameters(96,0)*_Training(i,1)*pow(_Training(i,3),2)+3*_Parameters(97,0)*_Training(i,2)*pow(_Training(i,3),2)+4*_Parameters(98,0)*pow(_Training(i,3),3)+_Parameters(101,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(103,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(104,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(106,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(107,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+3*_Parameters(108,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(111,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(113,0)*_Training(i,2)*pow(_Training(i,4),2)+2*_Parameters(114,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(117,0)*pow(_Training(i,4),3)+_Parameters(121,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(123,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(127,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(125,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(133,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(132,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(136,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(130,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(135,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(137,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(141,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(144,0)*_Training(i,2)*pow(_Training(i,5),2)+2*_Parameters(146,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(147,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(150,0)*pow(_Training(i,5),3)+_Parameters(156,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(159,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(166,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(162,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(163,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(176,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(173,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(174,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(180,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(181,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(182,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(170,0)*pow(_Training(i,2),2)*_Training(i,6)+3*_Parameters(179,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(183,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(184,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(191,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(195,0)*_Training(i,2)*pow(_Training(i,6),2)+2*_Parameters(198,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(200,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(201,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(207,0)*pow(_Training(i,6),3);
                    
                    _Jacobian(5,3) = _Parameters(4,0)+_Parameters(13,0)*_Training(i,1)+_Parameters(14,0)*_Training(i,2)+_Parameters(15,0)*_Training(i,3)+2*_Parameters(16,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6)+_Parameters(38,0)*pow(_Training(i,1),2)+_Parameters(39,0)*pow(_Training(i,2),2)+_Parameters(40,0)*pow(_Training(i,3),2)+_Parameters(41,0)*_Training(i,1)*_Training(i,2)+_Parameters(42,0)*_Training(i,1)*_Training(i,3)+_Parameters(43,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(44,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(45,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(46,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(47,0)*pow(_Training(i,4),2)+2*_Parameters(51,0)*_Training(i,4)*_Training(i,5)+_Parameters(54,0)*_Training(i,1)*_Training(i,5)+_Parameters(56,0)*_Training(i,2)*_Training(i,5)+_Parameters(57,0)*_Training(i,3)*_Training(i,5)+_Parameters(61,0)*pow(_Training(i,5),2)+2*_Parameters(66,0)*_Training(i,4)*_Training(i,6)+_Parameters(70,0)*_Training(i,1)*_Training(i,6)+_Parameters(73,0)*_Training(i,2)*_Training(i,6)+_Parameters(75,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,5)*_Training(i,6)+_Parameters(81,0)*pow(_Training(i,6),2)+_Parameters(99,0)*pow(_Training(i,1),3)+_Parameters(100,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(101,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(102,0)*pow(_Training(i,2),2)*_Training(i,1)+_Parameters(103,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(104,0)*pow(_Training(i,3),2)*_Training(i,1)+_Parameters(105,0)*pow(_Training(i,3),3)+_Parameters(106,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(107,0)*pow(_Training(i,3),2)*_Training(i,2)+_Parameters(108,0)*pow(_Training(i,3),3)+2*_Parameters(109,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(110,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(111,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(112,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(113,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+2*_Parameters(114,0)*pow(_Training(i,3),2)*_Training(i,4)+3*_Parameters(115,0)*pow(_Training(i,4),2)*_Training(i,1)+3*_Parameters(116,0)*pow(_Training(i,4),2)*_Training(i,2)+3*_Parameters(117,0)*pow(_Training(i,4),2)*_Training(i,3)+4*_Parameters(118,0)*pow(_Training(i,4),3)+_Parameters(122,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(124,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(125,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(128,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(132,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(134,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(137,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(131,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(136,0)*pow(_Training(i,3),2)*_Training(i,5)+3*_Parameters(138,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(142,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(145,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(147,0)*_Training(i,3)*pow(_Training(i,5),2)+2*_Parameters(148,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(152,0)*pow(_Training(i,5),3)+_Parameters(156,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(160,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(162,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(167,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(164,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(173,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(177,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(175,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(183,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(182,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(186,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(172,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(180,0)*pow(_Training(i,3),2)*_Training(i,6)+3*_Parameters(185,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(188,0)*pow(_Training(i,5),2)*_Training(i,6)+2*_Parameters(192,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(196,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(199,0)*_Training(i,3)*pow(_Training(i,6),2)+2*_Parameters(201,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(202,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(207,0)*pow(_Training(i,6),3);
                    
                    _Jacobian(5,4) = _Parameters(5,0)+_Parameters(17,0)*_Training(i,1)+_Parameters(18,0)*_Training(i,2)+_Parameters(19,0)*_Training(i,3)+2*_Parameters(21,0)*_Training(i,5)+_Parameters(20,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,6)+_Parameters(48,0)*pow(_Training(i,1),2)+_Parameters(49,0)*pow(_Training(i,2),2)+_Parameters(50,0)*pow(_Training(i,3),2)+_Parameters(51,0)*pow(_Training(i,4),2)+_Parameters(52,0)*_Training(i,1)*_Training(i,2)+_Parameters(53,0)*_Training(i,1)*_Training(i,3)+_Parameters(54,0)*_Training(i,1)*_Training(i,4)+_Parameters(55,0)*_Training(i,2)*_Training(i,3)+_Parameters(56,0)*_Training(i,2)*_Training(i,4)+_Parameters(57,0)*_Training(i,3)*_Training(i,4)+2*_Parameters(58,0)*_Training(i,1)*_Training(i,5)+2*_Parameters(59,0)*_Training(i,2)*_Training(i,5)+2*_Parameters(60,0)*_Training(i,3)*_Training(i,5)+2*_Parameters(61,0)*_Training(i,4)*_Training(i,5)+3*_Parameters(62,0)*pow(_Training(i,5),2)+2*_Parameters(67,0)*_Training(i,5)*_Training(i,6)+_Parameters(71,0)*_Training(i,1)*_Training(i,6)+_Parameters(74,0)*_Training(i,2)*_Training(i,6)+_Parameters(76,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,4)*_Training(i,6)+_Parameters(82,0)*pow(_Training(i,6),2)+_Parameters(119,0)*pow(_Training(i,1),3)+_Parameters(120,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(126,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(131,0)*pow(_Training(i,2),3)+_Parameters(121,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(123,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(130,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(127,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(133,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(135,0)*pow(_Training(i,3),3)+_Parameters(122,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(124,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(125,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(131,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(132,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(136,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(100,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(106,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(137,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(138,0)*pow(_Training(i,4),3)+2*_Parameters(139,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(140,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(141,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(142,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(144,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(145,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(147,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+2*_Parameters(143,0)*pow(_Training(i,2),2)*_Training(i,5)+2*_Parameters(146,0)*pow(_Training(i,3),2)*_Training(i,5)+2*_Parameters(148,0)*pow(_Training(i,4),2)*_Training(i,5)+3*_Parameters(149,0)*_Training(i,1)*pow(_Training(i,5),2)+3*_Parameters(150,0)*_Training(i,2)*pow(_Training(i,5),2)+3*_Parameters(151,0)*_Training(i,3)*pow(_Training(i,5),2)+3*_Parameters(152,0)*_Training(i,4)*pow(_Training(i,5),2)+4*_Parameters(153,0)*pow(_Training(i,5),3)+_Parameters(158,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(161,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(163,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(164,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(168,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(174,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(175,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(178,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(182,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(184,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(187,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(172,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(181,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(186,0)*pow(_Training(i,4),2)*_Training(i,6)+2*_Parameters(188,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(193,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(197,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(200,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(202,0)*_Training(i,4)*pow(_Training(i,6),2)+2*_Parameters(203,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(208,0)*pow(_Training(i,6),3);
                    
                    _Jacobian(5,5) = _Parameters(6,0)+_Parameters(22,0)*_Training(i,1)+_Parameters(23,0)*_Training(i,2)+_Parameters(24,0)*_Training(i,3)+2*_Parameters(27,0)*_Training(i,6)+_Parameters(25,0)*_Training(i,4)+_Parameters(26,0)*_Training(i,5)+_Parameters(63,0)*pow(_Training(i,1),2)+_Parameters(64,0)*pow(_Training(i,2),2)+_Parameters(65,0)*pow(_Training(i,3),2)+_Parameters(66,0)*pow(_Training(i,4),2)+_Parameters(68,0)*_Training(i,1)*_Training(i,2)+_Parameters(69,0)*_Training(i,1)*_Training(i,3)+_Parameters(70,0)*_Training(i,1)*_Training(i,4)+_Parameters(71,0)*_Training(i,1)*_Training(i,5)+_Parameters(72,0)*_Training(i,2)*_Training(i,3)+_Parameters(73,0)*_Training(i,2)*_Training(i,4)+_Parameters(74,0)*_Training(i,2)*_Training(i,5)+_Parameters(75,0)*_Training(i,3)*_Training(i,4)+_Parameters(76,0)*_Training(i,3)*_Training(i,5)+_Parameters(77,0)*_Training(i,4)*_Training(i,5)+2*_Parameters(78,0)*_Training(i,1)*_Training(i,6)+2*_Parameters(79,0)*_Training(i,2)*_Training(i,6)+2*_Parameters(80,0)*_Training(i,3)*_Training(i,6)+2*_Parameters(81,0)*_Training(i,4)*_Training(i,6)+2*_Parameters(85,0)*_Training(i,5)*_Training(i,6)+2*_Parameters(67,0)*pow(_Training(i,5),2)+3*_Parameters(83,0)*pow(_Training(i,6),2)+_Parameters(154,0)*pow(_Training(i,1),3)+_Parameters(155,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(165,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(169,0)*pow(_Training(i,2),3)+_Parameters(156,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(159,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(170,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(166,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(176,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(179,0)*pow(_Training(i,3),3)+3*_Parameters(157,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(160,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(162,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(171,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(173,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(180,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(167,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(177,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(183,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(185,0)*pow(_Training(i,4),3)+3*_Parameters(158,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(161,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(163,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(164,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(174,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(175,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(182,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(172,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(181,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(186,0)*pow(_Training(i,4),2)*_Training(i,5)+2*_Parameters(168,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(178,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(184,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(187,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(188,0)*pow(_Training(i,5),3)+2*_Parameters(189,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(190,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(191,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(192,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(193,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(195,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(196,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(197,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(199,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(200,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(202,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+2*_Parameters(195,0)*pow(_Training(i,2),2)*_Training(i,6)+2*_Parameters(198,0)*pow(_Training(i,3),2)*_Training(i,6)+2*_Parameters(196,0)*pow(_Training(i,4),2)*_Training(i,6)+2*_Parameters(203,0)*pow(_Training(i,5),2)*_Training(i,6)+3*_Parameters(204,0)*_Training(i,1)*pow(_Training(i,6),2)+3*_Parameters(205,0)*_Training(i,2)*pow(_Training(i,6),2)+3*_Parameters(206,0)*_Training(i,3)*pow(_Training(i,6),2)+3*_Parameters(207,0)*_Training(i,4)*pow(_Training(i,6),2)+3*_Parameters(208,0)*_Training(i,5)*pow(_Training(i,6),2)+4*_Parameters(209,0)*pow(_Training(i,6),3);
                    
                    
                    
                    break;
                    
                case 7:
                    
                    _Jacobian(6,0) = _Parameters(1,0)+2*_Parameters(8,0)*_Training(i,1)+_Parameters(9,0)*_Training(i,2)+_Parameters(11,0)*_Training(i,3)+_Parameters(14,0)*_Training(i,4)+_Parameters(18,0)*_Training(i,5)+_Parameters(23,0)*_Training(i,6)+_Parameters(29,0)*_Training(i,7)+3*_Parameters(36,0)*pow(_Training(i,1),2)+2*_Parameters(37,0)*_Training(i,1)*_Training(i,2)+_Parameters(38,0)*pow(_Training(i,2),2)+2*_Parameters(40,0)*_Training(i,1)*_Training(i,3)+_Parameters(42,0)*_Training(i,2)*_Training(i,3)+_Parameters(43,0)*pow(_Training(i,3),2)+2*_Parameters(46,0)*_Training(i,1)*_Training(i,4)+_Parameters(49,0)*_Training(i,2)*_Training(i,4)+_Parameters(50,0)*_Training(i,3)*_Training(i,4)+_Parameters(52,0)*pow(_Training(i,4),2)+2*_Parameters(56,0)*_Training(i,1)*_Training(i,5)+_Parameters(60,0)*_Training(i,2)*_Training(i,5)+_Parameters(61,0)*_Training(i,3)*_Training(i,5)+_Parameters(62,0)*_Training(i,4)*_Training(i,5)+_Parameters(66,0)*pow(_Training(i,5),2)+2*_Parameters(71,0)*_Training(i,1)*_Training(i,6)+_Parameters(76,0)*_Training(i,2)*_Training(i,6)+_Parameters(77,0)*_Training(i,3)*_Training(i,6)+_Parameters(78,0)*_Training(i,4)*_Training(i,6)+_Parameters(79,0)*_Training(i,5)*_Training(i,6)+_Parameters(86,0)*pow(_Training(i,6),2)+2*_Parameters(92,0)*_Training(i,1)*_Training(i,7)+_Parameters(98,0)*_Training(i,2)*_Training(i,7)+_Parameters(99,0)*_Training(i,3)*_Training(i,7)+_Parameters(100,0)*_Training(i,4)*_Training(i,7)+_Parameters(101,0)*_Training(i,5)*_Training(i,7)+_Parameters(102,0)*_Training(i,6)*_Training(i,7)+_Parameters(113,0)*pow(_Training(i,7),2)+4*_Parameters(120,0)*pow(_Training(i,1),3)+3*_Parameters(121,0)*pow(_Training(i,1),2)*_Training(i,2)+2*_Parameters(122,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(123,0)*pow(_Training(i,2),3)+3*_Parameters(125,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(127,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(128,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(129,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(130,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(132,0)*pow(_Training(i,3),3)+3*_Parameters(99,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(136,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(137,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+2*_Parameters(138,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(139,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(140,0)*pow(_Training(i,3),2)*_Training(i,4)+2*_Parameters(145,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(146,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(147,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(151,0)*pow(_Training(i,4),3)+3*_Parameters(155,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(156,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(157,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(158,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(159,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(160,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(161,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(162,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(163,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(164,0)*pow(_Training(i,4),2)*_Training(i,5)+2*_Parameters(175,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(176,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(177,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(178,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(185,0)*pow(_Training(i,5),3)+3*_Parameters(190,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(191,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(156,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(193,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(194,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(195,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(196,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(197,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(198,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(199,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(200,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(201,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(202,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(203,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(204,0)*pow(_Training(i,5),2)*_Training(i,6)+2*_Parameters(225,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(226,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(227,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(228,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(229,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(239,0)*pow(_Training(i,6),3)+3*_Parameters(246,0)*pow(_Training(i,1),2)*_Training(i,7)+2*_Parameters(247,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+2*_Parameters(248,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+2*_Parameters(249,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+2*_Parameters(250,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+2*_Parameters(251,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+_Parameters(252,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+_Parameters(253,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+_Parameters(254,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+_Parameters(255,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+_Parameters(256,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+_Parameters(257,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+_Parameters(258,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+_Parameters(259,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+_Parameters(260,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+_Parameters(261,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+_Parameters(262,0)*pow(_Training(i,2),2)*_Training(i,7)+_Parameters(263,0)*pow(_Training(i,3),2)*_Training(i,7)+_Parameters(264,0)*pow(_Training(i,4),2)*_Training(i,7)+_Parameters(265,0)*pow(_Training(i,5),2)*_Training(i,7)+_Parameters(266,0)*pow(_Training(i,6),2)*_Training(i,7)+2*_Parameters(302,0)*_Training(i,1)*pow(_Training(i,7),2)+_Parameters(303,0)*_Training(i,2)*pow(_Training(i,7),2)+_Parameters(304,0)*_Training(i,3)*pow(_Training(i,7),2)+_Parameters(305,0)*_Training(i,4)*pow(_Training(i,7),2)+_Parameters(306,0)*_Training(i,5)*pow(_Training(i,7),2)+_Parameters(307,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(323,0)*pow(_Training(i,7),3);
                    
                    _Jacobian(6,1) = _Parameters(2,0)+2*_Parameters(10,0)*_Training(i,2)+_Parameters(9,0)*_Training(i,1)+_Parameters(12,0)*_Training(i,3)+_Parameters(15,0)*_Training(i,4)+_Parameters(19,0)*_Training(i,5)+_Parameters(24,0)*_Training(i,6)+_Parameters(30,0)*_Training(i,7)+3*_Parameters(39,0)*pow(_Training(i,2),2)+2*_Parameters(38,0)*_Training(i,1)*_Training(i,2)+_Parameters(37,0)*pow(_Training(i,1),2)+2*_Parameters(41,0)*_Training(i,2)*_Training(i,3)+_Parameters(42,0)*_Training(i,1)*_Training(i,3)+_Parameters(44,0)*pow(_Training(i,3),2)+2*_Parameters(47,0)*_Training(i,2)*_Training(i,4)+_Parameters(49,0)*_Training(i,1)*_Training(i,4)+_Parameters(51,0)*_Training(i,3)*_Training(i,4)+_Parameters(53,0)*pow(_Training(i,4),2)+2*_Parameters(57,0)*_Training(i,2)*_Training(i,5)+_Parameters(60,0)*_Training(i,1)*_Training(i,5)+_Parameters(63,0)*_Training(i,3)*_Training(i,5)+_Parameters(64,0)*_Training(i,4)*_Training(i,5)+_Parameters(67,0)*pow(_Training(i,5),2)+2*_Parameters(72,0)*_Training(i,2)*_Training(i,6)+_Parameters(76,0)*_Training(i,1)*_Training(i,6)+_Parameters(80,0)*_Training(i,3)*_Training(i,6)+_Parameters(81,0)*_Training(i,4)*_Training(i,6)+_Parameters(82,0)*_Training(i,5)*_Training(i,6)+_Parameters(87,0)*pow(_Training(i,6),2)+2*_Parameters(93,0)*_Training(i,2)*_Training(i,7)+_Parameters(98,0)*_Training(i,1)*_Training(i,7)+_Parameters(103,0)*_Training(i,3)*_Training(i,7)+_Parameters(104,0)*_Training(i,4)*_Training(i,7)+_Parameters(105,0)*_Training(i,5)*_Training(i,7)+_Parameters(106,0)*_Training(i,6)*_Training(i,7)+_Parameters(114,0)*pow(_Training(i,7),2)+3*_Parameters(85+36,0)*pow(_Training(i,1),3)+2*_Parameters(86+36,0)*pow(_Training(i,1),2)*_Training(i,2)+3*_Parameters(87+36,0)*_Training(i,1)*pow(_Training(i,2),2)+4*_Parameters(88+36,0)*pow(_Training(i,2),3)+3*_Parameters(90+36,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(91+36,0)*_Training(i,1)*_Training(i,3)+2*_Parameters(92+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(94+36,0)*_Training(i,1)*pow(_Training(i,3),2)+2*_Parameters(95+36,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(97+36,0)*pow(_Training(i,3),3)+_Parameters(136,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(138,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(139,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+3*_Parameters(141,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(142,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(143,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(146,0)*_Training(i,1)*pow(_Training(i,4),2)+2*_Parameters(148,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(149,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(152,0)*pow(_Training(i,4),3)+_Parameters(156,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(163,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(159,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(160,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(166,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(167,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(168,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+3*_Parameters(165,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(169,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(170,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(178,0)*_Training(i,1)*pow(_Training(i,5),2)+2*_Parameters(179,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(180,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(181,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(186,0)*pow(_Training(i,5),3)+_Parameters(191,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(201,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(195,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(196,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(197,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(206,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(207,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(208,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(209,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(210,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(211,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+3*_Parameters(205,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(206,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(213,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(214,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(226,0)*_Training(i,1)*pow(_Training(i,6),2)+2*_Parameters(230,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(231,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(232,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(233,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(240,0)*pow(_Training(i,6),3)+_Parameters(247,0)*pow(_Training(i,1),2)*_Training(i,7)+2*_Parameters(262,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+_Parameters(252,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+_Parameters(253,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+_Parameters(254,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+_Parameters(255,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+2*_Parameters(268,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+2*_Parameters(269,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+2*_Parameters(270,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+2*_Parameters(271,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+_Parameters(272,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+_Parameters(273,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+_Parameters(274,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+_Parameters(275,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+_Parameters(276,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+_Parameters(277,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+3*_Parameters(267,0)*pow(_Training(i,2),2)*_Training(i,7)+_Parameters(278,0)*pow(_Training(i,3),2)*_Training(i,7)+_Parameters(279,0)*pow(_Training(i,4),2)*_Training(i,7)+_Parameters(280,0)*pow(_Training(i,5),2)*_Training(i,7)+_Parameters(281,0)*pow(_Training(i,6),2)*_Training(i,7)+_Parameters(303,0)*_Training(i,1)*pow(_Training(i,7),2)+2*_Parameters(308,0)*_Training(i,2)*pow(_Training(i,7),2)+_Parameters(309,0)*_Training(i,3)*pow(_Training(i,7),2)+_Parameters(310,0)*_Training(i,4)*pow(_Training(i,7),2)+_Parameters(311,0)*_Training(i,5)*pow(_Training(i,7),2)+_Parameters(312,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(324,0)*pow(_Training(i,7),3);
                    
                    
                    _Jacobian(6,2) = _Parameters(3,0)+2*_Parameters(13,0)*_Training(i,3)+_Parameters(12,0)*_Training(i,2)+_Parameters(11,0)*_Training(i,1)+_Parameters(16,0)*_Training(i,4)+_Parameters(20,0)*_Training(i,5)+_Parameters(25,0)*_Training(i,6)+_Parameters(31,0)*_Training(i,7)+3*_Parameters(45,0)*pow(_Training(i,3),2)+2*_Parameters(44,0)*_Training(i,3)*_Training(i,2)+_Parameters(41,0)*pow(_Training(i,2),2)+2*_Parameters(43,0)*_Training(i,1)*_Training(i,3)+_Parameters(42,0)*_Training(i,1)*_Training(i,2)+_Parameters(40,0)*pow(_Training(i,1),2)+2*_Parameters(48,0)*_Training(i,3)*_Training(i,4)+_Parameters(50,0)*_Training(i,1)*_Training(i,4)+_Parameters(51,0)*_Training(i,2)*_Training(i,4)+_Parameters(54,0)*pow(_Training(i,4),2)+2*_Parameters(58,0)*_Training(i,3)*_Training(i,5)+_Parameters(61,0)*_Training(i,1)*_Training(i,5)+_Parameters(63,0)*_Training(i,2)*_Training(i,5)+_Parameters(65,0)*_Training(i,4)*_Training(i,5)+_Parameters(68,0)*pow(_Training(i,5),2)+2*_Parameters(73,0)*_Training(i,3)*_Training(i,6)+_Parameters(77,0)*_Training(i,1)*_Training(i,6)+_Parameters(80,0)*_Training(i,2)*_Training(i,6)+_Parameters(83,0)*_Training(i,4)*_Training(i,6)+_Parameters(84,0)*_Training(i,5)*_Training(i,6)+_Parameters(88,0)*pow(_Training(i,6),2)+2*_Parameters(94,0)*_Training(i,3)*_Training(i,7)+_Parameters(99,0)*_Training(i,1)*_Training(i,7)+_Parameters(103,0)*_Training(i,2)*_Training(i,7)+_Parameters(107,0)*_Training(i,4)*_Training(i,7)+_Parameters(108,0)*_Training(i,5)*_Training(i,7)+_Parameters(109,0)*_Training(i,6)*_Training(i,7)+_Parameters(115,0)*pow(_Training(i,7),2)+_Parameters(89+36,0)*pow(_Training(i,1),3)+_Parameters(90+36,0)*pow(_Training(i,2),3)+_Parameters(91+36,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(92+36,0)*_Training(i,1)*pow(_Training(i,2),2)+2*_Parameters(93+36,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(94+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+2*_Parameters(95+36,0)*pow(_Training(i,2),2)*_Training(i,3)+3*_Parameters(96+36,0)*_Training(i,1)*pow(_Training(i,3),2)+3*_Parameters(97+36,0)*_Training(i,2)*pow(_Training(i,3),2)+4*_Parameters(98+36,0)*pow(_Training(i,3),3)+_Parameters(101+36,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(103+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(104+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(106+36,0)*pow(_Training(i,2),2)*_Training(i,4)+2*_Parameters(107+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+3*_Parameters(108+36,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(111+36,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(113+36,0)*_Training(i,2)*pow(_Training(i,4),2)+2*_Parameters(114+36,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(117,0)*pow(_Training(i,4),3)+_Parameters(121,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(123+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(127+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(125+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(133+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(132+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(136+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(130+36,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(135+36,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(137+36,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(141+36,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(144+36,0)*_Training(i,2)*pow(_Training(i,5),2)+2*_Parameters(146+36,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(147+36,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(186,0)*pow(_Training(i,5),3)+_Parameters(156+36,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(159+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(166+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(162+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(163+36,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(176+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(173+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(174,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(180,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(181+36,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(182+36,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(170+36,0)*pow(_Training(i,2),2)*_Training(i,6)+3*_Parameters(179+36,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(183+36,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(184+36,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(191+36,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(195+36,0)*_Training(i,2)*pow(_Training(i,6),2)+2*_Parameters(198+36,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(200+36,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(201+36,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(207+36,0)*pow(_Training(i,6),3)+_Parameters(248,0)*pow(_Training(i,1),2)*_Training(i,7)+_Parameters(252,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+2*_Parameters(263,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+_Parameters(256,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+_Parameters(257,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+_Parameters(258,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+2*_Parameters(278,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+_Parameters(272,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+_Parameters(273,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+_Parameters(274,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+2*_Parameters(283,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+2*_Parameters(284,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+2*_Parameters(285,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+_Parameters(286,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+_Parameters(287,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+_Parameters(288,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+_Parameters(268,0)*pow(_Training(i,2),2)*_Training(i,7)+3*_Parameters(282,0)*pow(_Training(i,3),2)*_Training(i,7)+_Parameters(289,0)*pow(_Training(i,4),2)*_Training(i,7)+_Parameters(290,0)*pow(_Training(i,5),2)*_Training(i,7)+_Parameters(291,0)*pow(_Training(i,6),2)*_Training(i,7)+_Parameters(304,0)*_Training(i,1)*pow(_Training(i,7),2)+_Parameters(309,0)*_Training(i,2)*pow(_Training(i,7),2)+2*_Parameters(313,0)*_Training(i,3)*pow(_Training(i,7),2)+_Parameters(314,0)*_Training(i,4)*pow(_Training(i,7),2)+_Parameters(315,0)*_Training(i,5)*pow(_Training(i,7),2)+_Parameters(316,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(325,0)*pow(_Training(i,7),3);
                    
                    _Jacobian(6,3) = _Parameters(4,0)+_Parameters(14,0)*_Training(i,1)+_Parameters(15,0)*_Training(i,2)+_Parameters(16,0)*_Training(i,3)+2*_Parameters(17,0)*_Training(i,4)+_Parameters(21,0)*_Training(i,5)+_Parameters(26,0)*_Training(i,6)+_Parameters(32,0)*_Training(i,7)+_Parameters(46,0)*pow(_Training(i,1),2)+_Parameters(47,0)*pow(_Training(i,2),2)+_Parameters(48,0)*pow(_Training(i,3),2)+_Parameters(49,0)*_Training(i,1)*_Training(i,2)+_Parameters(50,0)*_Training(i,1)*_Training(i,3)+_Parameters(51,0)*_Training(i,2)*_Training(i,3)+2*_Parameters(52,0)*_Training(i,1)*_Training(i,4)+2*_Parameters(53,0)*_Training(i,2)*_Training(i,4)+2*_Parameters(53,0)*_Training(i,3)*_Training(i,4)+3*_Parameters(55,0)*pow(_Training(i,4),2)+2*_Parameters(59,0)*_Training(i,4)*_Training(i,5)+_Parameters(62,0)*_Training(i,1)*_Training(i,5)+_Parameters(64,0)*_Training(i,2)*_Training(i,5)+_Parameters(65,0)*_Training(i,3)*_Training(i,5)+_Parameters(69,0)*pow(_Training(i,5),2)+2*_Parameters(74,0)*_Training(i,4)*_Training(i,6)+_Parameters(78,0)*_Training(i,1)*_Training(i,6)+_Parameters(81,0)*_Training(i,2)*_Training(i,6)+_Parameters(83,0)*_Training(i,3)*_Training(i,6)+_Parameters(85,0)*_Training(i,5)*_Training(i,6)+_Parameters(89,0)*pow(_Training(i,6),2)+2*_Parameters(95,0)*_Training(i,1)*_Training(i,7)+_Parameters(100,0)*_Training(i,1)*_Training(i,7)+_Parameters(104,0)*_Training(i,2)*_Training(i,7)+_Parameters(107,0)*_Training(i,3)*_Training(i,7)+_Parameters(110,0)*_Training(i,5)*_Training(i,7)+_Parameters(111,0)*_Training(i,6)*_Training(i,7)+_Parameters(116,0)*pow(_Training(i,7),2)+_Parameters(99+36,0)*pow(_Training(i,1),3)+_Parameters(100+36,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(101+36,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(102+36,0)*pow(_Training(i,2),2)*_Training(i,1)+_Parameters(103+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(104+36,0)*pow(_Training(i,3),2)*_Training(i,1)+_Parameters(105+36,0)*pow(_Training(i,3),3)+_Parameters(106+36,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(107+36,0)*pow(_Training(i,3),2)*_Training(i,2)+_Parameters(108+36,0)*pow(_Training(i,3),3)+2*_Parameters(109+36,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(110+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+2*_Parameters(111+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(112+36,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(113+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+2*_Parameters(114+36,0)*pow(_Training(i,3),2)*_Training(i,4)+3*_Parameters(115+36,0)*pow(_Training(i,4),2)*_Training(i,1)+3*_Parameters(116+36,0)*pow(_Training(i,4),2)*_Training(i,2)+3*_Parameters(117+36,0)*pow(_Training(i,4),2)*_Training(i,3)+4*_Parameters(118+36,0)*pow(_Training(i,4),3)+_Parameters(122+36,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(124+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(125+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(128+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(132+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(134+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(137+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(131+36,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(136+36,0)*pow(_Training(i,3),2)*_Training(i,5)+3*_Parameters(138+36,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(142+36,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(145+36,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(147+36,0)*_Training(i,3)*pow(_Training(i,5),2)+2*_Parameters(148+36,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(152+36,0)*pow(_Training(i,5),3)+_Parameters(156+36,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(160+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(162+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(167+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(164+36,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(173+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(177+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(175+36,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(183+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(182+36,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(186+36,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(172+36,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(180+36,0)*pow(_Training(i,3),2)*_Training(i,6)+3*_Parameters(185+36,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(188+36,0)*pow(_Training(i,5),2)*_Training(i,6)+2*_Parameters(192+36,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(196+36,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(199+36,0)*_Training(i,3)*pow(_Training(i,6),2)+2*_Parameters(201+36,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(202+36,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(207+36,0)*pow(_Training(i,6),3)+_Parameters(249,0)*pow(_Training(i,1),2)*_Training(i,7)+_Parameters(253,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+_Parameters(256,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+2*_Parameters(264,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+_Parameters(259,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+_Parameters(260,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+_Parameters(272,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+2*_Parameters(279,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+_Parameters(275,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+_Parameters(276,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+2*_Parameters(289,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+_Parameters(286,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+_Parameters(287,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+2*_Parameters(293,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+2*_Parameters(294,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+_Parameters(295,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+_Parameters(269,0)*pow(_Training(i,2),2)*_Training(i,7)+_Parameters(283,0)*pow(_Training(i,3),2)*_Training(i,7)+3*_Parameters(292,0)*pow(_Training(i,4),2)*_Training(i,7)+_Parameters(296,0)*pow(_Training(i,5),2)*_Training(i,7)+_Parameters(297,0)*pow(_Training(i,6),2)*_Training(i,7)+_Parameters(305,0)*_Training(i,1)*pow(_Training(i,7),2)+_Parameters(310,0)*_Training(i,2)*pow(_Training(i,7),2)+_Parameters(314,0)*_Training(i,3)*pow(_Training(i,7),2)+2*_Parameters(317,0)*_Training(i,4)*pow(_Training(i,7),2)+_Parameters(318,0)*_Training(i,5)*pow(_Training(i,7),2)+_Parameters(319,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(326,0)*pow(_Training(i,7),3);
                    
                    _Jacobian(6,4) = _Parameters(5,0)+_Parameters(18,0)*_Training(i,1)+_Parameters(19,0)*_Training(i,2)+_Parameters(20,0)*_Training(i,3)+_Parameters(21,0)*_Training(i,4)+2*_Parameters(22,0)*_Training(i,5)+_Parameters(27,0)*_Training(i,6)+_Parameters(33,0)*_Training(i,7)+_Parameters(56,0)*pow(_Training(i,1),2)+_Parameters(57,0)*pow(_Training(i,2),2)+_Parameters(58,0)*pow(_Training(i,3),2)+_Parameters(59,0)*pow(_Training(i,4),2)+_Parameters(60,0)*_Training(i,1)*_Training(i,2)+_Parameters(61,0)*_Training(i,1)*_Training(i,3)+_Parameters(62,0)*_Training(i,1)*_Training(i,4)+_Parameters(63,0)*_Training(i,2)*_Training(i,3)+_Parameters(64,0)*_Training(i,2)*_Training(i,4)+_Parameters(65,0)*_Training(i,3)*_Training(i,4)+2*_Parameters(66,0)*_Training(i,1)*_Training(i,5)+2*_Parameters(67,0)*_Training(i,2)*_Training(i,5)+2*_Parameters(68,0)*_Training(i,3)*_Training(i,5)+2*_Parameters(69,0)*_Training(i,4)*_Training(i,5)+3*_Parameters(70,0)*pow(_Training(i,5),2)+2*_Parameters(75,0)*_Training(i,5)*_Training(i,6)+_Parameters(79,0)*_Training(i,1)*_Training(i,6)+_Parameters(82,0)*_Training(i,2)*_Training(i,6)+_Parameters(84,0)*_Training(i,3)*_Training(i,6)+_Parameters(85,0)*_Training(i,4)*_Training(i,6)+_Parameters(90,0)*pow(_Training(i,6),2)+2*_Parameters(96,0)*_Training(i,5)*_Training(i,7)+_Parameters(101,0)*_Training(i,1)*_Training(i,7)+_Parameters(105,0)*_Training(i,2)*_Training(i,7)+_Parameters(108,0)*_Training(i,3)*_Training(i,7)+_Parameters(110,0)*_Training(i,4)*_Training(i,7)+_Parameters(112,0)*_Training(i,6)*_Training(i,7)+_Parameters(117,0)*pow(_Training(i,7),2)+_Parameters(119+36,0)*pow(_Training(i,1),3)+_Parameters(120+36,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(126+36,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(131+36,0)*pow(_Training(i,2),3)+_Parameters(121+36,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(123+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(130+36,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(127+36,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(133+36,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(135+36,0)*pow(_Training(i,3),3)+_Parameters(122+36,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(124+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(125+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(131+36,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(132+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(136+36,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(100+36,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(106+36,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(137+36,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(138+36,0)*pow(_Training(i,4),3)+2*_Parameters(139+36,0)*pow(_Training(i,1),2)*_Training(i,5)+2*_Parameters(140+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+2*_Parameters(141+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(142+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+2*_Parameters(144+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+2*_Parameters(145+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+2*_Parameters(147+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+2*_Parameters(143+36,0)*pow(_Training(i,2),2)*_Training(i,5)+2*_Parameters(146+36,0)*pow(_Training(i,3),2)*_Training(i,5)+2*_Parameters(148+36,0)*pow(_Training(i,4),2)*_Training(i,5)+3*_Parameters(149+36,0)*_Training(i,1)*pow(_Training(i,5),2)+3*_Parameters(150+36,0)*_Training(i,2)*pow(_Training(i,5),2)+3*_Parameters(151+36,0)*_Training(i,3)*pow(_Training(i,5),2)+3*_Parameters(152+36,0)*_Training(i,4)*pow(_Training(i,5),2)+4*_Parameters(153+36,0)*pow(_Training(i,5),3)+_Parameters(158+36,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(161+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(163+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(164+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(168+36,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(174+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(175+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(178+36,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(182+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(184+36,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(187+36,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(172+36,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(181+36,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(186+36,0)*pow(_Training(i,4),2)*_Training(i,6)+2*_Parameters(188+36,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(193+36,0)*_Training(i,1)*pow(_Training(i,6),2)+_Parameters(197+36,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(200+36,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(202+36,0)*_Training(i,4)*pow(_Training(i,6),2)+2*_Parameters(203+36,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(208+36,0)*pow(_Training(i,6),3)+_Parameters(250,0)*pow(_Training(i,1),2)*_Training(i,7)+_Parameters(254,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+_Parameters(257,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+_Parameters(259,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+2*_Parameters(265,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+_Parameters(261,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+_Parameters(273,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+_Parameters(275,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+2*_Parameters(280,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+_Parameters(277,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+_Parameters(286,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+2*_Parameters(290,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+_Parameters(288,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+2*_Parameters(296,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+_Parameters(295,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+2*_Parameters(298,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+_Parameters(270,0)*pow(_Training(i,2),2)*_Training(i,7)+_Parameters(284,0)*pow(_Training(i,3),2)*_Training(i,7)+_Parameters(293,0)*pow(_Training(i,4),2)*_Training(i,7)+3*_Parameters(298,0)*pow(_Training(i,5),2)*_Training(i,7)+_Parameters(300,0)*pow(_Training(i,6),2)*_Training(i,7)+_Parameters(306,0)*_Training(i,1)*pow(_Training(i,7),2)+_Parameters(311,0)*_Training(i,2)*pow(_Training(i,7),2)+_Parameters(315,0)*_Training(i,3)*pow(_Training(i,7),2)+_Parameters(318,0)*_Training(i,4)*pow(_Training(i,7),2)+2*_Parameters(320,0)*_Training(i,5)*pow(_Training(i,7),2)+_Parameters(321,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(327,0)*pow(_Training(i,7),3);
                    
                    _Jacobian(6,5) = _Parameters(6,0)+_Parameters(23,0)*_Training(i,1)+_Parameters(24,0)*_Training(i,2)+_Parameters(25,0)*_Training(i,3)+_Parameters(26,0)*_Training(i,4)+_Parameters(27,0)*_Training(i,5)+2*_Parameters(28,0)*_Training(i,6)+_Parameters(34,0)*_Training(i,7)+_Parameters(71,0)*pow(_Training(i,1),2)+_Parameters(72,0)*pow(_Training(i,2),2)+_Parameters(73,0)*pow(_Training(i,3),2)+_Parameters(74,0)*pow(_Training(i,4),2)+_Parameters(75,0)*pow(_Training(i,5),2)+_Parameters(76,0)*_Training(i,1)*_Training(i,2)+_Parameters(77,0)*_Training(i,1)*_Training(i,3)+_Parameters(78,0)*_Training(i,1)*_Training(i,4)+_Parameters(79,0)*_Training(i,1)*_Training(i,5)+_Parameters(80,0)*_Training(i,2)*_Training(i,3)+_Parameters(81,0)*_Training(i,2)*_Training(i,4)+_Parameters(82,0)*_Training(i,2)*_Training(i,5)+_Parameters(83,0)*_Training(i,3)*_Training(i,4)+_Parameters(84,0)*_Training(i,3)*_Training(i,5)+_Parameters(85,0)*_Training(i,4)*_Training(i,5)+2*_Parameters(86,0)*_Training(i,1)*_Training(i,6)+2*_Parameters(87,0)*_Training(i,2)*_Training(i,6)+2*_Parameters(88,0)*_Training(i,3)*_Training(i,6)+2*_Parameters(89,0)*_Training(i,4)*_Training(i,6)+2*_Parameters(90,0)*_Training(i,5)*_Training(i,6)+3*_Parameters(91,0)*pow(_Training(i,6),2)+2*_Parameters(97,0)*_Training(i,6)*_Training(i,7)+_Parameters(102,0)*_Training(i,1)*_Training(i,7)+_Parameters(106,0)*_Training(i,2)*_Training(i,7)+_Parameters(109,0)*_Training(i,3)*_Training(i,7)+_Parameters(111,0)*_Training(i,4)*_Training(i,7)+_Parameters(112,0)*_Training(i,5)*_Training(i,7)+_Parameters(118,0)*pow(_Training(i,7),2)+_Parameters(154+36,0)*pow(_Training(i,1),3)+_Parameters(155+36,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(165+36,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(169+36,0)*pow(_Training(i,2),3)+_Parameters(156+36,0)*pow(_Training(i,1),2)*_Training(i,3)+2*_Parameters(159+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(170+36,0)*pow(_Training(i,2),2)*_Training(i,3)+2*_Parameters(166+36,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(176+36,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(179+36,0)*pow(_Training(i,3),3)+3*_Parameters(157+36,0)*pow(_Training(i,1),2)*_Training(i,4)+2*_Parameters(160+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(162+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(171+36,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(173+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(180+36,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(167+36,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(177+36,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(183+36,0)*_Training(i,3)*pow(_Training(i,4),2)+_Parameters(185+36,0)*pow(_Training(i,4),3)+3*_Parameters(158+36,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(161+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(163+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+2*_Parameters(164+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(174+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(175+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(182+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(172+36,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(181+36,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(186+36,0)*pow(_Training(i,4),2)*_Training(i,5)+2*_Parameters(168+36,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(178+36,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(184+36,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(187+36,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(188+36,0)*pow(_Training(i,5),3)+2*_Parameters(189+36,0)*pow(_Training(i,1),2)*_Training(i,6)+2*_Parameters(190+36,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+2*_Parameters(191+36,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+2*_Parameters(192+36,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+2*_Parameters(193+36,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+2*_Parameters(195+36,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+2*_Parameters(196+36,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+2*_Parameters(197+36,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+2*_Parameters(199+36,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+2*_Parameters(200+36,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+2*_Parameters(202+36,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+2*_Parameters(195+36,0)*pow(_Training(i,2),2)*_Training(i,6)+2*_Parameters(198+36,0)*pow(_Training(i,3),2)*_Training(i,6)+2*_Parameters(196+36,0)*pow(_Training(i,4),2)*_Training(i,6)+2*_Parameters(203+36,0)*pow(_Training(i,5),2)*_Training(i,6)+3*_Parameters(204+36,0)*_Training(i,1)*pow(_Training(i,6),2)+3*_Parameters(205+36,0)*_Training(i,2)*pow(_Training(i,6),2)+3*_Parameters(206+36,0)*_Training(i,3)*pow(_Training(i,6),2)+3*_Parameters(207+36,0)*_Training(i,4)*pow(_Training(i,6),2)+3*_Parameters(208+36,0)*_Training(i,5)*pow(_Training(i,6),2)+4*_Parameters(209+36,0)*pow(_Training(i,6),3)+_Parameters(251,0)*pow(_Training(i,1),2)*_Training(i,7)+_Parameters(255,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+_Parameters(258,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+_Parameters(260,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+_Parameters(261,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+2*_Parameters(266,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+_Parameters(274,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+_Parameters(276,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+_Parameters(277,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+2*_Parameters(281,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+_Parameters(287,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+_Parameters(288,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+_Parameters(291,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+_Parameters(295,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+2*_Parameters(297,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+2*_Parameters(300,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+_Parameters(271,0)*pow(_Training(i,2),2)*_Training(i,7)+_Parameters(285,0)*pow(_Training(i,3),2)*_Training(i,7)+_Parameters(294,0)*pow(_Training(i,4),2)*_Training(i,7)+_Parameters(299,0)*pow(_Training(i,5),2)*_Training(i,7)+3*_Parameters(301,0)*pow(_Training(i,6),2)*_Training(i,7)+_Parameters(307,0)*_Training(i,1)*pow(_Training(i,7),2)+_Parameters(312,0)*_Training(i,2)*pow(_Training(i,7),2)+_Parameters(316,0)*_Training(i,3)*pow(_Training(i,7),2)+_Parameters(319,0)*_Training(i,4)*pow(_Training(i,7),2)+_Parameters(321,0)*_Training(i,5)*pow(_Training(i,7),2)+2*_Parameters(322,0)*_Training(i,6)*pow(_Training(i,7),2)+_Parameters(328,0)*pow(_Training(i,7),3);
                    
                    _Jacobian(6,6) = _Parameters(7,0)+_Parameters(29,0)*_Training(i,1)+_Parameters(30,0)*_Training(i,2)+_Parameters(31,0)*_Training(i,3)+_Parameters(32,0)*_Training(i,4)+_Parameters(33,0)*_Training(i,5)+_Parameters(34,0)*_Training(i,6)+2*_Parameters(35,0)*_Training(i,7)+_Parameters(92,0)*pow(_Training(i,1),2)+_Parameters(93,0)*pow(_Training(i,2),2)+_Parameters(94,0)*pow(_Training(i,3),2)+_Parameters(95,0)*pow(_Training(i,4),2)+_Parameters(96,0)*pow(_Training(i,5),2)+_Parameters(97,0)*pow(_Training(i,6),2)+_Parameters(98,0)*_Training(i,1)*_Training(i,2)+_Parameters(99,0)*_Training(i,1)*_Training(i,3)+_Parameters(100,0)*_Training(i,1)*_Training(i,4)+_Parameters(101,0)*_Training(i,1)*_Training(i,5)+_Parameters(102,0)*_Training(i,1)*_Training(i,6)+_Parameters(103,0)*_Training(i,2)*_Training(i,3)+_Parameters(104,0)*_Training(i,2)*_Training(i,4)+_Parameters(105,0)*_Training(i,2)*_Training(i,5)+_Parameters(106,0)*_Training(i,2)*_Training(i,6)+_Parameters(107,0)*_Training(i,3)*_Training(i,4)+_Parameters(108,0)*_Training(i,3)*_Training(i,5)+_Parameters(109,0)*_Training(i,3)*_Training(i,6)+_Parameters(110,0)*_Training(i,4)*_Training(i,5)+_Parameters(111,0)*_Training(i,4)*_Training(i,6)+_Parameters(112,0)*_Training(i,5)*_Training(i,6)+2*_Parameters(113,0)*_Training(i,1)*_Training(i,7)+2*_Parameters(114,0)*_Training(i,2)*_Training(i,7)+2*_Parameters(115,0)*_Training(i,3)*_Training(i,7)+2*_Parameters(116,0)*_Training(i,4)*_Training(i,7)+2*_Parameters(117,0)*_Training(i,5)*_Training(i,7)+2*_Parameters(118,0)*_Training(i,6)*_Training(i,7)+3*_Parameters(119,0)*pow(_Training(i,7),2)+_Parameters(246,0)*pow(_Training(i,1),3)+_Parameters(247,0)*pow(_Training(i,1),2)*_Training(i,2)+_Parameters(248,0)*pow(_Training(i,1),2)*_Training(i,3)+_Parameters(249,0)*pow(_Training(i,1),2)*_Training(i,4)+_Parameters(250,0)*pow(_Training(i,1),2)*_Training(i,5)+_Parameters(251,0)*pow(_Training(i,1),2)*_Training(i,6)+_Parameters(252,0)*_Training(i,1)*_Training(i,2)*_Training(i,3)+_Parameters(253,0)*_Training(i,1)*_Training(i,2)*_Training(i,4)+_Parameters(254,0)*_Training(i,1)*_Training(i,2)*_Training(i,5)+_Parameters(255,0)*_Training(i,1)*_Training(i,2)*_Training(i,6)+_Parameters(256,0)*_Training(i,1)*_Training(i,3)*_Training(i,4)+_Parameters(257,0)*_Training(i,1)*_Training(i,3)*_Training(i,5)+_Parameters(258,0)*_Training(i,1)*_Training(i,3)*_Training(i,6)+_Parameters(259,0)*_Training(i,1)*_Training(i,4)*_Training(i,5)+_Parameters(260,0)*_Training(i,1)*_Training(i,4)*_Training(i,6)+_Parameters(261,0)*_Training(i,1)*_Training(i,5)*_Training(i,6)+_Parameters(262,0)*_Training(i,1)*pow(_Training(i,2),2)+_Parameters(263,0)*_Training(i,1)*pow(_Training(i,3),2)+_Parameters(264,0)*_Training(i,1)*pow(_Training(i,4),2)+_Parameters(265,0)*_Training(i,1)*pow(_Training(i,5),2)+_Parameters(266,0)*_Training(i,1)*pow(_Training(i,4),6)+_Parameters(267,0)*pow(_Training(i,2),3)+_Parameters(268,0)*pow(_Training(i,2),2)*_Training(i,3)+_Parameters(269,0)*pow(_Training(i,2),2)*_Training(i,4)+_Parameters(270,0)*pow(_Training(i,2),2)*_Training(i,5)+_Parameters(271,0)*pow(_Training(i,2),2)*_Training(i,6)+_Parameters(272,0)*_Training(i,2)*_Training(i,3)*_Training(i,4)+_Parameters(273,0)*_Training(i,2)*_Training(i,3)*_Training(i,5)+_Parameters(274,0)*_Training(i,2)*_Training(i,3)*_Training(i,6)+_Parameters(275,0)*_Training(i,2)*_Training(i,4)*_Training(i,5)+_Parameters(276,0)*_Training(i,2)*_Training(i,4)*_Training(i,6)+_Parameters(277,0)*_Training(i,2)*_Training(i,5)*_Training(i,6)+_Parameters(278,0)*_Training(i,2)*pow(_Training(i,3),2)+_Parameters(279,0)*_Training(i,2)*pow(_Training(i,4),2)+_Parameters(280,0)*_Training(i,2)*pow(_Training(i,5),2)+_Parameters(281,0)*_Training(i,2)*pow(_Training(i,6),2)+_Parameters(282,0)*pow(_Training(i,3),3)+_Parameters(283,0)*pow(_Training(i,3),2)*_Training(i,4)+_Parameters(284,0)*pow(_Training(i,3),2)*_Training(i,5)+_Parameters(285,0)*pow(_Training(i,3),2)*_Training(i,6)+_Parameters(286,0)*_Training(i,3)*_Training(i,4)*_Training(i,5)+_Parameters(287,0)*_Training(i,3)*_Training(i,4)*_Training(i,6)+_Parameters(288,0)*_Training(i,3)*_Training(i,5)*_Training(i,6)+_Parameters(289,0)*_Training(i,3)*pow(_Training(i,3),2)+_Parameters(290,0)*_Training(i,3)*pow(_Training(i,5),2)+_Parameters(291,0)*_Training(i,3)*pow(_Training(i,6),2)+_Parameters(292,0)*pow(_Training(i,4),3)+_Parameters(293,0)*pow(_Training(i,4),2)*_Training(i,5)+_Parameters(294,0)*pow(_Training(i,4),2)*_Training(i,6)+_Parameters(295,0)*_Training(i,4)*_Training(i,5)*_Training(i,6)+_Parameters(296,0)*_Training(i,4)*pow(_Training(i,5),2)+_Parameters(297,0)*_Training(i,4)*pow(_Training(i,6),2)+_Parameters(298,0)*pow(_Training(i,5),3)+_Parameters(299,0)*pow(_Training(i,5),2)*_Training(i,6)+_Parameters(300,0)*_Training(i,5)*pow(_Training(i,6),2)+_Parameters(301,0)*pow(_Training(i,6),3)+2*_Parameters(302,0)*pow(_Training(i,1),2)*_Training(i,7)+2*_Parameters(303,0)*_Training(i,1)*_Training(i,2)*_Training(i,7)+2*_Parameters(304,0)*_Training(i,1)*_Training(i,3)*_Training(i,7)+2*_Parameters(305,0)*_Training(i,1)*_Training(i,4)*_Training(i,7)+2*_Parameters(306,0)*_Training(i,1)*_Training(i,5)*_Training(i,7)+2*_Parameters(307,0)*_Training(i,1)*_Training(i,6)*_Training(i,7)+2*_Parameters(308,0)*pow(_Training(i,2),2)*_Training(i,7)+2*_Parameters(309,0)*_Training(i,2)*_Training(i,3)*_Training(i,7)+2*_Parameters(310,0)*_Training(i,2)*_Training(i,4)*_Training(i,7)+2*_Parameters(311,0)*_Training(i,2)*_Training(i,5)*_Training(i,7)+2*_Parameters(312,0)*_Training(i,2)*_Training(i,6)*_Training(i,7)+2*_Parameters(313,0)*pow(_Training(i,3),2)*_Training(i,7)+2*_Parameters(314,0)*_Training(i,3)*_Training(i,4)*_Training(i,7)+2*_Parameters(315,0)*_Training(i,3)*_Training(i,5)*_Training(i,7)+2*_Parameters(316,0)*_Training(i,3)*_Training(i,6)*_Training(i,7)+2*_Parameters(317,0)*pow(_Training(i,4),2)*_Training(i,7)+2*_Parameters(318,0)*_Training(i,4)*_Training(i,5)*_Training(i,7)+2*_Parameters(319,0)*_Training(i,4)*_Training(i,6)*_Training(i,7)+2*_Parameters(320,0)*pow(_Training(i,5),2)*_Training(i,7)+2*_Parameters(321,0)*_Training(i,5)*_Training(i,6)*_Training(i,7)+2*_Parameters(322,0)*pow(_Training(i,6),2)*_Training(i,7)+3*_Parameters(323,0)*_Training(i,1)*pow(_Training(i,7),2)+3*_Parameters(324,0)*_Training(i,2)*pow(_Training(i,7),2)+3*_Parameters(325,0)*_Training(i,3)*pow(_Training(i,7),2)+3*_Parameters(326,0)*_Training(i,4)*pow(_Training(i,7),2)+3*_Parameters(327,0)*_Training(i,5)*pow(_Training(i,7),2)+3*_Parameters(328,0)*_Training(i,6)*pow(_Training(i,7),2)+4*_Parameters(329,0)*pow(_Training(i,7),3);
                    
                    break;
            }
                break;
                
        }
        
        
        _TempJacobian = boost::numeric::ublas::prod(_Jacobian, _Q0);
        label = QR_factorization(_Q , _R , _TempJacobian);
        _Q0 = _Q;
        
        for (int k = 0 ; k < d ; k++)
            _LEs(k) = (i * _LEs(k) + log(_R(k,k)))/(i+1);
        
        double MaxLE = -1000000000.0;
        
        for (int mm = 0 ; mm < _LEs.size();mm++)
        {
            if (_LEs(mm) > MaxLE)
                MaxLE = _LEs(mm);
        }
        
        _LargestLyapunov(i) = MaxLE;
        
    }
    
    double f = abs(_LargestLyapunov(MatrixSize - 2*MAX_ED));
    
    
        return f;
}





// Estimating the Lyapunov exponent second approach.
template <typename Embedding, typename Lyapunov>
double lyapunov_estimation(Embedding& d , Lyapunov& l) {
    
    // input data can be found here (defined in config file or command line):
    
    
    matrix_type _InputMatrix;
    matrix_type _InitialMatrix;
    vector_type_distance _EuclideanDistance;
    _InputMatrix.resize(_IntegerInput.size() - MAX_ED + 1, d);
    _InitialMatrix.resize(_IntegerInput.size() - MAX_ED + 1, d);
    _EuclideanDistance.resize(_IntegerInput.size() - MAX_ED + 1);
    _EuclideanDistance(0) = std::numeric_limits<double>::max();
    
    for (int i = 0 ; i < d ; i++)
    {
    
        for (int j = 0 ; j < (_IntegerInput.size() - MAX_ED + 1) ; j++)
            _InputMatrix (j , i) = _IntegerInput(j + i);
        
        for (int j = 0 ; j < (_IntegerInput.size() - MAX_ED + 1) ; j++)
            _InitialMatrix (j , i) = _IntegerInput(j + i);
    
    }
    
    
    unsigned Index;
    l = 0; //Largest LE
    double MinDistance = std::numeric_limits<double>::max();
    
    double EvolZero;
    double EvolPrime;
    double EvolZeroTemp;
    double EvolPrimeTemp;
    
    for (unsigned i = 0 ; i < _EuclideanDistance.size() - 1; i++)
    {
        
        row_type InitialRow(_InitialMatrix, i);
        
        for (int j = 0 ; j < _EuclideanDistance.size() - 1; j++)
        {
            row_type InputRow(_InputMatrix, j);
            _EuclideanDistance(j) = norm_2(InputRow - InitialRow);
        }
        
        _EuclideanDistance(i) = MinDistance;
        
        Index = 0;
        double TempMinDistance = MinDistance;
        
        for (unsigned j = 0 ; j < _EuclideanDistance.size() - 1 ; j++)
        {
            if (_EuclideanDistance(j) < TempMinDistance)
            {
                TempMinDistance = _EuclideanDistance(j);
                Index = j;
            }
        }
        
        row_type EvolRowZero(_InputMatrix, i);
        row_type EvolRowOne (_InputMatrix, Index);
        
        EvolZero = norm_2(EvolRowOne - EvolRowZero);
        
        if (EvolZero == 0)
            EvolZero = EvolZeroTemp;
        
        row_type EvolRowPrime    (_InputMatrix, i + 1);
        row_type EvolRowPrimeOne (_InputMatrix, Index + 1);
        
        EvolPrime = norm_2(EvolRowPrimeOne - EvolRowPrime);
        
        if (EvolPrime == 0)
            EvolPrime = EvolPrimeTemp;
        
        l += log2(EvolPrime/EvolZero)/(_EuclideanDistance.size() - 1);
        EvolZeroTemp  = EvolZero;
        EvolPrimeTemp = EvolPrime;
    }
    
    if (l <= 0)
        return 0.01;
        
    return l;
}



// Estimating the prediction horizon.
template <typename PredictionHorizon, typename Lyapunov>
unsigned prediction_horizon_estimation(PredictionHorizon& h , Lyapunov& l) {
    namespace bnu=boost::numeric::ublas;
    // input data can be found here (defined in config file or command line):
    h = unsigned(1.0 / l + 1);
    return h;
}





int main(int argc, const char * argv[])
{

    typedef boost::numeric::ublas::matrix<int> matrix_type; //!< Type for matrix that will store raw sunspot numbers.
    typedef boost::numeric::ublas::matrix<double> matrix_type_estimated;
    typedef boost::numeric::ublas::matrix_row<matrix_type> row_type; //!< Row type for the matrix.
    typedef boost::numeric::ublas::vector<int> vector_type; //!< Type for a vector of sunspot numbers.
    typedef boost::numeric::ublas::vector<double> vector_type_distance; //!< Type for a vector of sunspot numbers.
    
    cout.setf(ios::boolalpha);
    // insert code here...
    std::vector<SunSpot> SSN;
    int Number , TotalNumber = 0 , Counter = 0;
    double * SunSpotTotal;
    
    std::vector<string> VectorLine;
    string Line;
    int Year , InitialYear , FinalYear;
    cout << "Enter the years you want to read (Beginning Ending): ";
    cin>> InitialYear >> FinalYear;
    
    for (Year = InitialYear ; Year <=FinalYear ; Year++)
    {
        if (Year % 4 == 0)
            TotalNumber += 366;
        else
            TotalNumber += 365;
    }
    
    SunSpotTotal = new (nothrow) double [TotalNumber];
    
    for (Year = InitialYear ; Year <=FinalYear ; Year++)
    {
        bool LeapYear;
        
        
        if (Year % 4 == 0)
        {
            LeapYear = true;
            Number = 366;
        }
        
        else
        {
            LeapYear = false;
            Number = 365;
        }
        
        string FileAddress = "/Users/mirmomeny/Desktop/SunSpotNumberViaMarkovNetwork/SunSpotNumber";
        
        string YearString;       // string which will contain the year
        
        ostringstream Convert;   // stream used for the conversion
        
        Convert << Year;
        
        YearString = Convert.str();
        
        FileAddress = FileAddress + YearString;
        
        ifstream MyFile (FileAddress);
        string SubString;
        double * SunSpot;
        
        if (MyFile.is_open())
        {
            for (int i = 1; i <= 4; i++)
            {
                getline (MyFile,Line);
            }
            
            if (LeapYear)
            {
                
                int MonthsDay[12] = {0,31,60,91,121,152,182,213,244,274,305,335};
                SunSpot = new (nothrow) double [366];
                
                for (int i = 0 ; i < 29 ; i++)
                {
                    getline (MyFile,Line);
                    istringstream LineStream(Line);
                    
                    for (int j = 0 ; j < 12 ; j++)
                    {
                        LineStream >> SubString;
                        stringstream StringSubStream (SubString);
                        int Temp;
                        StringSubStream >> Temp;
                        SunSpot[i + MonthsDay[j]] = Temp;
                        
                    }
                }
                
                int MonthsDayTemp1[11] = {29,89,120,150,181,211,242,273,303,334,364};
                
                getline (MyFile,Line);
                istringstream LineStream(Line);
                
                for (int j = 0 ; j < 11 ; j++)
                {
                    LineStream >> SubString;
                    stringstream StringSubStream (SubString);
                    int Temp;
                    StringSubStream >> Temp;
                    SunSpot[MonthsDayTemp1[j]] = Temp;
                    
                }
                
                int MonthsDayTemp2[7] = {30,90,151,212,243,304,365};
                
                getline (MyFile,Line);
                istringstream LineStream2(Line);
                
                for (int j = 0 ; j < 7 ; j++)
                {
                    LineStream2 >> SubString;
                    stringstream StringSubStream (SubString);
                    int Temp;
                    StringSubStream >> Temp;
                    SunSpot[MonthsDayTemp2[j]] = Temp;
                    
                }
                
            }
            else
            {
                int MonthsDay[12] = {0,31,59,90,120,151,181,212,243,273,304,334};
                
                SunSpot = new (nothrow) double [365];
                
                for (int i = 0 ; i < 28 ; i++)
                {
                    getline (MyFile,Line);
                    istringstream LineStream(Line);
                    
                    for (int j = 0 ; j < 12 ; j++)
                    {
                        LineStream >> SubString;
                        stringstream StringSubStream (SubString);
                        int Temp;
                        StringSubStream >> Temp;
                        SunSpot[i + MonthsDay[j]] = Temp;
                        
                    }
                }
                
                int MonthsDayTemp1[11] = {28,87,118,148,179,209,240,271,301,331,362};
                
                for (int i = 0 ; i < 2 ; i++)
                {
                    getline (MyFile,Line);
                    istringstream LineStream(Line);
                    
                    for (int j = 0 ; j < 11 ; j++)
                    {
                        LineStream >> SubString;
                        stringstream StringSubStream (SubString);
                        int Temp;
                        StringSubStream >> Temp;
                        SunSpot[i + MonthsDayTemp1[j]] = Temp;
                        
                    }
                }
                
                int MonthsDayTemp2[7] = {30,89,150,211,242,303,364};
                
                getline (MyFile,Line);
                istringstream LineStream2(Line);
                
                for (int j = 0 ; j < 7 ; j++)
                {
                    LineStream2 >> SubString;
                    stringstream StringSubStream (SubString);
                    int Temp;
                    StringSubStream >> Temp;
                    SunSpot[MonthsDayTemp2[j]] = Temp;
                    
                }
            }
            
            for (int i = 0; i < Number ; i++)
                SunSpotTotal[Counter + i] = SunSpot[i];
            
            Counter += Number;
            
            MyFile.close();
            delete [] SunSpot;
            
        }
        else
        {
            cout << "Unable to open file \n";
            break;
        }
    }
    
    int d=7 , n=4;
    unsigned EstimatedEmbedding;
    EstimatedEmbedding = embedding_dimension (d , n);
    double EstimatedLyapunov = 0.0;
    double l = 0.0;
    double p = 0.0;
    
    l = lyapunov_estimation(EstimatedEmbedding , EstimatedLyapunov);
    prediction_horizon_estimation(p,l);
    
    double ll = 0.0;
    double pp = 0.0;
    
    vector_type_distance  _LEs;
    vector_type_distance  Estimatedll;
    
    _LEs.resize(d);
    Estimatedll.resize(MatrixSize - 1);
    
    ll = lyapunov_estimation(d , n , _LEs , Estimatedll);
    prediction_horizon_estimation(pp,ll);
    
    
    
    
    
    int InputDimension = EstimatedEmbedding;
    
    int PredictionHorizon = p;
    
    
    int TempArray[InputDimension];
    
    for (int i = 0; i < TotalNumber - InputDimension - PredictionHorizon + 1; i++)
    {
        
        for (int j = 0 ; j < InputDimension ; j++)
            TempArray[j] = SunSpotTotal[i + j];
        int TempOutput = SunSpotTotal[i + InputDimension + PredictionHorizon - 1];
        SSN.push_back(SunSpot(TempArray , TempOutput , PredictionHorizon , InputDimension));
    }
    
    delete [] SunSpotTotal;
    
    ofstream MyFile;
    MyFile.open ("/Users/mirmomeny/Desktop/Checking/MarkovNetworkBinarykData.txt");
    MyFile <<"This file contains the integer representation of sunspot numbers."<<endl;
    MyFile<<"The first (n-1)th integers of each row represent the input SSNs for Markov network, and the last integer represents the corresponding output to those inputs (prediction target)."<<endl;
    MyFile<<"This file contains sunspot number from "<<InitialYear<<" to " <<FinalYear<<"."<<endl;
    
    
    MyFile <<"Number of data in this file is:"<<endl;
    MyFile<<SSN.size()<<endl;
    
    
    SunSpot TempSSN(TempArray);
    std::vector<int> TempInput;
    int TempOutput;
    
    for (int i = 0 ; i <SSN.size() ; i++)
    {
        TempSSN=SSN[i];
        TempInput = TempSSN.GetInput();
        TempOutput = TempSSN.GetOutput();
        
        for (int j = 0 ; j < TempSSN.GetInputDimension() ; j++)
        {
            
            /* int* TempBinary;
             TempBinary = IntToBinary(TempInput[j]);
             
             for (int k = 0; k<9; k++) {
             MyFile <<TempBinary[k]<<" ";
             } */
            
            MyFile << TempInput[j] << " ";
            
            
        }
        
        /* int* TempBinary = IntToBinary(TempOutput);
         
         for (int k = 0; k<8; k++) {
         MyFile <<TempBinary[k]<<" ";
         }*/
        
        MyFile <<TempOutput<<endl;
    }
    
    MyFile.close();
    cout<<"Preprocessing is done :-)!" << endl;
    cout << "Estimated Embedding Dimension  is: " << EstimatedEmbedding <<endl;
    cout << "Estimated Nonlinearity Order   is: " << n <<endl;
    cout << "Estimated Lyapunov Exponent    is: " << EstimatedLyapunov <<endl;
    cout << "Estimated Prediction Horizon   is: " << p <<endl;
    
    cout<<"$$$$$$$$"<<endl;
    
    cout << "Estimated Lyapunov Exponent    is (2): " << ll <<endl;
    cout << "Estimated Prediction Horizon   is (2): " << pp <<endl;
    
    /*for (int i = 0 ; i < Estimatedll.size();i++)
        cout << Estimatedll(i)<<endl;
    */
    /*matrix_type_estimated Q , R , c1 , c2 , TempMatrix;
    Q.resize(3,3);
    R.resize(3,3);
    c1.resize(3,3);
    c2.resize(3,3);
    TempMatrix.resize(3,3);
    
    TempMatrix(0,0)=2;
    TempMatrix(0,1)=2;
    TempMatrix(0,2)=3;
    TempMatrix(1,0)=21;
    TempMatrix(1,1)=-5;
    TempMatrix(1,2)=10;
    TempMatrix(2,0)=2;
    TempMatrix(2,1)=7;
    TempMatrix(2,2)=-2;
    
    QR_factorization(Q , R , TempMatrix);
    c1 = boost::numeric::ublas::prod (boost::numeric::ublas::trans(Q) , Q);
    c2 = boost::numeric::ublas::prod (Q , R);
    
    cout<<Q<<endl;
    cout<<R<<endl;
    cout<<TempMatrix<<endl;
    cout<<c1<<endl;
    cout<<c2<<endl;
    */
    
    
    return 0;
}






