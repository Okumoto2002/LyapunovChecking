//
//  SunSpot.cpp
//  Checking
//
//  Created by Masoud Mirmomeni on 5/20/13.
//  Copyright (c) 2013 Masoud Mirmomeni. All rights reserved.
//

#include "SunSpot.h"
#include <iomanip>

SunSpot::SunSpot(SunSpot const& ssn)
{
    Input = ssn.GetInput();
    Output = ssn.GetOutput();
    PredictionHorizon = ssn.GetPredictionHorizon();
    InputDimension = ssn.GetInputDimension();
    
}

SunSpot& SunSpot::operator=(SunSpot const& ssn)
{
    
    Input = ssn.GetInput();
    Output = ssn.GetOutput();
    PredictionHorizon = ssn.GetPredictionHorizon();
    InputDimension = ssn.GetInputDimension();
    return *this;
    
}

ostream& operator<<( ostream& Out, const SunSpot& ssn )
{
    vector<int>::iterator it;
    vector<int> TempSSN;
    for (int i = 0; i <ssn.InputDimension;i++)
        TempSSN.push_back(ssn.GetInput()[i]);
    Out << "myvector contains:";
    for ( it=TempSSN.begin() ; it < TempSSN.end(); it++ )
        Out << " " << *it;
    
    Out<<endl;
    Out << "Output is: " << ssn.Output <<endl;
    Out<<"Prediction horizon is: " <<ssn.PredictionHorizon<<endl;
    Out<< "Input dimension is: " << ssn.InputDimension <<endl;
    return Out;
}


int* IntToBinary(int Number)
{
    int* TempBinary = new int[9];
    for (int i = 0; i < 9; i++) {
        TempBinary[8 - i] = Number % 2;
        Number = Number / 2;
    }
    return TempBinary;
}