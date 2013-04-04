/*
 * FeatureVector.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "FeatureVector.h"
#include <iostream>

using namespace std;

FeatureVector::FeatureVector(float * data, int n) {
	cv::Mat(1, n, CV_32FC1, data).copyTo(data_);
}

FeatureVector::~FeatureVector() {
	// TODO Auto-generated destructor stub
}

