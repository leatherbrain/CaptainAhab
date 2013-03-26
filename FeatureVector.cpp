/*
 * FeatureVector.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "FeatureVector.h"

FeatureVector::FeatureVector(float * data, int n) {
	data_ = cv::Mat(1, n, CV_32FC1, data);

}

FeatureVector::~FeatureVector() {
	// TODO Auto-generated destructor stub
}

