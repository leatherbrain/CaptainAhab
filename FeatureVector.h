/*
 * FeatureVector.h
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#ifndef FEATUREVECTOR_H_
#define FEATUREVECTOR_H_

#include <opencv/cv.h>

class FeatureVector {
public:
	FeatureVector(float * data, int n);
	~FeatureVector();

	cv::Mat data_;
};

#endif /* FEATUREVECTOR_H_ */
