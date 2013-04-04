/*
 * WhaleSoundRecognition.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "WhaleSoundRecognition.h"
#include "FeatureVector.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#define LIBAIFF_NOCOMPAT 1
extern "C" {
#include "libaiff/libaiff.h"
}

#include "xtract/libxtract.h"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include <boost/shared_array.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "HelperFunctions.h"

using namespace boost::filesystem;

using namespace std;

const int numEltsInHistogram = 64;
int numPerClipFeatures = 0;

WhaleSoundRecognition::WhaleSoundRecognition() {
	// TODO Auto-generated constructor stub
	float priors[2];
	priors[0] = 0.5;
	priors[1] = 0.5;

	params = new CvRTParams(100, 5, 0.2, false, 2, 0, true, 0, 200, 0.1,
			CV_TERMCRIT_ITER);

	statModel = new CvRTrees();
	//statModel = new CvSVM();
}

WhaleSoundRecognition::~WhaleSoundRecognition() {
	// TODO Auto-generated destructor stub
	delete statModel;
	delete params;
}

//============================================================================
// Purpose     : Set paths to training and testing data
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::Init(const string & pathToTrainData,
		const string & trainDataLabelsFilename, const string & pathToTestData) {
	pathToTrainingData_ = pathToTrainData;
	pathToTestData_ = pathToTestData;
	trainDataLabelsFilename_ = trainDataLabelsFilename;
}

//============================================================================
// Purpose     : For each feature vector passed in determine which cluster
//				centre it is closest to and increment the count in the histogram
//				for that centre.
// Author      : Michael King
//============================================================================
float * WhaleSoundRecognition::GetHistogramOfFrameFeatures(
		const vector<float *> & f) {
	float * histogram = new float[numEltsInHistogram];
	for (int i = 0; i < numEltsInHistogram; ++i) {
		histogram[i] = 0.0;
	}
	for (size_t i = 0; i < f.size(); ++i) {
		float minDistance = 10000.0;
		int label = 0;
		for (int j = 0; j < centres.rows; ++j) {
			float currentDistance = 0;
			for (int k = 0; k < centres.cols; ++k) {
				float diff = centres.at<float>(j, k) - f[i][k];
				currentDistance += diff * diff;
			}
			//cout << "current distance = " << currentDistance << endl;
			if (currentDistance < minDistance) {
				minDistance = currentDistance;
				label = j;
			}
		}
		histogram[label] += 1.0;
	}
	return histogram;
}

// class generator:
struct c_unique {
	int current;
	c_unique() {
		current = 0;
	}
	int operator()() {
		return current++;
	}
} UniqueNumber;

//============================================================================
// Purpose     : Print an array to console
// Author      : Michael King
//============================================================================
template<typename T>
void PrintArray(T * array, int numElts) {
	for (int i = 0; i < numElts; ++i) {
		cout << array[i] << " ";
	}
	cout << endl;
}

//============================================================================
// Purpose     : Read the training data csv and cycle through the training data
//				to extract features and train with it. Optionally leave some
//				proportion out in order to do a mock test with them.
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::Train() {
	double proportionTrainingDataToUseForTesting = 0.25;
	trainingData.clear();

	ifstream trainDataFile(trainDataLabelsFilename_.c_str(), ios_base::in);

	string currentLine;
	getline(trainDataFile, currentLine);
	int lineNo = 0;
	while (trainDataFile.good() && !trainDataFile.eof()) {
		if (++lineNo % 1000 == 0)
			cout << lineNo << endl;
		getline(trainDataFile, currentLine);
		if (trainDataFile.good() && !trainDataFile.eof()) {
			int commaPos = currentLine.find_first_of(',');
			string filename = currentLine.substr(0, commaPos);
			bool containsWhaleSound = currentLine[commaPos + 1] == '1';
			if ((double) rand() / RAND_MAX
					< proportionTrainingDataToUseForTesting) {
				AddTestingData(pathToTrainingData_ + filename,
						containsWhaleSound);
			} else {
				AddTrainingData(pathToTrainingData_ + filename,
						containsWhaleSound);
			}
		}
	}

	// Cluster the per frame training data
	int totalPerFrameVectors = 0;
	for (size_t i = 0; i < soundClipsPerFrameVector.size(); ++i) {
		totalPerFrameVectors += soundClipsPerFrameVector[i].size();
	}
	cout << "calculated total = " << totalPerFrameVectors << endl;
	cv::Mat allPerFrameData(40000, 39, CV_32FC1);
	vector<int> index(totalPerFrameVectors, 0);
	generate(index.begin(), index.end(), UniqueNumber);
	random_shuffle(index.begin(), index.end());
	index.resize(40000);
	for (size_t k = 0; k < index.size(); ++k) {
		int i = index[k] / soundClipsPerFrameVector.front().size();
		int j = index[k] % soundClipsPerFrameVector.front().size();
		cv::Mat(1, 39, CV_32FC1, soundClipsPerFrameVector[i][j]).copyTo(
				allPerFrameData(cv::Range(k, k + 1), cv::Range::all()));
	}
	cout << "finished copying data" << endl;
	cv::Mat labels;
	cv::TermCriteria criteria;
	criteria.epsilon = 0.5;

	cout << "kmeans ..." << endl;
	cv::kmeans(allPerFrameData, numEltsInHistogram, labels, criteria, 1,
			cv::KMEANS_RANDOM_CENTERS, centres);
	cout << " done" << endl;

	assert(soundClipsPerFrameVector.size() == soundClipsWholeClipVector.size());
	for (size_t i = 0; i < soundClipsPerFrameVector.size(); ++i) {
		if (i % 1000 == 0)
			cout << "i = " << i << endl;
		float * histogram = GetHistogramOfFrameFeatures(
				soundClipsPerFrameVector[i]);
		float * wholeClipFeatures = soundClipsWholeClipVector[i];
		float * overallArray;
		int overallArraySize;
		ConcatenateArrays<float>(histogram, numEltsInHistogram,
				wholeClipFeatures, numPerClipFeatures, overallArray,
				overallArraySize);
		FeatureVector f(overallArray, overallArraySize);
		delete[] histogram;
		trainingData.push_back(f);
	}

	for (size_t i = 0; i < soundClipsPerFrameVector.size(); ++i) {
		for (size_t j = 0; j < soundClipsPerFrameVector[i].size(); ++j) {
			delete[] soundClipsPerFrameVector[i][j];
		}
	}
	for (size_t i = 0; i < soundClipsWholeClipVector.size(); ++i) {
		delete[] soundClipsWholeClipVector[i];
	}
	cv::Mat allTrainingData(trainingData.size(),
			trainingData.front().data_.cols, CV_32FC1);
	cv::Mat allTrainingLabels(trainingData.size(), 1, CV_32SC1);
	for (size_t i = 0; i < trainingData.size(); ++i) {
		trainingData[i].data_.copyTo(
				allTrainingData(cv::Range(i, i + 1), cv::Range::all()));
		allTrainingLabels.at<int>(i, 0) = trainingDataLabels[i] ? 1 : 0;
	}
	trainingData.clear();
	trainingDataLabels.clear();

	cout << "calling train" << endl;
	statModel->train(allTrainingData, CV_ROW_SAMPLE, allTrainingLabels,
			cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), *params);
	//statModel->train(allTrainingData, allTrainingLabels);

}

//============================================================================
// Purpose     : Add data to perform a mock test with
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::AddTestingData(const std::string & filename,
		bool containsWhaleSound) {
	mockTestFilenames.push_back(filename);
	mockTestGroundTruth.push_back(containsWhaleSound);
}

//============================================================================
// Purpose     : Perform a mock test using training data
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::MockTest() {
	cout << "mock test" << endl;
	int num0sCorrect = 0, num1sCorrect = 0, num0s = 0, num1s = 0;
	for (size_t i = 0; i < mockTestFilenames.size(); ++i) {
		vector<float *> f = GetPerFrameFeatures(mockTestFilenames[i]);
		// Classify the sequence f
		float * hist = GetHistogramOfFrameFeatures(f);
		for (size_t j = 0; j < f.size(); ++j) {
			delete f[j];
		}
		float * wholeClipFeatures = GetWholeClipFeatures(mockTestFilenames[i]);
		float * overallArray;
		int sizeOfOverallArray;
		ConcatenateArrays<float>(hist, numEltsInHistogram, wholeClipFeatures,
				numPerClipFeatures, overallArray, sizeOfOverallArray);
		FeatureVector f2(overallArray, sizeOfOverallArray);
		delete[] hist;

		float result = statModel->predict(f2.data_);
		if (mockTestGroundTruth[i]) {
			++num1s;
			if (fabs((double) result - 1.0) < 0.01) {
				++num1sCorrect;
			}
		} else {
			++num0s;
			if (fabs((double) result - 0.0) < 0.01) {
				++num0sCorrect;
			}
		}
	}
	double proportion0s = (double) num0sCorrect / num0s;
	double proportion1s = (double) num1sCorrect / num1s;
	cout << num0sCorrect << " 0s correct out of " << num0s << " "
			<< proportion0s << endl;
	cout << num1sCorrect << " 1s correct out of " << num1s << " "
			<< proportion1s << endl;
	cout << "result on mock test = " << (proportion0s + proportion1s) / 2.0
			<< endl;
}

//============================================================================
// Purpose     : Collect results on the test data
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::Test() {
	// Get all filenames ending .aiff
	path current_dir(pathToTestData_);
	boost::regex pattern("test[0-9]+.aiff");
	for (recursive_directory_iterator iter(current_dir), end; iter != end;
			++iter) {
		string name = iter->path().leaf().string();
		if (regex_match(name, pattern)) {
			vector<float *> f = GetPerFrameFeatures(iter->path().string());
			// Classify the sequence f
			float * hist = GetHistogramOfFrameFeatures(f);
			for (size_t i = 0; i < f.size(); ++i) {
				delete f[i];
			}
			float * wholeClipFeatures = GetWholeClipFeatures(
					iter->path().string());
			float * overallArray;
			int sizeOfOverallArray;
			ConcatenateArrays<float>(hist, numEltsInHistogram,
					wholeClipFeatures, numPerClipFeatures, overallArray,
					sizeOfOverallArray);
			FeatureVector f2(overallArray, sizeOfOverallArray);
			delete[] hist;
			float result = statModel->predict(f2.data_);
			cout << "result = " << result << endl;
		}
	}
}

//============================================================================
// Purpose     : Add data to train the classifiers
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::AddTrainingData(const std::string & filename,
		bool containsWhaleSound) {
	vector<float *> f = GetPerFrameFeatures(filename);
	soundClipsPerFrameVector.push_back(f);
	float * f2 = GetWholeClipFeatures(filename);
	soundClipsWholeClipVector.push_back(f2);

	trainingDataLabels.push_back(containsWhaleSound);

}

//============================================================================
// Purpose     : Get features on short frames of the sequence
// Author      : Michael King
//============================================================================
vector<float *> WhaleSoundRecognition::GetPerFrameFeatures(
		const std::string& filename) {
	double* samples;
	int nSamples;
	GetRawSamples(filename, samples, nSamples);

	// Num frames we can use is num frames - 8
	// Split into frames
	int numSamplesPerFrame = 50;
	int numSamplesPowerOf2 = GetLowestPowerOf2GreaterThanN(numSamplesPerFrame);
	int numSamplesInJump = 20;
	int currentFrameBegin = 0;
	int currentFrameEnd = numSamplesPerFrame;
	vector<double *> perFrameMFCCs;
	while (currentFrameEnd < nSamples) {
		// Use Hamming window
		double samplesHamming[numSamplesPowerOf2];
		const double alpha = 0.46;
		for (int i = 0; i < numSamplesPerFrame; ++i) {
			samplesHamming[i] = samples[currentFrameBegin + i] * (1 - alpha)
					- alpha * cos(2 * 3.14159 * i / (numSamplesPerFrame - 1));
		}
		for (int i = numSamplesPerFrame; i < numSamplesPowerOf2; ++i) {
			samplesHamming[i] = 0.0;
		}

		double spectrum[numSamplesPowerOf2];
		const int SAMPLERATE = 44100;
		GetSpectrum(SAMPLERATE, numSamplesPowerOf2, samplesHamming, spectrum);

		perFrameMFCCs.push_back(NULL);
		ExtractMFCCs(currentFrameEnd - currentFrameBegin, SAMPLERATE, spectrum,
				perFrameMFCCs.back());
		perFrameMFCCs.back()[0] = (float) CalculateLogEnergy(samplesHamming,
				numSamplesPerFrame);

		currentFrameEnd += numSamplesInJump;
		currentFrameBegin += numSamplesInJump;
	}

	if (perFrameMFCCs.size() < 9) {
		cout << "Throwing runtime error" << endl;
		throw std::runtime_error("too few frames in clip");
	}

	vector<double *> deltas(perFrameMFCCs.size(), NULL);
	vector<double *> deltaDeltas(perFrameMFCCs.size(), NULL);
	for (size_t i = 2; i < perFrameMFCCs.size() - 2; ++i) {
		deltas[i] = new double[13];
		for (size_t j = 0; j < 13; ++j) {
			deltas[i][j] = (perFrameMFCCs[i + 1][j] - perFrameMFCCs[i - 1][j]
					+ 2 * (perFrameMFCCs[i + 2][j] - perFrameMFCCs[i - 2][j]))
					/ 10.0;
		}
	}

	vector<float *> perFrameFeatures;
	for (size_t i = 4; i < perFrameMFCCs.size() - 4; ++i) {
		deltaDeltas[i] = new double[13];
		float * perFrameFeature = new float[39];
		for (size_t j = 0; j < 13; ++j) {
			deltaDeltas[i][j] = (deltas[i + 1][j] - deltas[i - 1][j]
					+ 2 * (deltas[i + 2][j] - deltas[i - 2][j])) / 10.0;
			perFrameFeature[j] = (float) perFrameMFCCs[i][j];
			perFrameFeature[j + 13] = (float) deltas[i][j];
			perFrameFeature[j + 26] = (float) deltaDeltas[i][j];
		}
		perFrameFeatures.push_back(perFrameFeature);
	}

	for (size_t i = 0; i < perFrameMFCCs.size(); ++i) {
		delete[] perFrameMFCCs[i];
		delete[] deltas[i];
		delete[] deltaDeltas[i];
	}

	delete[] samples;
	return perFrameFeatures;
}

//============================================================================
// Purpose     : Get the raw sound from the aiff file
// Author      : Michael King
// Note		   : Caller should use delete [] samples
//============================================================================
void WhaleSoundRecognition::GetRawSamples(const std::string & filename,
		double * & samples, int & nSamples) {
	AIFF_Ref ref = AIFF_OpenFile(filename.c_str(), F_RDONLY);

	if (!ref) {
		cout << "Failed to open " << filename << endl;
		return;
	}

	nSamples = ref->nSamples;

	// Our AIFF files have audio format LPCM and 16 bits per sample.
	int16_t * samplesI = new int16_t[nSamples];
	int bitsPerSample = ref->bitsPerSample;
	int numBytesRead = AIFF_ReadSamples(ref, samplesI,
			nSamples * bitsPerSample / 8);
	assert(numBytesRead == nSamples * bitsPerSample / 8);
	// Convert to doubles to use with the fftw
	samples = new double[nSamples];
	for (int i = 0; i < nSamples; ++i) {
		samples[i] = (double) samplesI[i];
	}
	delete[] samplesI;

	AIFF_CloseFile(ref);
}

//============================================================================
// Purpose     : Get the frequency spectrum
// Author      : Michael King
//============================================================================
void WhaleSoundRecognition::GetSpectrum(const int SAMPLERATE, int nSamples,
		double* input, double * spectrum) {
	double argd[4];
	argd[0] = (double) (SAMPLERATE) / (double) (nSamples);
	argd[1] = (double) (XTRACT_MAGNITUDE_SPECTRUM);
	argd[2] = 0.f; /* No DC component */
	argd[3] = 0.f; /* No Normalisation */
	xtract_init_fft(2 * nSamples, XTRACT_SPECTRUM);
	xtract_spectrum(input, nSamples, &argd[0], spectrum);
}

//============================================================================
// Purpose     : Extract MFCCs
// Author      : Michael King
//============================================================================
const int WhaleSoundRecognition::ExtractMFCCs(int nSamples,
		const int SAMPLERATE, double * spectrum, double*& mfccs) {

	const int MFCC_FREQ_BANDS = 13;
	mfccs = new double[MFCC_FREQ_BANDS];
	int n;
	xtract_mel_filter mel_filters;
	mel_filters.n_filters = MFCC_FREQ_BANDS;
	mel_filters.filters =
			(double**) (malloc(MFCC_FREQ_BANDS * sizeof(double*)));
	for (n = 0; n < MFCC_FREQ_BANDS; ++n) {
		mel_filters.filters[n] = (double*) (malloc(nSamples * sizeof(double)));
	}
	const int MFCC_FREQ_MIN = 20;
	const int MFCC_FREQ_MAX = 20000;
	xtract_init_mfcc(nSamples >> 1, SAMPLERATE >> 1, XTRACT_EQUAL_GAIN,
			MFCC_FREQ_MIN, MFCC_FREQ_MAX, mel_filters.n_filters,
			mel_filters.filters);
	xtract_mfcc(spectrum, nSamples >> 1, &mel_filters, mfccs);
	for (n = 0; n < MFCC_FREQ_BANDS; ++n) {
		free(mel_filters.filters[n]);
	}
	free(mel_filters.filters);

	bool lifter = false;
	if (lifter) {
		for (int i = 0; i < MFCC_FREQ_BANDS; ++i) {
			mfccs[i] *= 1.0
					+ double(nSamples / 2) * sin(3.14159 * i / nSamples);
		}
	}

	return MFCC_FREQ_BANDS;
}

//============================================================================
// Purpose     : Calculate the log energy
// Author      : Michael King
//============================================================================
double WhaleSoundRecognition::CalculateLogEnergy(double * input, int nSamples) {
	double sum = 0.0;
	for (int i = 0; i < nSamples; ++i) {
		sum += input[i] * input[i];
	}
	return log(sum);
}

//============================================================================
// Purpose     : Get features from the whole clip
// Author      : Michael King
//============================================================================
float * WhaleSoundRecognition::GetWholeClipFeatures(
		const std::string& filename) {

	double* samples2;
	int nSamples;
	GetRawSamples(filename, samples2, nSamples);
	int nSamplesPowerOf2 = GetLowestPowerOf2GreaterThanN(nSamples);
	double* samples = new double[nSamplesPowerOf2];
	std::copy(samples2, samples2 + nSamples, samples);
	for (int i = nSamples; i < nSamplesPowerOf2; ++i) {
		samples[i] = 0.0;
	}
	delete[] samples2;
	nSamples = nSamplesPowerOf2;

	numPerClipFeatures = 0;

	double spectrum[nSamples];
	const int SAMPLERATE = 44100;
	GetSpectrum(SAMPLERATE, nSamples, samples, spectrum);

	// 0 to 3840
	double * frameMFCCs1, logEnergy1;
	const int MFCC_FREQ_BANDS = ExtractMFCCs(3840, SAMPLERATE, spectrum + 0,
			frameMFCCs1);
	logEnergy1 = (float) CalculateLogEnergy(samples + 0, 3840);
	// 20 to 3860
	double * frameMFCCs2, logEnergy2;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 20, frameMFCCs2);
	logEnergy2 = CalculateLogEnergy(samples + 20, 3840);
	// 40 to 3880
	double * frameMFCCs3, logEnergy3;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 40, frameMFCCs3);
	logEnergy3 = CalculateLogEnergy(samples + 40, 3840);
	// 60 to 3900
	double * frameMFCCs4, logEnergy4;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 60, frameMFCCs4);
	logEnergy4 = CalculateLogEnergy(samples + 60, 3840);
	// 80 to 3920
	double * frameMFCCs5, logEnergy5;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 80, frameMFCCs5);
	logEnergy5 = CalculateLogEnergy(samples + 80, 3840);
	// 100 to 3940
	double * frameMFCCs6, logEnergy6;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 100, frameMFCCs6);
	logEnergy6 = CalculateLogEnergy(samples + 100, 3840);
	// 120 to 3960
	double * frameMFCCs7, logEnergy7;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 120, frameMFCCs7);
	logEnergy7 = CalculateLogEnergy(samples + 120, 3840);
	// 140 to 3980
	double * frameMFCCs8, logEnergy8;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 140, frameMFCCs8);
	logEnergy8 = CalculateLogEnergy(samples + 140, 3840);
	// 160 to 4000
	double * frameMFCCs9, logEnergy9;
	ExtractMFCCs(3840, SAMPLERATE, spectrum + 160, frameMFCCs9);
	logEnergy9 = CalculateLogEnergy(samples + 160, 3840);

	float data[3 * MFCC_FREQ_BANDS];
	data[0] = logEnergy5;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		data[i] = (float) frameMFCCs5[i];
	}
	float delta3[MFCC_FREQ_BANDS], delta4[MFCC_FREQ_BANDS],
			delta6[MFCC_FREQ_BANDS], delta7[MFCC_FREQ_BANDS];
	data[MFCC_FREQ_BANDS] = ((logEnergy6 - logEnergy4)
			+ 2 * (logEnergy7 - logEnergy3)) / 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		data[MFCC_FREQ_BANDS + i] = ((frameMFCCs6[i] - frameMFCCs4[i])
				+ 2 * (frameMFCCs7[i] - frameMFCCs3[i])) / 10.0;
		;
	}
	delta3[0] = ((logEnergy4 - logEnergy2) + 2 * (logEnergy5 - logEnergy1))
			/ 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		delta3[i] = ((frameMFCCs4[i] - frameMFCCs2[i])
				+ 2 * (frameMFCCs5[i] - frameMFCCs1[i])) / 10.0;
		;
	}
	delta4[0] = ((logEnergy5 - logEnergy3) + 2 * (logEnergy6 - logEnergy2))
			/ 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		delta4[i] = ((frameMFCCs5[i] - frameMFCCs3[i])
				+ 2 * (frameMFCCs6[i] - frameMFCCs2[i])) / 10.0;
		;
	}
	delta6[0] = ((logEnergy7 - logEnergy5) + 2 * (logEnergy8 - logEnergy4))
			/ 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		delta6[i] = ((frameMFCCs7[i] - frameMFCCs5[i])
				+ 2 * (frameMFCCs8[i] - frameMFCCs4[i])) / 10.0;
		;
	}
	delta7[0] = ((logEnergy8 - logEnergy6) + 2 * (logEnergy9 - logEnergy5))
			/ 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		delta7[i] = ((frameMFCCs8[i] - frameMFCCs6[i])
				+ 2 * (frameMFCCs9[i] - frameMFCCs5[i])) / 10.0;
		;
	}
	data[2 * MFCC_FREQ_BANDS] = ((logEnergy4 - logEnergy2)
			+ 2 * (logEnergy5 - logEnergy1)) / 10.0;
	for (int i = 1; i < MFCC_FREQ_BANDS; ++i) {
		data[2 * MFCC_FREQ_BANDS + i] = ((delta6[i] - delta4[i])
				+ 2 * (delta7[i] - delta3[i])) / 10.0;
		;
	}

	delete[] frameMFCCs1;
	delete[] frameMFCCs2;
	delete[] frameMFCCs3;
	delete[] frameMFCCs4;
	delete[] frameMFCCs5;

	int numExtraFeatures = 13;
	double extraFeatures[numExtraFeatures];
	float extraFeaturesF[numExtraFeatures];
//	xtract_mean(spectrum, nSamples, NULL, extraFeatures);
//	xtract_variance (spectrum, nSamples, extraFeatures, extraFeatures + 1);
////	 	Extract the deviation of an input vector.
// 	xtract_average_deviation (nSamples, const int N, const void *argv, NULL)
////	 	Extract the average deviation of an input vector.
// 	xtract_skewness (const float *data, const int N, const void *argv, float *result)
////	 	Extract the skewness of an input vector.
// 	xtract_kurtosis (const float *data, const int N, const void *argv, float *result)
////	 	Extract the kurtosis of an input vector.
// 	xtract_spectral_mean (const float *data, const int N, const void *argv, float *result)
////	 	Extract the mean of an input spectrum.
// 	xtract_spectral_variance (const float *data, const int N, const void *argv, float *result)
////	 	Extract the variance of an input spectrum.
// 	xtract_spectral_standard_deviation (const float *data, const int N, const void *argv, float *result)
////	 	Extract the deviation of an input spectrum.
// 	xtract_spectral_average_deviation (const float *data, const int N, const void *argv, float *result)
////	 	Extract the average deviation of an input spectrum.
// 	xtract_spectral_skewness (const float *data, const int N, const void *argv, float *result)
////	 	Extract the skewness of an input spectrum.
// 	xtract_spectral_kurtosis (const float *data, const int N, const void *argv, float *result)
////	 	Extract the kurtosis of an input spectrum.
// 	xtract_spectral_centroid (const float *data, const int N, const void *argv, float *result)
////	 	Extract the centroid of an input vector.
	int featureNumber = 0;
	xtract_irregularity_k (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_irregularity_j (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_tristimulus_1 (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_tristimulus_2 (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_tristimulus_3 (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_smoothness (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	//xtract_spread (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//cout << __LINE__ << endl;
	xtract_zcr (samples, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_rolloff (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	// Requires bark coefficients
	//xtract_loudness (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_flatness (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_tonality (spectrum, nSamples, extraFeatures + featureNumber - 1, extraFeatures + featureNumber++);
	//xtract_noisiness (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_rms_amplitude (samples, nSamples, NULL, extraFeatures + featureNumber++);
	// Requires spectral peaks
	//xtract_spectral_inharmonicity (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_crest (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_power (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_odd_even_ratio (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_sharpness (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_spectral_slope (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	//xtract_hps (spectrum, nSamples, NULL, extraFeatures + featureNumber++);
	float sampleRate = 2000;
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_f0 (samples, nSamples, &sampleRate, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;
	xtract_failsafe_f0 (samples, nSamples, &sampleRate, extraFeatures + featureNumber++);
	if (isnan(extraFeatures[featureNumber - 1]) || extraFeatures[featureNumber - 1] > 1e7 || extraFeatures[featureNumber - 1]  < -1e7) extraFeatures[featureNumber - 1] = 0.0;

	for (int i=0; i<numExtraFeatures; ++i)
	{
		extraFeaturesF[i] = extraFeatures[i];
	}

	float * overallData;
	int overallSize;
	ConcatenateArrays<float>(data, 3*MFCC_FREQ_BANDS, extraFeaturesF, numExtraFeatures,
			overallData, overallSize);

	numPerClipFeatures = 3*MFCC_FREQ_BANDS + numExtraFeatures;

	delete [] samples;
	return overallData;
}
