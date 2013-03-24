//============================================================================
// Name        : CaptainAhab.cpp
// Author      : Michael King and Sudeep Sundaram
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include "WhaleSoundRecognition.h"

int main() {
	WhaleSoundRecognition w;
	w.Init("/home/michael/Documents/Kaggle/Whale sounds/data/train/",
			"/home/michael/Documents/Kaggle/Whale sounds/data/train.csv",
			"/home/michael/Documents/Kaggle/Whale sounds/data/test/");
	w.Train();
	return 0;
}
