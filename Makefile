CXXFLAGS =	-g -O2 -Wall -fmessage-length=0

OBJS =		CaptainAhab.o WhaleSoundRecognition.o FeatureVector.o hmm.o

CPPS =		CaptainAhab.cpp WhaleSoundRecognition.cpp FeatureVector.cpp hmm.c

LIBPATH =	-L/usr/local/lib -L/usr/lib

INCPATH =	-I/usr/local/include -I/usr/local/include/opencv -I/usr/include -I/home/michael/Documents/Kaggle/Code/CaptainAhab

LIBS =		-llapack -lblas -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml \
			-lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect \
			-lopencv_contrib -lopencv_legacy -lopencv_flann -lxtract -lfftw3 -lm \
			-laiff -lboost_system -lboost_regex -lboost_filesystem -lpthread

TARGET =	CaptainAhab

$(TARGET):	$(OBJS)
	$(CXX) $(LIBPATH) $(INCPATH) $(CPPS) $(LIBS) -o $(TARGET) 

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

