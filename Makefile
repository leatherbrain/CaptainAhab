CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		CaptainAhab.o WhaleSoundRecognition.o FeatureVector.o

CPPS =		CaptainAhab.cpp WhaleSoundRecognition.cpp FeatureVector.cpp

LIBPATH =	-L/usr/local/lib -L/usr/lib

INCPATH =	-I/usr/local/include -I/usr/include

LIBS =		-lxtract -lfftw3 -lm -laiff -lboost_system -lboost_regex -lboost_filesystem -lpthread

TARGET =	CaptainAhab

$(TARGET):	$(OBJS)
	$(CXX) -static $(LIBPATH) $(INCPATH) $(CPPS) $(LIBS) -o $(TARGET) 

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
