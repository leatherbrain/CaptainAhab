CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		CaptainAhab.o

CPPS =		CaptainAhab.cpp

LIBPATH =	-L/usr/local/lib

INCPATH =	-I/usr/local/include

LIBS =		-lfftw3 -lm -laiff

TARGET =	CaptainAhab

$(TARGET):	$(OBJS)
	$(CXX) -static $(LIBPATH) $(INCPATH) $(CPPS) $(LIBS) -o $(TARGET) 

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
