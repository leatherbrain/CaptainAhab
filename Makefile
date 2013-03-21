#*****************************************************************************
# FILE:				Makefile
# DESCRIPTION:			Makefile for slammer library and tests.
# 
# REVISION HISTORY:		AUTHOR:		Sudeep S.
# 						DATE:		03 Nov 2010
# 						COMMENTS:	File created.
#*****************************************************************************/

OBJDIR = ./../lib/
TSTDIR = ./../tst/

CXX = colorgcc
CXX_GPP34 = g++34

FLAGS_REL_PASS_1 = -O3 -pipe -Wall -W -fprofile-arcs -D_FILE_OFFSET_BITS=64 -msse -finline-functions -ffast-math -fPIC
FLAGS_REL_PASS_2 = -O3 -pipe -Wall -W -freorder-functions -fbranch-probabilities -D_FILE_OFFSET_BITS=64 -msse -finline-functions -ffast-math -fPIC
FLAGS_REL_TST = -O3 -pipe -Wall -W -D_FILE_OFFSET_BITS=64 -msse -finline-functions -ffast-math

FLAGS_DEBUG = -g -DDEBUG -D_FILE_OFFSET_BITS=64 -Wall -W -fPIC
FLAGS_DEBUG_TST = -g -DDEBUG -D_FILE_OFFSET_BITS=64

#**********************************************************
# SET DEBUG / RELEASE MODES HERE
#**********************************************************
#
FLAGS_1 = $(FLAGS_REL_PASS_1)
FLAGS_2 = $(FLAGS_REL_PASS_2)
FLAGS_TST = $(FLAGS_REL_TST)
#
# UNCOMMENT THE FOLLOWING LINES FOR DEBUG MODE
#
#FLAGS_1 = $(FLAGS_DEBUG)
#FLAGS_2 = $(FLAGS_DEBUG)
#FLAGS_TST = $(FLAGS_DEBUG_TST)
#
#**********************************************************

INCLUDES = -I
		
LIBPATHS = 
			
LIBS = 

OBJS = $(OBJDIR)example.o
	$(OBJDIR)libmfcc.o
		
TARGET = $(OBJDIR)mfcc_ex

all: tst

$(OBJDIR)example.o: libmfcc/libmfcc_example/example.c
	$(CXX) $(FLAGS_1) -c libmfcc/libmfcc_example/example.c $(INCLUDES) -o $(OBJDIR)example.o
	$(CXX) $(FLAGS_2) -c libmfcc/libmfcc_example/example.c $(INCLUDES) -o $(OBJDIR)example.o

$(OBJDIR)libmfcc.o: libmfcc/libmfcc.c libmfcc/libmfcc.h
	$(CXX) $(FLAGS_1) -c libmfcc/libmfcc.c $(INCLUDES) -o $(OBJDIR)libmfcc.o
	$(CXX) $(FLAGS_2) -c libmfcc/libmfcc.c $(INCLUDES) -o $(OBJDIR)libmfcc.o

lib: $(OBJS)
	rm -f $(TARGET).so*
	ar rcv $(TARGET).a $(OBJS)
	ranlib $(TARGET).a
	cp $(TARGET).a $(TARGET).so.1.0.0
	ln -s $(TARGET).so.1.0.0 $(TARGET).so.1.0
	ln -s $(TARGET).so.1.0.0 $(TARGET).so.1
	ln -s $(TARGET).so.1.0.0 $(TARGET).so
	
tst: TestOptFlow TestProjection TestBgSub

TestOptFlow: $(TSTDIR)TestOptFlow.cpp ../inc/OptFlow.h
	$(CXX) $(FLAGS_TST) -c $(TSTDIR)TestOptFlow.cpp $(INCLUDES) -o $(OBJDIR)TestOptFlow.o
	$(CXX) $(OBJDIR)TestOptFlow.o $(LIBPATHS) $(LIBS) -o $(OBJDIR)OptFlow

TestProjection: $(TSTDIR)TestProjection.cpp ../inc/Projection.h
	$(CXX) $(FLAGS_TST) -c $(TSTDIR)TestProjection.cpp $(INCLUDES) -o $(OBJDIR)TestProjection.o
	$(CXX) $(OBJDIR)TestProjection.o $(LIBPATHS) $(LIBS) -o $(OBJDIR)Projection

TestBgSub: $(TSTDIR)TestBgSub.cpp ../inc/BgSub.h ../inc/OptFlow.h ../inc/Projection.h
	$(CXX) $(FLAGS_TST) -c $(TSTDIR)TestBgSub.cpp $(INCLUDES) -o $(OBJDIR)TestBgSub.o
	$(CXX) $(OBJDIR)TestBgSub.o $(LIBPATHS) $(LIBS) -o $(OBJDIR)BgSub

Visualise: $(TSTDIR)Visualise.cpp ../inc/BgSub.h ../inc/OptFlow.h ../inc/Projection.h
	$(CXX) $(FLAGS_TST) -c $(TSTDIR)Visualise.cpp $(INCLUDES) -o $(OBJDIR)Visualise.o
	$(CXX) $(OBJDIR)Visualise.o $(LIBPATHS) $(LIBS) -o $(OBJDIR)Visualise

clean:
	rm -f $(OBJDIR)*.o $(OBJDIR)*.gcno $(TSTDIR)*.o $(TARGET)

