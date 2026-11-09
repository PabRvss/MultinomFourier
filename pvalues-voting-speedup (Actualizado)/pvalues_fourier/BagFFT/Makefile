CPP = g++-15
#CPPFLAGS = -g -O3 -Winline --param large-function-growth=10000 --param max-inline-insns-single=10000 --param inline-unit-growth=10000 -Wno-deprecated -m32
CPPFLAGS = -g -O3 -Winline --param large-function-growth=10000 --param max-inline-insns-single=10000 --param inline-unit-growth=10000 -Wno-deprecated
NR = 0

FFTW = /opt/homebrew/lib/

ifeq ($(NR), 1)
LIBS = -lm
else
LIBS = -lm -L $(FFTW) -lfftw3
endif

BINDIR = .

ME_OBJS = motif_evaluator.o theta_fns.o minimize.o convolution.o fft.o utils.o llr_score.o
HS_OBJS = HS-test.cpp HS.o utils.o llr_score.o
BAGFFT_OBJS = bagFFT-test.cpp bagFFT.o theta_fns.o minimize.o convolution.o fft.o utils.o llr_score.o
ENTROPY_OBJS = entropy-test.cpp entropy-combined.o LD.o entropy.o convolution.o HS.o bagFFT.o theta_fns.o minimize.o fft.o utils.o llr_score.o
MW_OBJS = mw-test.cpp mw.o convolution.o theta_fns.o minimize.o fft.o utils.o
EXES = HSarray HSlist HSsingle bagFFT_array bagFFT_single cbagFFT NE_NCpruned HS_NCpruned HS_sFFT HS_csFFT bag_sFFT LD mw_sFFT mw_csFFT mw_NA mw_LD

.cpp.o:
	$(CPP) -D NR=$(NR) $(CPPFLAGS) -c $<

depend:	
	$(CPP) $(CPPFLAGS) -MM *.cpp > depend 

clean:
	-rm *~ *.o

find-best: find-best.cpp
	$(CPP) $(CPPFLAGS) find-best.cpp -lm -o $(BINDIR)/$@

motif-find:	motif-find.cpp $(ME_OBJS)
	$(CPP) $(CPPFLAGS) motif-find.cpp $(ME_OBJS) $(LIBS) -o $(BINDIR)/$@

MEtest:	me-test.cpp $(ME_OBJS)
	$(CPP) $(CPPFLAGS) me-test.cpp $(ME_OBJS) $(LIBS) -o $(BINDIR)/$@

HSarray:	$(HS_OBJS)
	$(CPP) $(CPPFLAGS) -D ARRAY $(HS_OBJS) -lm -o $(BINDIR)/$@

HSlist:	$(HS_OBJS)
	$(CPP) $(CPPFLAGS) -D LIST $(HS_OBJS) -lm -o $(BINDIR)/$@

HSsingle:	$(HS_OBJS)
	$(CPP) $(CPPFLAGS) -D SINGLE $(HS_OBJS) -lm -o $(BINDIR)/$@

HSmultiarray:	$(HS_OBJS)
	$(CPP) $(CPPFLAGS) -D MULTIARRAY $(HS_OBJS) -lm -o $(BINDIR)/$@

bagFFT_%:	$(BAGFFT_OBJS)
		$(CPP) $(CPPFLAGS) -D BAGFFT_$* $(BAGFFT_OBJS) $(LIBS) -o $(BINDIR)/$@

cbagFFT:	$(BAGFFT_OBJS)
	$(CPP) $(CPPFLAGS) -D CBAGFFT $(BAGFFT_OBJS) $(LIBS) -o $(BINDIR)/$@

LD:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D LD $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

stitched_LD_csFFT:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D STITCHED $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

NE_NCpruned:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D NE_NCPRUNED $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

HS_NCpruned:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D HS_NCPRUNED $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

NE_sFFT:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D HS_NCPRUNED $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

HS_sFFT:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D HS_SFFT $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

HS_csFFT:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D HS_CSFFT $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

bag_sFFT:	$(ENTROPY_OBJS)
	$(CPP) $(CPPFLAGS) -D BAG_SFFT $(ENTROPY_OBJS) $(LIBS) -o $(BINDIR)/$@

mw_%:	$(MW_OBJS)
	$(CPP) $(CPPFLAGS) -D MW_$* $(MW_OBJS) $(LIBS) -o $(BINDIR)/$@

lib:	*.o
	$(CPP) $(CPPFLAGS) *.o $(LIBS) -shared -o libpvalue.a

motifeval:	$(ME_OBJS)
	$(CPP) $(CPPFLAGS) $(ME_OBJS) $(LIBS) -shared -o libmotifeval.a

all:	
	make $(EXES)

include depend
