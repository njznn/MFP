//=============================================================================
// Program za command line izracun korelacije !!! realnih podatkov !!!
//  ./corr <N> < input > output
//  N - predvidena kolicina podatkov 
//  input = datoteka z eno kolono podatkov
//  outout = datoteka z eno kolono podatkov
//
// Prevajanje:
// g++ corr_v3.cpp -o corr_v3 -O3 -Wall -lm -lfftw3
//
// Avtor: Martin Horvat, 2005 (netestirano)
// ============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>

//using version 3.x fftw

#include <fftw3.h>

using namespace std;

int main(int argc, char *argv[]){
  
  if (argc != 2){
    cerr << "Usage: ./corr_v3 <N0:N1> < input > output 2> <mean>\n\n"
         << "Note: [N0, N1] - interval upostevanih tock \n";
 
    exit(1);
  }

  long N0, N1, N, M,
       i, j;
  
  sscanf(argv[1],"%ld:%ld", &N0, &N1); 
  N = N1 - N0 + 1;
    
  double  *d = new double [N],
          *e = new double [N],
	  mean;  
  
  cerr << "Reading ..." << endl;
       
  for (i = j = 0; (j < N) && cin; i++) 
    if (N0 < i) {
      cin >> d[j];
      j++;
    }
     
  for (j = i; j < N; j++) d[i] = 0;     
  
  fftw_plan pf, pb;
    
  pf = fftw_plan_r2r_1d(N, d, e, FFTW_R2HC, /*FFTW_PATIENT*/FFTW_ESTIMATE);
  pb = fftw_plan_r2r_1d(N, e, d, FFTW_HC2R, /*FFTW_PATIENT*/FFTW_ESTIMATE);

  cerr << "Processing ..." << endl;
  
  fftw_execute(pf);         
  
  mean = e[0]/N; e[0] = 0;
  
  M = (N + 1)/2;        
  for (i = 1; i < M; i++){  	/* (k < n/2 rounded up) */
    e[i] = e[i]*e[i] + e[N-i]*e[N-i];
    e[N-i] = 0;
  }
    
  if (!(N % 2)) e[N/2] *= e[N/2];  /* n is even */
  //if (N % 2 == 0) e[N/2] = 0;      /* n is even */
		  
  fftw_execute(pb);         
  
  cerr << "Outputing ..." << endl;
  
  cerr << setprecision(12); cout << setprecision(12);
    
  double f = 1.0/(double(N)*N);
  for (i = 0; i < N; i++) cout << f*d[i]  << '\n';
  //for (i = 0; i < N/2; i++) cout << f*d[i]  << '\n';
  
  cerr << "Mean:" << mean << " Nr. points:" << N << '\n';  
  
  delete [] d;     
  delete [] e;
      
  fftw_destroy_plan(pf);  
  fftw_destroy_plan(pb);
  
  return 0;
}

