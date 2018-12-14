#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <fstream>
#include "complex.h"
#include "input_image.h"
#include <cmath>
#include <string.h>
#include <chrono>

using namespace std;

/**
 * Do 2d dft in one thread. If forward is false, the inverse will be done
 * (forward is the default, though)
 */
Complex* doDftHoriz(Complex* data, int width, int height, bool forward = true) {
    int expOp = forward ? 1 : -1;
    float sumOp = forward ? 1.0 : float(1.0/width);
    Complex* dftData = new Complex[width*height];
    // Horiz
    for (int iH = 0; iH < height; iH++) {
        for (int iW = 0; iW < width; iW++) {
            Complex sum = Complex();
            for (int t = 0; t < width; t++) {
                double angle = 2.0 * 3.14159265358979f * float(t) * float(iW) / float(width);
                //sum = sum + data[iH * width + t] * exp(-angle);
                sum.real = sum.real + data[iH * width + t].real * cos(angle) + (data[iH * width + t].imag * sin(angle))*expOp;
                // sum.imag = sum.imag - data[iH * width + t].real * sin(angle);
                sum.imag = sum.imag - data[iH * width + t].real * sin(angle) * expOp + data[iH * width + t].imag * cos(angle);
            }
            //dftData[iH * width + iW] = sum;
            dftData[iH * width + iW] = sum * sumOp;
        }
    }

    return dftData;
}

Complex* doDftVert(Complex* data, int width, int height, bool forward = true) {
    int expOp = forward ? 1 : -1;
    float sumOp = forward ? 1.0 : float(1.0/width);
    Complex* dftData = new Complex[width*height];

    // Vert
    sumOp = forward ? 1.0 : float(1.0/height);
    for (int iW = 0; iW < width; iW++) {
        for (int iH = 0; iH < height; iH++) {
            Complex sum = Complex();
            for (int t = 0; t < height; t++) {
                double angle = 2.0 * 3.14159265358979f * float(t) * float(iH) / float(height);
                sum.real = sum.real + data[width * t + iW].real * cos(angle) + data[width * t + iW].imag * sin(angle) *expOp;
                //sum.imag = sum.imag - dftData[width * t + iW].real * sin(angle);
                sum.imag = sum.imag - data[width * t + iW].real * sin(angle) * expOp + data[width * t + iW].imag * cos(angle);
            }
            //dftData2[iH * width + iW] = sum;
            dftData[iH * width + iW] = sum * sumOp;
        }
    }

    return dftData;
}

void separate(Complex* data, int n) {
    Complex* temp = new Complex[n/2];
    for (int i = 0; i < n/2; i++) // Copy odd
        temp[i] = data[i*2 + 1];
    for (int i = 0; i < n/2; i++) // Copy even to lower half
        data[i] = data[i*2];
    for (int i = 0; i < n/2; i++) // Copy odd to upper half
        data[i + n/2] = temp[i];
    delete[] temp;
}

void fftHoriz(Complex* data, int n) {
    if (n < 2) {

    } else {
        separate(data, n);
        fftHoriz(data, n/2);
        fftHoriz(data+n/2, n/2);
        for (int i = 0; i < n/2; i++) {
            Complex e = data[i];
            Complex o = data[i + n/2];
            Complex wCC = Complex();
            double angle = -2.*M_PI*float(i)/float(n);
            wCC.real = cos(angle);
            // sum.imag = sum.imag - data[iH * width + t].real * sin(angle);
            wCC.imag = sin(angle);
            data[i] = e + wCC * o;
            data[i + n/2] = e - wCC * o;
        }
    }
}

void fft1(Complex* data, int width, int height, int startWidth, int startHeight) {
    for (int i = startWidth; i < startHeight; i++) {
        fftHoriz(&data[i*height], width);
    }
}

// INITIALIZE
int main(int argc,char**argv){
  //cout << "START" << endl;
  auto start = std::chrono::system_clock::now();
  
  // CHECK MPI INIT
  int rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
      printf ("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
  int rc1;
  int rc2;

  // INITIALIZE MPI
  int  numProcs, rank; //numProcs = total number of processors; rank = computer, 
  MPI_Comm_size(MPI_COMM_WORLD,&numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //printf ("NUMBER OF PROCESSORS = %d MY RANK =  %d\n", numProcs,rank);

  if (argc != 4) {
      cout << "wrong # inputs" << endl;
      return -1;
  }
  bool isForward(!strcmp(argv[1], "forward"));
  char* inputFile = argv[2];
  char* outputFile = argv[3];
  //cout << inputFile << endl;
  InputImage im(inputFile);    
  int width = im.get_width();
  int height = im.get_height();
  int GridSize = width*height;
    
  float RealGrid[GridSize] = {};
  float ImagGrid[GridSize] = {};
  float RealGrid_recv[GridSize] = {};
  float ImagGrid_recv[GridSize] = {};
  float RealGrid_T[GridSize] = {};
  float ImagGrid_T[GridSize] = {};

  int tasks = GridSize/numProcs;
  //cout << "GRIDSIZE: " << GridSize << endl;
  //cout << "TASKS PER PROCESSOR: " << tasks << endl;

  Complex* data = im.get_image_data();
  Complex* dataT = new Complex[width*height];
  //Complex* data_T = im.get_image_data();

//===============================================
//================== RANK 0 ===================
//============================================

  if (rank == 0){
//================== FIRST RUN ========================
    for(int i = 0; i < GridSize; i++){
      RealGrid[i] = data[i].real;
      ImagGrid[i] = data[i].imag;
    }
    
    if (isForward) {
        std::cout << "doing forward, " << width << std::endl;
    } else { // Note this doesn't do anything rn 
        std::cout << "doing reverse" << std::endl;
    }

    // SEND 
    for(int i = 1; i < numProcs; i++){
      rc1 = MPI_Send(&RealGrid[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      rc2 = MPI_Send(&ImagGrid[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " <<  rc2 << endl;
        MPI_Finalize();
        exit(1);
      }
    }
  
    // CALCULATIONS
    fft1(data, width, height, 0, height/numProcs);
  
    // RECIEVE
    for(int i = 1; i < numProcs; i++){
      MPI_Status status;
      rc1 = MPI_Recv(&RealGrid_recv[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
      rc2 = MPI_Recv(&ImagGrid_recv[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
      if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
        MPI_Finalize();
        exit(1);
      }

      // REPLACE
      for(int j = 0; j < tasks; j++){
        data[j + i*tasks].real = RealGrid_recv[j + i*tasks];
        data[j + i*tasks].imag = ImagGrid_recv[j + i*tasks];      
      }
    }
   
    // FIRST GRID 
    //im.save_image_data_real("FirstKevin.txt", data, width, height);

    
//================= SECOND RUN =======================
	// TRANSPOSE
    for(int row = 0; row < height; row++){
      for(int col = 0; col < width; col++){
        dataT[col*width + row].real = data[row*height + col].real;
        dataT[col*width + row].imag = data[row*height + col].imag;
      }
    }

    for(int i = 0; i < GridSize; i++){
      //RealGrid[i] = dataT[i].real;
      //ImagGrid[i] = dataT[i].imag;
      RealGrid_T[i] = dataT[i].real;
      ImagGrid_T[i] = dataT[i].imag;
    }

    // TRANSPOSE
    /*for(int row = 0; row < height; row++){
      for(int col = 0; col < width; col++){
        RealGrid_T[col*width + row] = RealGrid[row*height + col];
        ImagGrid_T[col*width + row] = ImagGrid[row*height + col];
      }
    }*/

    // SEND
    for(int i = 1; i < numProcs; i++){
      rc1 = MPI_Send(&RealGrid_T[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      rc2 = MPI_Send(&ImagGrid_T[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " <<  rc2 << endl;
        MPI_Finalize();
        exit(1);
      }
    }
    
    // CALCULATIONS
    fft1(dataT, width, height, 0, height/numProcs);

    // RECIEVE
    for(int i = 1; i < numProcs; i++){
      MPI_Status status;
      rc1 = MPI_Recv(&RealGrid_recv[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
      rc2 = MPI_Recv(&ImagGrid_recv[0], GridSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
      if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
        MPI_Finalize();
        exit(1);
      }
    
      // REPLACE
      for(int j = 0; j < tasks; j++){
        RealGrid_T[j + i*tasks] = RealGrid_recv[j + i*tasks];
        ImagGrid_T[j + i*tasks] = ImagGrid_recv[j + i*tasks];
      }
    }
    for (int i = 0; i < tasks; i++) {
        RealGrid_T[i] = dataT[i].real;
        ImagGrid_T[i] = dataT[i].imag;
    }

    // INVERSE TRANSPOSE
    for(int row = 0; row < height; row++){
      for(int col = 0; col < width; col++){
        dataT[col*width + row].real = RealGrid_T[row*height + col];
        dataT[col*width + row].imag = ImagGrid_T[row*height + col];
      }
    }
     
    // FINAL GRID 
    if (isForward) {
	    cout << "writing grid" << endl;
	    im.save_image_data(outputFile, dataT, width, height);
	    cout << "dunzo" << endl;
	} else {
		Complex* dataToInv = im.get_image_data();
		dataToInv = doDftHoriz(dataToInv, width, height, false);
		dataToInv = doDftVert(dataToInv, width, height, false);
		im.save_image_data(outputFile, dataToInv, width, height);
	}

  }

//=============================================================
//====================== IF RANK NOT ZERO=========================
//===============================================================
  else{
    MPI_Status status;

//================= FIRST RUN ====================== 
    // RECIEVE
    rc1 = MPI_Recv(&RealGrid_recv[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    rc2 = MPI_Recv(&ImagGrid_recv[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);     
    if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
      cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
      MPI_Finalize();
      exit(1); 
    }
 
    // TRANSLATE TO COMPLEX	
    for(int i = 0; i < tasks; i++){
      data[i + rank*tasks].real = RealGrid_recv[i + rank*tasks];
      data[i + rank*tasks].imag = ImagGrid_recv[i + rank*tasks];
      data[i + rank*tasks].real = data[i + rank*tasks].real;
      data[i + rank*tasks].imag = data[i + rank*tasks].imag;
    }
    
    // CALCULATIONS
    //fft1(data, width, height, rank*(GridSize/height), (rank+1)*(GridSize/height));
    fft1(data, width, height, rank * height/numProcs, (rank + 1) *height/numProcs);
     
    // TRANSLATE TO FLOAT
    for(int i = 0; i < tasks; i++){
      RealGrid[i + rank*tasks] = data[i + rank*tasks].real;
      ImagGrid[i + rank*tasks] = data[i + rank*tasks].imag;
    }
 
    // SEND BOTH GRIDS
    rc1 = MPI_Send(&RealGrid[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    rc2 = MPI_Send(&ImagGrid[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
      cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
      MPI_Finalize();
      exit(1);
    }
  
//================== SECOND RUN ===================== 
    // RECIEVE
    rc1 = MPI_Recv(&RealGrid_recv[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    rc2 = MPI_Recv(&ImagGrid_recv[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
      cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
      MPI_Finalize();
      exit(1);
    }
    
    // TRANSLATE TO COMPLEX 
    for(int i = 0; i < tasks; i++){
      data[i + rank*tasks].real = RealGrid_recv[i + rank*tasks];
      data[i + rank*tasks].imag = ImagGrid_recv[i + rank*tasks];
      data[i + rank*tasks].real = data[i + rank*tasks].real;
      data[i + rank*tasks].imag = data[i + rank*tasks].imag;
     // data[i + rank*tasks].real = rank;
     // data[i + rank*tasks].imag = rank;
    }

    // CALCULATE
    fft1(data, width, height, rank * height/numProcs, (rank + 1) *height/numProcs);
    
    // TRANSLATE TO FLOAT 
    for(int i = 0; i < tasks; i++){
      RealGrid[i + rank*tasks] = data[i + rank*tasks].real;
      ImagGrid[i + rank*tasks] = data[i + rank*tasks].imag;
    }

    // SEND
    rc1 = MPI_Send(&RealGrid[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    rc2 = MPI_Send(&ImagGrid[0], GridSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    if (rc1 != MPI_SUCCESS || rc2 != MPI_SUCCESS) {
      cout << "Rank " << rank << " send failed, rc1 " << rc1 << " rc2 " << rc2 << endl;
      MPI_Finalize();
      exit(1);
    }
  }

 
  //cout << "FINISHED" << endl;
  MPI_Finalize();
  
  if(rank == 0){
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << elapsed_seconds.count() << std::endl;
  }
}





    







		

