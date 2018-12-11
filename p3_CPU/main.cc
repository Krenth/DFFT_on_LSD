#include <iostream>
#include <vector>
#include <cmath>
#include <string.h>
#include "complex.h"
#include "input_image.h"

using std::vector;

/**
 * Do 2d dft in one thread. If forward is false, the inverse will be done
 * (forward is the default, though)
 */
Complex* doDft(Complex* data, int width, int height, bool forward = true) {
    int expOp = forward ? 1 : -1;
    float sumOp = forward ? 1.0 : float(1.0/width);
    Complex* dftData = new Complex[width*height];
    // Horiz
    for (int iH = 0; iH < height; iH++) {
        for (int iW = 0; iW < width; iW++) {
            Complex sum = Complex();
            for (int t = 0; t < width; t++) {
                double angle = 2.0 * M_PI * float(t) * float(iW) / float(width);
                //sum = sum + data[iH * width + t] * exp(-angle); 
                sum.real = sum.real + data[iH * width + t].real * cos(angle);
                sum.imag = sum.imag + expOp * data[iH * width + t].real * sin(angle);
            }        
            dftData[iH * width + iW] = sum * sumOp;
        }
    } 

    // Vert
    Complex* dftData2 = new Complex[width*height];
    sumOp = forward ? 1.0 : float(1.0/height);
    for (int iW = 0; iW < width; iW++) {
        for (int iH = 0; iH < height; iH++) {
            Complex sum = Complex();
            for (int t = 0; t < height; t++) {
                double angle = 2.0 * M_PI * float(t) * float(iW) / float(height);
                sum.real = sum.real + dftData[iH * t + iW].real * cos(angle);
                sum.imag = sum.imag + expOp * dftData[iH * t + iW].real * sin(angle);

            }        
            dftData2[iH * width + iW] = sum * sumOp;
        }
    }

    delete[] dftData;
    return dftData2;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "wrong # inputs" << std::endl;
        return -1;
    }
    bool isForward(!strcmp(argv[1], "forward"));
    char* inputFile = argv[2];
    char* outputFile = argv[3];

    if (isForward) {
        std::cout << "doing forward" << std::endl;
    } else { // Note this doesn't do anything rn 
        std::cout << "doing reverse" << std::endl;
    }
    std::cout << inputFile << std::endl;
    InputImage im(inputFile); 
    int width = im.get_width();
    int height = im.get_height();
   
    Complex* data = im.get_image_data();
    Complex* dftData = doDft(data, width, height);
    //Complex* invDftData = doDft(dftData, width, height, false);
  
    std::cout << "writing" << std::endl;
    im.save_image_data(outputFile, dftData, width, height);
    // im.save_image_data(outputFile, invDftData, width, height);
    std::cout << "dunzo" << std::endl;
    return 0;
}
