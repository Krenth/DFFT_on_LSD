#include <iostream>
#include <vector>
#include <cmath>
#include <string.h>
#include "complex.h"
#include "input_image.h"
#include <thread>

using std::vector;

struct blockData
{
    int width;
    int height;
    int expOp;
    Complex sumOp;
};

struct ComplexCUDA
{
    float real;
    float imag;
};

__global__ void blockDftHoriz(struct ComplexCUDA *dftData, struct ComplexCUDA *indata, struct blockData *hbd)
{
    __shared__ struct blockData bd;
    bd = *hbd;
    __shared__ Complex data[bd.width];
    __shared__ double angle[bd.width];
    __shared__ Complex expTerm[bd.width];
    __shared__ Complex sum[bd.width];

    data[threadIdx.x] = indata[blockIdx.x*bd.width+threadIdx.x];
    data[threadIdx.x+(bd.width+1)/2] = indata[blockIdx.x*bd.width+threadIdx.x+(bd.width+1)/2];
        
    for (int t = 0; t < bd.width; t++)
    {
        angle[threadIdx.x] = bd.expOp * 2.0 * M_PI * float(t) * float(threadIdx.x) / float(bd.width);
        expTerm[threadIdx.x] = Complex(cos(angle[threadIdx.x]), sin(angle[threadIdx.x]));
        sum[threadIdx.x] = sum[threadIdx.x] + data[t] * expTerm[threadIdx.x];

        if (threadIdx.x+(bd.width+1)/2 < bd.width) 
        {
            angle[threadIdx.x+(bd.width+1)/2] = td->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd.width+1)/2) / float(bd.width);
            expTerm[threadIdx.x+(bd.width+1)/2] = Complex(cos(angle[threadIdx.x+(bd.width+1)/2]), sin(angle[threadIdx.x+(bd.width+1)/2]));
            sum[threadIdx.x+(bd.width+1)/2] = sum[threadIdx.x+(bd.width+1)/2] + data[t+(bd.width+1)/2] * bd.expTerm[threadIdx.x+(bd.width+1)/2];
        }
    }
    dftData[blockIdx.x * bd.width + threadIdx.x] = sum[threadIdx.x] * bd.sumOp;
    if (threadIdx.x+(bd.width+1)/2 < bd.width) 
    {
        dftData[blockIdx.x * bd.width + threadIdx.x+(bd.width+1)/2] = sum[threadIdx.x+(bd.width+1)/2] * bd.sumOp;
    }
}


__global__ void blockDftVert(Complex *dftData, Complex *indata, struct blockData *hbd)
{
    __shared__ struct blockData bd;
    bd = *hbd;
    __shared__ Complex data[bd.height];
    __shared__ double angle[bd.height];
    __shared__ Complex expTerm[bd.height];
    __shared__ Complex sum[bd.height];

    data[threadIdx.x] = indata[threadIdx.x*bd.width+blockIdx.x];
    data[threadIdx.x+(bd.height+1)/2] = indata[(threadIdx.x+(bd.height+1)/2)*bd.width+blockIdx.x];

    for (int t = 0; t < bd.height; t++)
    {
        angle[threadIdx.x] = bd.expOp * 2.0 * M_PI * float(t) * float(threadIdx.x) / float(bd.height);
        expTerm[threadIdx.x] = Complex(cos(angle[threadIdx.x]), sin(angle[threadIdx.x]));
        sum[threadIdx.x] = sum[threadIdx.x] + data[t] * expTerm[threadIdx.x];

        if (threadIdx.x+(bd.height+1)/2 < bd.height) 
        {
            angle[threadIdx.x+(bd.height+1)/2] = td->expOp * 2.0 * M_PI * float(t) * float(threadIdx.x+(bd.height+1)/2) / float(bd.height);
            expTerm[threadIdx.x+(bd.height+1)/2] = Complex(cos(angle[threadIdx.x+(bd.height+1)/2]), sin(angle[threadIdx.x+(bd.height+1)/2]));
            sum[threadIdx.x+(bd.height+1)/2] = sum[threadIdx.x+(bd.height+1)/2] + data[t+(bd.height+1)/2] * bd.expTerm[threadIdx.x+(bd.height+1)/2];
        }
    }
    dftData2[threadIdx.x * bd.width + blockIdx.x] = sum[threadIdx.x] * bd.sumOp;
    if (threadIdx.x+(bd.height+1)/2 < bd.height) 
    {
        dftData2[(threadIdx.x+(bd.height+1)/2) * bd.width + blockIdx.x] = sum[threadIdx.x+(bd.height+1)/2] * bd.sumOp;
    }
}

/**
 * Do 2d dft in one thread. If forward is false, the inverse will be done
 * (forward is the default, though)
 */
Complex *doDft(Complex *data, int width, int height, bool forward = true)
{
    int expOp = forward ? -1 : 1;
    Complex sumOp = forward ? Complex(1.0) : Complex(float(1.0 / width));
    Complex *dftData2 = new Complex[width * height];

    struct blockData bd;

    Complex *d_data;
    Complex *d_dftData;
    Complex *d_dftData2;
    struct blockData *d_bd;

    bd.width = width;
    bd.height = height;
    bd.expOp = expOp;
    bd.sumOp = sumOp;

    cudaMalloc((void **) &d_data, sizeof(Complex[width*height]));
    cudaMalloc((void **) &d_dftData, sizeof(Complex[width*height]));
    cudaMalloc((void **) &d_dftData2, sizeof(Complex[width*height]));
    cudaMalloc((void **) &d_bd, sizeof(bd));

    cudaMemcpy(d_data, data, sizeof(Complex[width*height]), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bd, bd, sizeof(bd), cudaMemcpyHostToDevice);
    
    blockDtfHoriz<<<height,(width+1)/2>>>(d_dftData,d_data,d_bd);

    blockDftVert<<<width,(height+1)/2>>>(d_dftData2,d_dftData,d_bd);

    cudaMemcpy(dftData2, d_dftData2, sizeof(Complex[width*height]), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_dftData);
    cudaFree(d_dftData2);
    cudaFree(d_bd);

    return dftData2;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "wrong # inputs" << std::endl;
        return -1;
    }
    bool isForward(!strcmp(argv[1], "forward"));
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    if (isForward)
    {
        std::cout << "doing forward" << std::endl;
    }
    else
    {
        std::cout << "doing reverse" << std::endl;
    }
    std::cout << inputFile << std::endl;
    InputImage im(inputFile);
    int width = im.get_width();
    int height = im.get_height();

    Complex *data = im.get_image_data();
    Complex *dftData = doDft(data, width, height, isForward);

    std::cout << "writing" << std::endl;
    im.save_image_data(outputFile, dftData, width, height);
    std::cout << "dunzo" << std::endl;
    return 0;
}
