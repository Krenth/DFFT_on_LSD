#include <iostream>
#include <vector>
#include <cmath>
#include <string.h>
#include "complex.h"
#include "input_image.h"
#include <pthread.h>

using std::vector;

struct horizThread_data
{
    Complex *dftData;
    Complex *data;
    int iH;
    int width;
    int expOp;
    Complex sumOp;
};

struct vertThread_data
{
    Complex *dftData2;
    Complex *dftData;
    int iW;
    int width;
    int height;
    int expOp;
    Complex sumOp;
};


void *threadDftHoriz(void *threadarg)
{
    struct horizThread_data *td;
    td = (struct horizThread_data *)threadarg;
    Complex sum;
    for (int iW = 0; iW < td->width; iW++)
    {
        sum = Complex();
        for (int t = 0; t < td->width; t++)
        {
            double angle = td->expOp * 2.0 * M_PI * float(t) * float(iW) / float(td->width);
            Complex expTerm = Complex(cos(angle), sin(angle));
            sum = sum + td->data[td->iH * td->width + t] * expTerm;
        }
        td->dftData[td->iH * td->width + iW] = sum * td->sumOp;
    }
    pthread_exit(NULL);
}

void *threadDftVert(void *threadarg)
{
    struct vertThread_data *td;
    td = (struct vertThread_data *)threadarg;
    Complex sum;
    for (int iH = 0; iH < td->height; iH++)
    {
        sum = Complex();
        for (int t = 0; t < td->height; t++)
        {
            double angle = td->expOp * 2.0 * M_PI * float(t) * float(iH) / float(td->height);
            Complex expTerm = Complex(cos(angle), sin(angle));
            sum = sum + td->dftData[t * td->width + td->iW] * expTerm;
        }
        td->dftData2[iH * td->width + td->iW] = sum * td->sumOp;
    }
    pthread_exit(NULL);
}

/**
 * Do 2d dft in one thread. If forward is false, the inverse will be done
 * (forward is the default, though)
 */
Complex* doDft(Complex* data, int width, int height, bool forward = true) {
    int expOp = forward ? -1 : 1;
    Complex sumOp = forward ? Complex(1.0) : Complex(float(1.0 / width));
    Complex* dftData = new Complex[width * height];

    // Horiz
    pthread_t horizThread[height];
    struct horizThread_data tdh[height];
    int rc;

    for (int iH = 0; iH < height; iH++)
    {
        tdh[iH].dftData = dftData;
        tdh[iH].data = data;
        tdh[iH].iH = iH;
        tdh[iH].width = width;
        tdh[iH].expOp = expOp;
        tdh[iH].sumOp = sumOp;

        rc = pthread_create(&horizThread[iH], NULL, threadDftHoriz, (void *)&tdh[iH]);
        if (rc) {
            std::cout << "<<< ERROR: Cannot create thread! >>>" << std::endl;
            exit(-1);
        }
        // for (int iW = 0; iW < width; iW++)
        // {
        //     sum = Complex();
        //     for (int t = 0; t < width; t++)
        //     {
        //         double angle = expOp * 2.0 * M_PI * float(t) * float(iW) / float(width);
        //         Complex expTerm = Complex(cos(angle), sin(angle));
        //         sum = sum + data[iH * width + t] * expTerm;
        //     }
        //     dftData[iH * width + iW] = sum * sumOp;
        // }
    }
    void *status;

    for (int iH = 0; iH < height; iH++)
    {
        rc = pthread_join(horizThread[iH], &status);
        if (rc) {
            std::cout << "<<< ERROR: Cannot join thread! >>>" << std::endl;
            exit(-1);
        }
    }




    // Vert
    Complex *dftData2 = new Complex[width * height];
    sumOp = forward ? Complex(1.0) : Complex(float(1.0 / height));

    pthread_t vertThread[width];
    struct vertThread_data tdv[width];

    for (int iW = 0; iW < width; iW++)
    {
        tdv[iW].dftData2 = dftData2;
        tdv[iW].dftData = dftData;
        tdv[iW].iW = iW;
        tdv[iW].width = width;
        tdv[iW].height = height;
        tdv[iW].expOp = expOp;
        tdv[iW].sumOp = sumOp;

        rc = pthread_create(&vertThread[iW], NULL, threadDftVert, (void *)&tdv[iW]);
        if (rc) {
            std::cout << "<<< ERROR: Cannot create thread! >>>" << std::endl;
            exit(-1);
        }
        // for (int iH = 0; iH < height; iH++)
        // {
        //     sum = Complex();
        //     for (int t = 0; t < height; t++)
        //     {
        //         double angle = expOp * 2.0 * M_PI * float(t) * float(iH) / float(height);
        //         Complex expTerm = Complex(cos(angle), sin(angle));
        //         sum = sum + dftData[t * width + iW] * expTerm;
        //     }
        //     dftData2[iH * width + iW] = sum * sumOp;
        // }
    }

    for (int iW = 0; iW < width; iW++)
    {
        rc = pthread_join(horizThread[iW], &status);
        if (rc) {
            std::cout << "<<< ERROR: Cannot join thread! >>>" << std::endl;
            exit(-1);
        }
    }

    delete[] dftData;
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
    { // Note this doesn't do anything rn
        std::cout << "doing reverse" << std::endl;
    }
    std::cout << inputFile << std::endl;
    InputImage im(inputFile);
    int width = im.get_width();
    int height = im.get_height();

    Complex *data = im.get_image_data();
    Complex *dftData = doDft(data, width, height, isForward);
    //Complex* invDftData = doDft(dftData, width, height, false);

    std::cout << "writing" << std::endl;
    im.save_image_data(outputFile, dftData, width, height);
    // im.save_image_data(outputFile, invDftData, width, height);
    std::cout << "dunzo" << std::endl;
    return 0;
}
