clear
clc
image = im2double(rgb2gray(imread('download.jpg')));
n = length(image);
complexArr = zeros(n,n);

for row = 1:n
    [vecReal,vecImag]= computeDft(image(row,:));
    complexArr(row,:) = vecReal+ 1i*vecImag;
end


midArr = complexArr;
for col = 1:n
    [vecReal,vecImag]= computeDft(midArr(:,col));
    complexArr(:,col) = vecReal+ 1i*vecImag;
end

imageFFT = fft2(image);

figure(1);
imagesc(real(ifft2(complexArr)))
figure(2);
imagesc(real(ifft2(imageFFT)))