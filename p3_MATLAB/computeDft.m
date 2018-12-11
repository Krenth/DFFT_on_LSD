function [outreal,outimag] = computeDft(inVec)
    n = length(inVec);
    inreal = real(inVec);
    inimag = imag(inVec);
    outreal = zeros(1,n);
    outimag = zeros(1,n);
    for k = 1:n
        sumreal = 0;
        sumimag = 0;
        for t= 1:n
            angle = 2 * pi * (t-1) * (k-1) / n;
            sumreal = sumreal + inreal(t) * cos(angle) + inimag(t) * sin(angle);
            sumimag = sumimag - inreal(t) * sin(angle) + inimag(t) * cos(angle);
        end
        outreal(k) = sumreal;
        outimag(k) = sumimag;
    end
end