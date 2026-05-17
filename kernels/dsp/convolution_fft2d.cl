__kernel void fft2d_radix2(
    __global float *real,
    __global float *imag,
    int width,
    int step)
{
    int x = get_global_id(0); // 0 to width/2
    int y = get_global_id(1);
    
    // Butterfly indexing
    int group = x / step;
    int idx0 = y * width + group * (step * 2) + (x % step);
    int idx1 = idx0 + step;
    
    float angle = -3.14159265359f * (float)(x % step) / (float)step;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    
    float r0 = real[idx0];
    float i0 = imag[idx0];
    float r1 = real[idx1];
    float i1 = imag[idx1];
    
    float tr = r1 * cos_a - i1 * sin_a;
    float ti = r1 * sin_a + i1 * cos_a;
    
    real[idx0] = r0 + tr;
    imag[idx0] = i0 + ti;
    
    real[idx1] = r0 - tr;
    imag[idx1] = i0 - ti;
}