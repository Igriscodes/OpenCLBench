__kernel void fwt_step(__global float *data, int step) {
    int i = get_global_id(0);
    
    int group = i / step;
    
    int idx0 = group * (step * 2) + (i % step);
    int idx1 = idx0 + step;
    
    float a = data[idx0];
    float b = data[idx1];
    
    data[idx0] = a + b;
    data[idx1] = a - b;
}
