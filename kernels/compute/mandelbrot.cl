__kernel void mandelbrot(__global int *out, int width, int height, int max_iters) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    
    float x0 = ((float)x / width) * 3.5f - 2.5f;
    float y0 = ((float)y / height) * 2.0f - 1.0f;
    
    float rx = 0.0f;
    float ry = 0.0f;
    
    int iter = 0;
    while (rx*rx + ry*ry <= 4.0f && iter < max_iters) {
        float x_new = rx*rx - ry*ry + x0;
        ry = 2.0f * rx * ry + y0;
        rx = x_new;
        iter++;
    }
    
    out[y * width + x] = iter;
}