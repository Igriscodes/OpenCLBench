__kernel void fdtd3d_step(__global const float *in, __global float *out, int dimx, int dimy, int dimz) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    if (x >= dimx || y >= dimy || z >= dimz) return;
    
    int idx = z * dimx * dimy + y * dimx + x;
    
    // Bounds checking for stencil (simple clamp to edge behavior by avoiding reads)
    int xm1 = (x > 0) ? x - 1 : x;
    int xp1 = (x < dimx - 1) ? x + 1 : x;
    int ym1 = (y > 0) ? y - 1 : y;
    int yp1 = (y < dimy - 1) ? y + 1 : y;
    int zm1 = (z > 0) ? z - 1 : z;
    int zp1 = (z < dimz - 1) ? z + 1 : z;
    
    float center = in[idx];
    float left = in[z * dimx * dimy + y * dimx + xm1];
    float right = in[z * dimx * dimy + y * dimx + xp1];
    float top = in[z * dimx * dimy + ym1 * dimx + x];
    float bottom = in[z * dimx * dimy + yp1 * dimx + x];
    float front = in[zm1 * dimx * dimy + y * dimx + x];
    float back = in[zp1 * dimx * dimy + y * dimx + x];
    
    // Standard 6-point stencil operation
    float result = center * 0.4f + (left + right + top + bottom + front + back) * 0.1f;
    
    out[idx] = result;
}