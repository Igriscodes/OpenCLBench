#define BLOCK_DIM 16

__kernel void recursive_gaussian_transpose(
    __global const float *in,
    __global float *out,
    int width,
    int height)
{
    __local float block[BLOCK_DIM][BLOCK_DIM+1]; // Pad to avoid bank conflicts
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    
    if (x < width && y < height) {
        block[ly][lx] = in[y * width + x];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int out_x = get_group_id(1) * BLOCK_DIM + lx;
    int out_y = get_group_id(0) * BLOCK_DIM + ly;
    
    if (out_x < height && out_y < width) {
        out[out_y * height + out_x] = block[lx][ly];
    }
}