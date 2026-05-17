__kernel void haar_1d_step(
    __global float *in,
    __global float *out,
    int step_length)
{
    int i = get_global_id(0); // 0 to step_length/2
    if (i >= step_length / 2) return;
    
    float a = in[2 * i];
    float b = in[2 * i + 1];
    
    // Average
    out[i] = (a + b) * 0.5f;
    // Difference (Detail)
    out[step_length / 2 + i] = (a - b) * 0.5f;
}