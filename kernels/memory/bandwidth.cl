__kernel void copy_float4(
    __global const float4 *src,
    __global float4 *dst,
    int N)
{
    int i = get_global_id(0);
    if (i < N) {
        dst[i] = src[i];
    }
}
