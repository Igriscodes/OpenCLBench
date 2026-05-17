const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                          CLK_ADDRESS_CLAMP_TO_EDGE | 
                          CLK_FILTER_NEAREST;

__kernel void sobel_filter(
    __read_only image2d_t src,
    __write_only image2d_t dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int2 coord = (int2)(x, y);
    
    float4 p00 = read_imagef(src, sampler, coord + (int2)(-1, -1));
    float4 p10 = read_imagef(src, sampler, coord + (int2)( 0, -1));
    float4 p20 = read_imagef(src, sampler, coord + (int2)( 1, -1));
    float4 p01 = read_imagef(src, sampler, coord + (int2)(-1,  0));
    float4 p21 = read_imagef(src, sampler, coord + (int2)( 1,  0));
    float4 p02 = read_imagef(src, sampler, coord + (int2)(-1,  1));
    float4 p12 = read_imagef(src, sampler, coord + (int2)( 0,  1));
    float4 p22 = read_imagef(src, sampler, coord + (int2)( 1,  1));
    
    float4 gx = -p00 + p20 + 
                -2.0f * p01 + 2.0f * p21 + 
                -p02 + p22;
                
    float4 gy = -p00 - 2.0f * p10 - p20 + 
                 p02 + 2.0f * p12 + p22;
                 
    float4 mag = hypot(gx, gy);
    mag.w = 1.0f;
    
    write_imagef(dst, coord, mag);
}
