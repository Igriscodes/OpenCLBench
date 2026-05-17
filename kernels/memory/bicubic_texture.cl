const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                          CLK_ADDRESS_CLAMP_TO_EDGE | 
                          CLK_FILTER_NEAREST;

float w0(float a) { return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f); }
float w1(float a) { return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f); }
float w2(float a) { return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f); }
float w3(float a) { return (1.0f/6.0f)*(a*a*a); }

__kernel void bicubic_upscale(
    __read_only image2d_t src,
    __write_only image2d_t dst,
    float scale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    float fx = (float)x / scale;
    float fy = (float)y / scale;
    
    int ix = (int)floor(fx - 0.5f);
    int iy = (int)floor(fy - 0.5f);
    
    float fpx = fx - 0.5f - (float)ix;
    float fpy = fy - 0.5f - (float)iy;
    
    float wx[4] = { w0(fpx), w1(fpx), w2(fpx), w3(fpx) };
    float wy[4] = { w0(fpy), w1(fpy), w2(fpy), w3(fpy) };
    
    float sum = 0.0f;
    for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
            float val = read_imagef(src, sampler, (int2)(ix + i - 1, iy + j - 1)).x;
            sum += val * wx[i] * wy[j];
        }
    }
    
    write_imagef(dst, (int2)(x, y), (float4)(sum, 0, 0, 0));
}