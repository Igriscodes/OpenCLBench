const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                          CLK_ADDRESS_CLAMP_TO_EDGE | 
                          CLK_FILTER_NEAREST;

__kernel void bilateral_filter(
    __read_only image2d_t src,
    __write_only image2d_t dst,
    float sigma_d,
    float sigma_r)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 coord = (int2)(x, y);
    
    float center_val = read_imagef(src, sampler, coord).x;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    float two_sigma_d2 = 2.0f * sigma_d * sigma_d;
    float two_sigma_r2 = 2.0f * sigma_r * sigma_r;
    
    // 5x5 window
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            float val = read_imagef(src, sampler, coord + (int2)(i, j)).x;
            
            float dist_sq = (float)(i*i + j*j);
            float color_dist = val - center_val;
            float color_sq = color_dist * color_dist;
            
            float w = exp(-dist_sq / two_sigma_d2 - color_sq / two_sigma_r2);
            
            sum += val * w;
            weight_sum += w;
        }
    }
    
    write_imagef(dst, coord, (float4)(sum / weight_sum, 0, 0, 0));
}