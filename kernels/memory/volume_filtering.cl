const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                          CLK_ADDRESS_CLAMP_TO_EDGE | 
                          CLK_FILTER_NEAREST;

__kernel void volume_filter(
    __read_only image3d_t src,
    __write_only image3d_t dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int4 coord = (int4)(x, y, z, 0);
    
    float sum = 0.0f;
    
    // 3x3x3 blur filter
    for(int k = -1; k <= 1; k++) {
        for(int j = -1; j <= 1; j++) {
            for(int i = -1; i <= 1; i++) {
                sum += read_imagef(src, sampler, coord + (int4)(i, j, k, 0)).x;
            }
        }
    }
    
    write_imagef(dst, coord, (float4)(sum / 27.0f, 0, 0, 0));
}