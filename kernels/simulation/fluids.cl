__kernel void sph_density(
    __global const float4 *pos,
    __global float *density,
    float h,
    int N,
    __local float4 *local_pos)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    
    if (global_id >= N) return;
    
    float4 p = pos[global_id];
    float d = 0.0f;
    float h2 = h * h;
    float poly6 = 315.0f / (64.0f * 3.141592f * pow(h, 9.0f));
    
    for (int i = 0; i < N; i += local_size) {
        local_pos[local_id] = pos[i + local_id];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int j = 0; j < local_size; ++j) {
            float4 pj = local_pos[j];
            float4 r = p - pj;
            float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
            
            if (r2 < h2) {
                float diff = h2 - r2;
                d += poly6 * diff * diff * diff;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    density[global_id] = d;
}