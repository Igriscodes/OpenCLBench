__kernel void nbody_integration(
    __global const float4 *pos,
    __global const float4 *vel,
    __global float4 *pos_next,
    __global float4 *vel_next,
    float dt,
    float eps_sq,
    int N,
    __local float4 *local_pos)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    
    float4 p = pos[global_id];
    float4 v = vel[global_id];
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < N; i += local_size) {
        local_pos[local_id] = pos[i + local_id];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int j = 0; j < local_size; ++j) {
            float4 pj = local_pos[j];
            float4 r = pj - p;
            float dist_sq = r.x*r.x + r.y*r.y + r.z*r.z + eps_sq;
            float inv_dist = rsqrt(dist_sq);
            float inv_dist_cube = inv_dist * inv_dist * inv_dist;
            float s = pj.w * inv_dist_cube;
            
            acc.x += r.x * s;
            acc.y += r.y * s;
            acc.z += r.z * s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    v.x += acc.x * dt;
    v.y += acc.y * dt;
    v.z += acc.z * dt;
    
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    
    pos_next[global_id] = p;
    vel_next[global_id] = v;
}
