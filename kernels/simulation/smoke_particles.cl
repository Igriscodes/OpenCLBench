__kernel void advect_smoke(
    __global float4 *pos,
    __global const float4 *vel,
    float dt,
    int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    
    float4 p = pos[i];
    float4 v = vel[i]; // In a real simulation, velocity is interpolated from a 3D grid
    
    // Add some noise/buoyancy
    v.y += 0.1f * dt; // upward buoyancy
    v.x += 0.01f * sin(p.y * 10.0f); // turbulence
    
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    
    // Lifespan/fade
    p.w -= dt * 0.5f; 
    if (p.w <= 0.0f) {
        p.w = 1.0f; // respawn
        p.y = 0.0f;
    }
    
    pos[i] = p;
}