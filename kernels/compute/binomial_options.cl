__kernel void binomial_options(
    __global const float *S_arr,
    __global const float *X_arr,
    __global const float *T_arr,
    __global float *Call_arr,
    float R,
    float V,
    int num_steps)
{
    int i = get_global_id(0);
    
    float S = S_arr[i];
    float X = X_arr[i];
    float T = T_arr[i];
    
    float dt = T / (float)num_steps;
    float vSqrtDt = V * sqrt(dt);
    
    float u = exp(vSqrtDt);
    float d = exp(-vSqrtDt);
    float p = (exp(R * dt) - d) / (u - d);
    float inv_r = exp(-R * dt);
    
    // Simplistic array in private memory, limited steps
    float call_values[256];
    
    for (int j = 0; j <= num_steps; j++) {
        float price = S * pow(u, (float)j) * pow(d, (float)(num_steps - j));
        call_values[j] = fmax(price - X, 0.0f);
    }
    
    for (int step = num_steps - 1; step >= 0; step--) {
        for (int j = 0; j <= step; j++) {
            call_values[j] = inv_r * (p * call_values[j+1] + (1.0f - p) * call_values[j]);
        }
    }
    
    Call_arr[i] = call_values[0];
}