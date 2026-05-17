__kernel void blackScholes(
    __global const float *S,
    __global const float *X,
    __global const float *T,
    __global float *Call,
    __global float *Put,
    float R,
    float V,
    int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    
    float sqrtT = sqrt(T[i]);
    float d1 = (log(S[i] / X[i]) + (R + 0.5f * V * V) * T[i]) / (V * sqrtT);
    float d2 = d1 - V * sqrtT;
    
    const float inv_sqrt_2pi = 0.39894228040f;
    
    float k1 = 1.0f / (1.0f + 0.2316419f * fabs(d1));
    float cnd1 = inv_sqrt_2pi * exp(-0.5f * d1 * d1) * 
                 (k1 * (0.319381530f + k1 * (-0.356563782f + k1 * (1.781477937f + k1 * (-1.821255978f + k1 * 1.330274429f)))));
    if (d1 > 0.0f) cnd1 = 1.0f - cnd1;
    
    float k2 = 1.0f / (1.0f + 0.2316419f * fabs(d2));
    float cnd2 = inv_sqrt_2pi * exp(-0.5f * d2 * d2) * 
                 (k2 * (0.319381530f + k2 * (-0.356563782f + k2 * (1.781477937f + k2 * (-1.821255978f + k2 * 1.330274429f)))));
    if (d2 > 0.0f) cnd2 = 1.0f - cnd2;
    
    float expRT = exp(-R * T[i]);
    
    Call[i] = S[i] * cnd1 - X[i] * expRT * cnd2;
    Put[i]  = X[i] * expRT * (1.0f - cnd2) - S[i] * (1.0f - cnd1);
}
