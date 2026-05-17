__kernel void sobol_qrng(
    __global const unsigned int *dir,
    __global float *out,
    int num_vectors,
    int num_dimensions)
{
    int i = get_global_id(0);
    if (i >= num_vectors) return;
    
    // Gray code implementation of Sobol
    for (int d = 0; d < num_dimensions; d++) {
        unsigned int result = 0;
        unsigned int n = i;
        
        for (int bit = 0; bit < 32; bit++) {
            if (n & 1) {
                result ^= dir[d * 32 + bit];
            }
            n >>= 1;
        }
        
        // Normalize to [0, 1)
        out[d * num_vectors + i] = (float)result / 4294967296.0f; 
    }
}