__kernel void quasirandom_generator(
    __global const unsigned int *table,
    __global float *out,
    int num_vectors)
{
    int i = get_global_id(0);
    if (i >= num_vectors) return;
    
    // Generate 3D quasirandom vectors
    for (int d = 0; d < 3; d++) {
        unsigned int result = 0;
        unsigned int n = i;
        
        for (int bit = 0; bit < 32; bit++) {
            if (n & 1) {
                result ^= table[d * 32 + bit];
            }
            n >>= 1;
        }
        
        out[d * num_vectors + i] = (float)result / 4294967296.0f;
    }
}