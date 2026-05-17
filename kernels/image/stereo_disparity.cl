__kernel void stereo_disparity(
    __global const unsigned char *left,
    __global const unsigned char *right,
    __global unsigned char *out,
    int width,
    int height,
    int max_disparity)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3) {
        if(x < width && y < height) out[y * width + x] = 0;
        return;
    }
    
    int best_d = 0;
    int min_sad = 9999999;
    
    for (int d = 0; d < max_disparity; d++) {
        if (x - d < 3) break;
        
        int sad = 0;
        // 7x7 block
        for (int j = -3; j <= 3; j++) {
            for (int i = -3; i <= 3; i++) {
                int val_l = left[(y + j) * width + (x + i)];
                int val_r = right[(y + j) * width + (x + i - d)];
                sad += abs(val_l - val_r);
            }
        }
        
        if (sad < min_sad) {
            min_sad = sad;
            best_d = d;
        }
    }
    
    out[y * width + x] = (unsigned char)(best_d * 255 / max_disparity);
}