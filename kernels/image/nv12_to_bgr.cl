__kernel void nv12_to_bgr(
    __global const unsigned char *nv12,
    __global unsigned char *bgr,
    int width,
    int height)
{
    int x2 = get_global_id(0) * 2;
    int y2 = get_global_id(1) * 2;
    
    if (x2 >= width || y2 >= height) return;
    
    int y_idx1 = y2 * width + x2;
    int y_idx2 = y_idx1 + 1;
    int y_idx3 = (y2 + 1) * width + x2;
    int y_idx4 = y_idx3 + 1;
    
    int uv_idx = width * height + (y2 / 2) * width + x2;
    
    float u = (float)nv12[uv_idx] - 128.0f;
    float v = (float)nv12[uv_idx + 1] - 128.0f;
    
    float y1 = (float)nv12[y_idx1];
    float y2_val = (float)nv12[y_idx2];
    float y3 = (float)nv12[y_idx3];
    float y4 = (float)nv12[y_idx4];
    
    float ru = u * 1.772f;
    float gu = u * 0.344f;
    float gv = v * 0.714f;
    float rv = v * 1.402f;
    
    // Simplistic YUV to BGR
    bgr[y_idx1 * 3 + 0] = clamp((int)(y1 + ru), 0, 255);
    bgr[y_idx1 * 3 + 1] = clamp((int)(y1 - gu - gv), 0, 255);
    bgr[y_idx1 * 3 + 2] = clamp((int)(y1 + rv), 0, 255);
    
    bgr[y_idx2 * 3 + 0] = clamp((int)(y2_val + ru), 0, 255);
    bgr[y_idx2 * 3 + 1] = clamp((int)(y2_val - gu - gv), 0, 255);
    bgr[y_idx2 * 3 + 2] = clamp((int)(y2_val + rv), 0, 255);
    
    bgr[y_idx3 * 3 + 0] = clamp((int)(y3 + ru), 0, 255);
    bgr[y_idx3 * 3 + 1] = clamp((int)(y3 - gu - gv), 0, 255);
    bgr[y_idx3 * 3 + 2] = clamp((int)(y3 + rv), 0, 255);
    
    bgr[y_idx4 * 3 + 0] = clamp((int)(y4 + ru), 0, 255);
    bgr[y_idx4 * 3 + 1] = clamp((int)(y4 - gu - gv), 0, 255);
    bgr[y_idx4 * 3 + 2] = clamp((int)(y4 + rv), 0, 255);
}