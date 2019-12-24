texture<unsigned int, 2, cudaReadModeElementType> tex;
    
__global__ void interpolate(unsigned int *res, const int originalWidth, const int originalHeight, const int newWidth, const int newHeight, const int channelsCount) {
    const int y = blockDim.x * blockIdx.x + threadIdx.x, x = blockDim.y * blockIdx.y + threadIdx.y;
    const float scaleX = newHeight / originalHeight, scaleY = newWidth / originalWidth;
    const float positionX = x / scaleX, positionY = y / scaleY;
    const int modXi = (int) positionX, modYi = (int) positionY;
    const float modXf = positionX - modXi, modYf = positionY - modYi;
    const int modXiPlusOneLim = modXi + 1 >= originalHeight ? 0 : (modXi + 1), 
        modYiPlusOneLim = modYi + 1 >= originalWidth ? 0 : (modYi + 1);
    unsigned int **channels = new unsigned int*[channelsCount];
    for (int i = 0; i < channelsCount; i++) {
        channels[i] = new unsigned int[4];
    }
    unsigned int *pxls = new unsigned int[4]{
        tex2D(tex, modXi, modYi),
        tex2D(tex, modXiPlusOneLim, modYi), 
        tex2D(tex, modXi, modYiPlusOneLim), 
        tex2D(tex, modXiPlusOneLim, modYiPlusOneLim)
    };
        
    for (int i = 0; i < channelsCount; i++) {
        for (int j = 0; j < 4; j++) {
            channels[channelsCount - j - 1][i] = pxls[i] % 256;
            pxls[i] >>= 8;
        }
    }  
        
    int *pxf = new int[4];
    for (int i = 0; i < channelsCount; i++) {
        const float b = (1.0 - modXf) * (float) channels[i][0] + modXf * (float) channels[i][1],
            t = (1.0 - modXf) * (float) channels[i][2] + modXf * (float) channels[i][3];
                
        pxf[i] = (int) ((1.0 - modYf) * b + modYf * t);
    }

    unsigned int resPx = 0;
    for (int i = 0; i < channelsCount; i++) {
        resPx += pxf[i] << (8 * (3 - i));
    } 
          
    res[y * newWidth + x] = resPx;

    delete[] pxf;
    for (int i = 0; i < channelsCount; i++) {
        delete[] channels[i];
    }
    delete[] channels;
    delete[] pxls;
}