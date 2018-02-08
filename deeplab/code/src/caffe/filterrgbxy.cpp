#include "caffe/filterrgbxy.hpp"

void filterrgbxy(const float* image, const float* values, int img_w, int img_h, float sigmar, float sigmag, float sigmab, float sigmax, float sigmay, float* out){
	float * pos;
	float * val;
	val = new float[img_w*img_h];
	pos = new float[img_w*img_h*5];
	//float pos_sum[5]={0,0,0,0,0};
    for(int j=0;j<img_h;j++){
	    for(int i=0;i<img_w;i++){
			int idx = j*img_w + i;
			pos[idx*5+0] = (float)((image[0*img_w*img_h+j*img_w+i])) / sigmar;
			pos[idx*5+1] = (float)((image[1*img_w*img_h+j*img_w+i])) / sigmag;
			pos[idx*5+2] = (float)((image[2*img_w*img_h+j*img_w+i])) / sigmab;
			pos[idx*5+3] = ((float)(i)) / sigmax;
			pos[idx*5+4] = ((float)(j)) / sigmay;
		    val[idx] = (float)(values[idx]);
		}
	}
	//clock_t start = clock();
	PermutohedralLattice::filter(pos, 5, val, 1, img_w*img_h, (float *)out);
	delete [] pos;
	delete [] val;
}

void initializePermutohedral(const float * image, int img_w, int img_h, float sigmargb, float sigmaxy, Permutohedral & lattice_){
    float * features = new float[img_w * img_h * 5];
	for( int j=0; j<img_h; j++ ){
		for( int i=0; i<img_w; i++ ){
		    int idx = j*img_w + i;
			features[idx*5+0] = float(i) / sigmaxy;
			features[idx*5+1] = float(j) / sigmaxy;
			features[idx*5+2] = float(image[0*img_w*img_h + idx]) / sigmargb;
			features[idx*5+3] = float(image[1*img_w*img_h + idx]) / sigmargb;
			features[idx*5+4] = float(image[2*img_w*img_h + idx]) / sigmargb;
		}
    }
    
    lattice_.init( features, 5, img_w * img_h );
    delete [] features;
}

