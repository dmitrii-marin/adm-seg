// normalized cut layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include "caffe/layers/normalizedcutbound_layer.hpp"

namespace caffe{

template <typename Dtype>
NormalizedCutBoundLayer<Dtype>::~NormalizedCutBoundLayer() {
  delete degrees;
  delete allones;
  delete AS;
  delete [] bilateral_kernel_buffer_;
}

template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  NormalizedCutParameter normalized_cut_param = this->layer_param_.normalized_cut_param();
  bi_xy_std_ = normalized_cut_param.bi_xy_std();
  bi_rgb_std_ = normalized_cut_param.bi_rgb_std();
  bi_weight_ = normalized_cut_param.bi_weight();
  N = bottom[1]->shape(0);
  C = bottom[1]->shape(1);
  H = bottom[1]->shape(2);
  W = bottom[1]->shape(3);
  printf("LayerSetup\n");
  degrees = new Blob<Dtype>(N,1,H,W);
  allones = new Blob<Dtype>(1,1,H,W);
  caffe_set(H*W, Dtype(1), allones->mutable_cpu_data());
  AS = new Blob<Dtype>(N, C, H, W);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  printf("has_ignore_label_ and ignore_label_ %d %d\n", has_ignore_label_, ignore_label_);
  ROIs.Reshape(N, 1, H, W);
  
  bilateral_kernel_buffer_ = new float[5 * H*W];
}
      
template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(N, C, H, W);
  //printf("image size: %d %d %d\n", bottom[0]->shape(1), bottom[0]->shape(2),bottom[0]->shape(3));
  //printf("probability size: %d %d %d\n", bottom[1]->shape(1), bottom[1]->shape(2),bottom[1]->shape(3));
}

template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  // ROI
  caffe_set(N*H*W, Dtype(1.0), ROIs.mutable_cpu_data());
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    Dtype * ROI = ROIs.mutable_cpu_data() + ROIs.offset(n);
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if(((int)(image[0*H*W + h*W + w])==0)&&((int)(image[1*H*W + h*W + w])==0)&&((int)(image[2*H*W + h*W + w])==0))
          ROI[h*W + w] = Dtype(0);
      }
    }
  }
  //exit(-1);
  // initialize modified permutohedrals
  bilateral_lattices_.resize(N);
  for (int n = 0; n < N; ++n) {
    compute_bilateral_kernel(bottom[0], n, bilateral_kernel_buffer_);
    bilateral_lattices_[n].reset(new ModifiedPermutohedral());
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_, 5, H*W);
  }
  
  //printf("NC forward\n");
  //printf("bi std %.2f %.2f\n", bi_xy_std_, bi_rgb_std_);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    const Dtype * segmentation = segmentations + H*W*C*n;
    Dtype * bounds = top_data + H*W*C*n;
    // compute degrees
    
    bilateral_lattices_[n]->compute((float *)degrees->mutable_cpu_data() + n*W*H, (float *)(ROIs.cpu_data() + ROIs.offset(n)), 1); // Meng
    
    //printf("sum of degrees %.8f\n", caffe_cpu_dot(H*W, degrees->mutable_cpu_data() + n*W*H, allones->cpu_data()));
    //printf("size of Dtype %d\n", sizeof(new Dtype[1]));
    //exit(-1);
    Compute_ncbound(image, segmentation, degrees->cpu_data() + n*W*H, AS->mutable_cpu_data() + n*C*H*W, ROIs.cpu_data() + ROIs.offset(n), bounds, *(bilateral_lattices_[n]));
  }

}

template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                   const vector<bool> & propagate_down,
                                   const vector<Blob<Dtype>*> & bottom)
{
    
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image inputs.";
  }
  if (propagate_down[1]) {
    //printf("NC backward\n");
    const Dtype* images = bottom[0]->cpu_data();
    const Dtype* segmentations = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    for(int n=0;n<N;n++){
      const Dtype * image = images + H*W*3*n;
      const Dtype * segmentation = segmentations + H*W*C*n;
      Gradient_ncbound(image, segmentation, degrees->cpu_data() + n*W*H, AS->cpu_data()+n*C*H*W, ROIs.cpu_data() + ROIs.offset(n), bottom_diff + H*W*C*n);
    }
    // Scale gradient
    //Dtype loss_weight = top[0]->cpu_diff()[0] / N;
    //caffe_scal(bottom[1]->count(), loss_weight, bottom_diff);
    
    //printf("loss_weight is %.2f\n", loss_weight);
  }
  
}

template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::Compute_ncbound(const Dtype * image, const Dtype * segmentation, const Dtype * degrees_data, Dtype * AS_data,  const Dtype * ROI, Dtype * bounds, ModifiedPermutohedral & permutohedral)
{
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    permutohedral.compute((float *)AS_data + c*W*H, (float *)temp, 1);
    
    Dtype nominator   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    Dtype denominator = caffe_cpu_dot(H*W, temp, degrees_data);
    if(isnan(nominator) || isnan(denominator))
      LOG(FATAL) << this->type()
               << " Layer nominator and denominator: "<< nominator <<" "<<denominator<<std::endl;
    //printf("NA for channel %d: %.7f = %.2f / %.2f \n", c, (nominator) / (denominator + FLT_MIN),(nominator), (denominator + FLT_MIN));
    for(int i=0;i<H * W;i++){
      bounds[H*W*c + i] = (degrees_data[i] * nominator - 2 * AS_data[i+c*H*W] * denominator) / (denominator*denominator + FLT_MIN) * bi_weight_;
    }    
    caffe_mul(H*W, bounds + c*H*W, ROI, bounds + c*H*W);
    //printf("nominator and denominator %.5f %.5f\n", nominator, denominator);
    
    //printf("sum of bounds %.8f\n", caffe_cpu_dot(H*W, bounds + c*H*W, allones->cpu_data()));
          
  }
  //exit(-1);
  delete [] temp;
}

template <typename Dtype>
void NormalizedCutBoundLayer<Dtype>::Gradient_ncbound(const Dtype * image, const Dtype * segmentation, const Dtype * degrees_data, const Dtype * AS_data,  const Dtype * ROI, Dtype * gradients){
  
  caffe_set(H*W*C, Dtype(0), gradients);
  return;
  
  // not implemented
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    Dtype nominator   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    Dtype denominator = caffe_cpu_dot(H*W, temp, degrees_data);
    for(int i=0;i<H * W;i++){
      gradients[H*W*c + i] = (degrees_data[i] * nominator - 2 * AS_data[i+c*H*W] * denominator) / (denominator*denominator + FLT_MIN);
      if(isnan(gradients[H*W*c + i]))
        LOG(FATAL) << this->type()
               << " Layer gradient is nan!"<<std::endl;
    }    
    caffe_mul(H*W, gradients + c*H*W, ROI, gradients + c*H*W);
    //printf("gradient max and min %.20f %.20f \n", gradientmax, gradientmin);
    //printf("degree max %.2f \n", degreemax);
  }
}

template<typename Dtype>
void NormalizedCutBoundLayer<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel) {

  for (int p = 0; p < H*W; ++p) {
    output_kernel[5 * p] = static_cast<float>(p % W) / bi_xy_std_;
    output_kernel[5 * p + 1] = static_cast<float>(p / W) / bi_xy_std_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<float>(rgb_data_start[p] / bi_rgb_std_);
    output_kernel[5 * p + 3] = static_cast<float>((rgb_data_start + H*W)[p] / bi_rgb_std_);
    output_kernel[5 * p + 4] = static_cast<float>((rgb_data_start + H*W * 2)[p] / bi_rgb_std_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizedCutBoundLayer);
#endif

INSTANTIATE_CLASS(NormalizedCutBoundLayer);
REGISTER_LAYER_CLASS(NormalizedCutBound);

} // namespace caffe
