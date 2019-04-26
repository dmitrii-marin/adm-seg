// cut layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include "caffe/layers/cut_layer.hpp"

namespace caffe{

template <typename Dtype>
CutLayer<Dtype>::~CutLayer() {
  delete AS;
  delete ROI_allimages;
  delete [] labelexists;
}

template <typename Dtype>
void CutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  // blob size
  N = bottom[0]->shape(0);
  C = bottom[1]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);
  // Gaussian kernel parameters
  NormalizedCutParameter normalized_cut_param = this->layer_param_.normalized_cut_param();
  bi_xy_std_ = normalized_cut_param.bi_xy_std();
  bi_rgb_std_ = normalized_cut_param.bi_rgb_std();
  encode_scribble_ = normalized_cut_param.encode_scribble();
  nonexist_penalty_ = normalized_cut_param.nonexist_penalty();
  printf("LayerSetup\n");
  AS = new Blob<Dtype>(N,C,H,W);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  printf("has_ignore_label_ and ignore_label_ %d %d\n", has_ignore_label_, ignore_label_);
  ROI_allimages = new Dtype[N*H*W];
  scribblesegmentations = new Blob<Dtype>(N,C,H,W); 
  scribbleROIs = new Blob<Dtype>(N,C,H,W); 
  permutohedrals = vector<Permutohedral>(N);
  labelexists =new bool[N*C];
}
      
template <typename Dtype>
void CutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void CutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  const Dtype* labels = bottom[2]->cpu_data();
  
  // segmentation encoded with scribbles
  if(encode_scribble_){
      for(int i=0;i<N*C;i++)
        labelexists[i] = false;
      scribblesegmentations->CopyFrom(*(bottom[1]));
      caffe_set(N*C*H*W, Dtype(1.0), scribbleROIs->mutable_cpu_data());
      Dtype * scribblesegmentation, * scribbleROI;
      for(int n=0;n<N;n++){
        scribblesegmentation = scribblesegmentations->mutable_cpu_data() + H*W*C*n;
        scribbleROI = scribbleROIs->mutable_cpu_data() + H*W*C*n;
        for(int h=0;h<H;h++){
          for(int w=0;w<W;w++){
            int gtlabel = (int)(labels[n*H*W + h*W + w]);
            if(gtlabel==ignore_label_)
              continue;
            labelexists[n * C + gtlabel] = true;
            for(int c=0;c<C;c++){
              if(c==gtlabel)
                scribblesegmentation[c*H*W + h*W + w] = 1.0;
              else
                scribblesegmentation[c*H*W + h*W + w] = 0.0;
              scribbleROI[c*H*W + h*W + w] = Dtype(0);
            }
          }
        }
      }
  }
  
  // print labelexists
  /*for(int n=0;n<N;n++){
    for(int c=0;c<C;c++){
      if(true== labelexists[ n * C + c]){
        printf("image %d channel %d exist!\n", n, c);
      }
    }
  }*/
  
  // ROI
  caffe_set(N*H*W, Dtype(1.0), ROI_allimages);
#pragma omp parallel for
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if(((int)(image[0*H*W + h*W + w])==0)&&((int)(image[1*H*W + h*W + w])==0)&&((int)(image[2*H*W + h*W + w])==0))
          ROI_allimages[n*H*W + h*W + w] = Dtype(0);
      }
    }
  }
    
  //printf("NC forward\n");
  //printf("bi std %.2f %.2f\n", bi_xy_std_, bi_rgb_std_);
  
  Dtype nc = Dtype(0);
#if 1
  // initialize permutohedrals
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    initializePermutohedral((float *)image, W, H, bi_rgb_std_, bi_xy_std_, permutohedrals[n]);
  }
  
#pragma omp parallel for reduction(+: nc)
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    
    //printf("size of Dtype %d\n", sizeof(new Dtype[1]));
    //exit(-1);
    if(!encode_scribble_)
        nc = nc + Compute_cut(image, segmentations + H*W*C*n, AS->mutable_cpu_data() + n*C*H*W, ROI_allimages + n*H*W, permutohedrals[n]);
    else
        nc = nc + Compute_cut(image, scribblesegmentations->cpu_data() + H*W*C*n, AS->mutable_cpu_data() + n*C*H*W, ROI_allimages + n*H*W, permutohedrals[n]);
    //exit(-1);
  }
  nc = nc / N;
#endif
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = nc;
}

template <typename Dtype>
void CutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
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
    int N = bottom[0]->shape(0);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
#pragma omp parallel for
    for(int n=0;n<N;n++){
      const Dtype * image = images + H*W*3*n;
      if(!encode_scribble_)
        Gradient_cut(image, segmentations + H*W*C*n, bottom_diff + H*W*C*n, AS->cpu_data()+n*C*H*W, ROI_allimages + n*H*W);
      else{
        Gradient_cut(image, scribblesegmentations->cpu_data() + H*W*C*n, bottom_diff + H*W*C*n, AS->cpu_data()+n*C*H*W, ROI_allimages + n*H*W);
        caffe_mul(H*W*C, bottom[1]->cpu_diff() + H*W*C*n, scribbleROIs->cpu_data() + H*W*C*n, bottom_diff + H*W*C*n);
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N;
    caffe_scal(bottom[1]->count(), loss_weight, bottom_diff);
    
    if(encode_scribble_){
        for(int n=0;n<N;n++){
          for(int c=0;c<C;c++){
            if(false == labelexists[ n * C + c]){
              caffe_set(H*W, Dtype(nonexist_penalty_), bottom_diff + bottom[1]->offset(n) + c*H*W);
            }
          }
        }
    }
    
    //printf("loss_weight is %.2f\n", loss_weight);
  }
  
}

template <typename Dtype>
Dtype CutLayer<Dtype>::Compute_cut(const Dtype * image, const Dtype * segmentation, Dtype * AS_data,  const Dtype * ROI, Permutohedral & permutohedral)
{
  float normalizedassociation = 0;
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    //printf("sum of segmentation of this channel %.8f\n", caffe_cpu_dot(H*W, temp, allones->cpu_data()));
    //filterrgbxy((float *)image, (float *)temp, W, H, bi_rgb_std_, bi_rgb_std_, bi_rgb_std_, bi_xy_std_, bi_xy_std_, (float *)AS_data + c*W*H);
    permutohedral.compute((float *)AS_data + c*W*H, (float *)temp, 1);
    
               
    Dtype SAS   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    normalizedassociation += SAS;
    if(isnan(SAS))
      LOG(FATAL) << this->type()
               << " Layer SAS: "<< SAS <<std::endl;
    //printf("NA for channel %d: %.7f = %.2f / %.2f \n", c, (nominator) / (denominator + FLT_MIN),(nominator), (denominator + FLT_MIN));
  }
  delete [] temp;
  return - normalizedassociation;
}

template <typename Dtype>
void CutLayer<Dtype>::Gradient_cut(const Dtype * image, const Dtype * segmentation, Dtype * gradients, const Dtype * AS_data,  const Dtype * ROI){
  
  caffe_set(H*W*C, Dtype(0), gradients);
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    Dtype nominator   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    Dtype gradientmax = Dtype(-1e+20);
    Dtype gradientmin = Dtype(1e+20);
    for(int i=0;i<H * W;i++){
      gradients[H*W*c + i] = - 2 * AS_data[i+c*H*W];
      if(gradients[H*W*c + i] > gradientmax)
        gradientmax = gradients[H*W*c + i];
      if(gradients[H*W*c + i] < gradientmin)
        gradientmin = gradients[H*W*c + i];
      if(isnan(gradients[H*W*c + i]))
        LOG(FATAL) << this->type()
               << " Layer gradient is nan!"<<std::endl;
    }    
    caffe_mul(H*W, gradients + c*H*W, ROI, gradients + c*H*W);
    //printf("gradient max and min %.20f %.20f \n", gradientmax, gradientmin);
    //printf("degree max %.2f \n", degreemax);
  }
  /*for(int i=0;i<H * W;i++){
    Dtype g0 = gradients[H*W*0 + i];
    Dtype g1 = gradients[H*W*1 + i];
    if(g0 < g1){ // (x0, x1) should be (1, 0)
      gradients[H*W*0 + i] = - (1.0 - segmentation[0*H*W + i]);
      gradients[H*W*1 + i] = - (0.0 - segmentation[1*H*W + i]);
    }else{ // (x0, x1) should be (0, 1)
      gradients[H*W*0 + i] = - (0.0 - segmentation[0*H*W + i]);
      gradients[H*W*1 + i] = - (1.0 - segmentation[1*H*W + i]);
    }
  }
  for(int c=0;c<C;c++)
    caffe_mul(H*W, gradients + c*H*W, ROI, gradients + c*H*W);*/
  //exit(-1);
  //printf("end of gradient_cut\n");
}

#ifdef CPU_ONLY
STUB_GPU(CutLayer);
#endif

INSTANTIATE_CLASS(CutLayer);
REGISTER_LAYER_CLASS(Cut);

} // namespace caffe
