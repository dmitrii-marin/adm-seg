// KMeans layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>
#include <algorithm>

#include "caffe/layers/kmeans_layer.hpp"

namespace caffe{

template <typename Dtype>
KMeansLayer<Dtype>::~KMeansLayer() {
  delete means_allimages;
  delete allones;
  delete ROI_allimages;
}

template <typename Dtype>
void KMeansLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  KMeansParameter kmeans_param = this->layer_param_.kmeans_param();
  xy_scale_ = kmeans_param.xy_scale();
  channels = bottom[1]->shape(1);
  printf("LayerSetup\n");
  means_allimages = new Blob<Dtype>(bottom[0]->shape(0),channels,5,1);
  allones = new Dtype[bottom[0]->shape(2) * bottom[0]->shape(3)];
  caffe_set(bottom[0]->shape(2) * bottom[0]->shape(3), Dtype(1.0), allones);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  printf("has_ignore_label_ and ignore_label_ %d %d\n", has_ignore_label_, ignore_label_);
  ROI_allimages = new Dtype[bottom[0]->shape(0)*bottom[0]->shape(2)*bottom[0]->shape(3)];
}
      
template <typename Dtype>
void KMeansLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void KMeansLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  caffe_set(means_allimages->count(), Dtype(0), means_allimages->mutable_cpu_data());
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  const Dtype* labels = bottom[2]->cpu_data();
  int N = bottom[0]->shape(0);
  int H = bottom[0]->shape(2);
  int W = bottom[0]->shape(3);
  
  // ROI
  caffe_set(N*H*W, Dtype(1.0), ROI_allimages);
  if(has_ignore_label_){
    for(int n=0;n<N;n++){
      for(int w=0;w<W;w++){
        // Is this column in ROI?
        bool isROI = false;
        for(int h=0;h<H;h++){
          int c = (int)(labels[n*H*W + h*W + w]);
          if(c!=ignore_label_){
            isROI = true;
            break;
          }
        }
        if(isROI) continue;
        for(int h=0;h<H;h++){
          ROI_allimages[n*H*W + h*W + w] = Dtype(0);
        }
      }
      
      for(int h=0;h<H;h++){
        // Is this row in ROI?
        bool isROI = false;
        for(int w=0;w<W;w++){
          int c = (int)(labels[n*H*W + h*W + w]);
          if(c!=ignore_label_){
            isROI = true;
            break;
          }
        }
        if(isROI) continue;
        for(int w=0;w<W;w++){
          ROI_allimages[n*H*W + h*W + w] = Dtype(0);
        }
      }
    }
    
  }
  
  //printf("has_ignore_label_ and ignore_label_ %d %d\n", has_ignore_label_, ignore_label_);
  
  Dtype kmeans = Dtype(0);
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    const Dtype * segmentation = segmentations + H*W*channels*n;
    Dtype * means = means_allimages->mutable_cpu_data() + n * channels * 5;
    kmeans = kmeans + Compute_kmeans(image, segmentation, H, W, means, ROI_allimages + n*H*W);
  }
  kmeans = kmeans / N  / Dtype(255.0 * 255.0);
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = kmeans;
  
}

template <typename Dtype>
void KMeansLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                   const vector<bool> & propagate_down,
                                   const vector<Blob<Dtype>*> & bottom)
{
    
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image inputs.";
  }
  if (propagate_down[1]) {
    //printf("KMeans backward\n");
    const Dtype* images = bottom[0]->cpu_data();
    const Dtype* segmentations = bottom[1]->cpu_data();
    int N = bottom[0]->shape(0);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    for(int n=0;n<N;n++){
      const Dtype * image = images + H*W*3*n;
      const Dtype * segmentation = segmentations + H*W*channels*n;
      Dtype * means = means_allimages->mutable_cpu_data() + n * channels * 5;
      Gradient_kmeans(image, segmentation, H, W, bottom_diff + H*W*channels*n, means, ROI_allimages + n*H*W);
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N / Dtype(255.0 * 255.0);
    caffe_scal(bottom[1]->count(), loss_weight , bottom_diff);
  }
  
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image label.";
  }
  //exit(-1);
  
}

template <typename Dtype>
Dtype KMeansLayer<Dtype>::Compute_kmeans(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * means, const Dtype * ROI)
{
  // compute means
  caffe_set(channels * 5, Dtype(0), means);
  Dtype * temp = new Dtype[H * W];
  for(int c=0;c<channels;c++){
    for(int rgb=0;rgb<3;rgb++){
      caffe_set(H*W, Dtype(0), temp);
      caffe_mul(H*W, image + rgb*H*W, segmentation + c*H*W, temp);
      means[c * 5 + rgb] =  caffe_cpu_dot(H*W, ROI, temp) / (caffe_cpu_dot(H*W, ROI, segmentation + c*H*W) + FLT_MIN);
      if(isnan(caffe_cpu_dot(H*W, ROI, temp))){
        LOG(FATAL) << this->type()
               << " Layer caffe_cpu_dot(H*W, ROI, temp) NAN.";
      }
      //printf("means: %.2f = %.2f / %.2f\n", means[c*5+rgb], caffe_cpu_dot(H*W, ROI, temp), caffe_cpu_dot(H*W, ROI, segmentation + c*H*W));
    }
  }
  Dtype kmeans = 0;
  for(int c=0;c<channels;c++){
    for(int rgb=0;rgb<3;rgb++){
      caffe_copy(H*W, image + rgb * H * W, temp);
      caffe_add_scalar(H*W, means[c*5+rgb]*Dtype(-1), temp);
      caffe_powx(H*W, temp, Dtype(2.0), temp);
      caffe_mul(H*W, temp, ROI, temp);
      caffe_mul(H*W, temp, segmentation + c*H*W, temp);
      kmeans = kmeans + caffe_cpu_asum(H*W, temp);
    }
  }
  delete [] temp;
  //printf("KMeans is %.2f\n", kmeans);
  return kmeans;
  /*float normalizedassociation = 0;
  int labelcount = 0;
  for(int c=0;c<channels;c++){
    if(all_labels == false && label_exist_one_image[c] == false)
      continue;
    labelcount ++;
    filterrgbxy(image, segmentation+c*W*H, W, H, bi_rgb_std_, bi_rgb_std_, bi_rgb_std_, bi_xy_std_, bi_xy_std_, AS_data + c*W*H);
    //float nominator = 0;
    //float denominator = 0;
    //for(int i=0;i<H * W;i++){
      //nominator = nominator + AS_data[i + c*W*H] * segmentation[c*W*H + i];
      //denominator = denominator + degrees_data[i] * segmentation[c*W*H + i];
    //}
    Dtype nominator = 0;
    caffe_cpu_gemm(CblasTrans,
    CblasNoTrans, 1, 1, H*W,
    Dtype(1.0), AS_data + c*W*H , segmentation + c*W*H, Dtype(0),
    & nominator);
    Dtype denominator = 0;
    caffe_cpu_gemm(CblasTrans,
    CblasNoTrans, 1, 1, H*W,
    Dtype(1.0), degrees_data , segmentation + c*W*H, Dtype(0),
    & denominator);
    //normalizedassociation += nominator / (denominator + FLT_MIN);
    normalizedassociation += (nominator) / (denominator + FLT_MIN);
    //printf("NA for channel %d: %.2f \n", c, (nominator) / (denominator + FLT_MIN));
  }
  return labelcount - normalizedassociation;*/
}

template <typename Dtype>
void KMeansLayer<Dtype>::Gradient_kmeans(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * gradients, Dtype * means,  const Dtype * ROI){

  caffe_set(H*W*channels, Dtype(0), gradients);
  Dtype * temp = new Dtype[H * W];
  // reset some means that are trivial
  for(int c=0;c<channels;c++){
    Dtype segmentsize = caffe_cpu_asum(H*W, segmentation + c*H*W);
    if(segmentsize > Dtype(H*W*0.05)) // large segment, continue
      continue;
    // for small segment, take random patch
    Dtype * patch = Takepatch(H, W);
    for(int rgb=0;rgb<3;rgb++){
      caffe_set(H*W, Dtype(0), temp);
      caffe_mul(H*W, image + rgb*H*W, patch, temp);
      means[c * 5 + rgb] = caffe_cpu_dot(H*W, allones, temp) / (caffe_cpu_asum(H*W, patch) + FLT_MIN);
      //printf("means reset: %.2f = %.2f / %.2f\n", means[c*5+rgb], caffe_cpu_dot(H*W, allones, temp), caffe_cpu_asum(H*W, patch));
    }
    delete [] patch;
  }
  
  
  for(int c=0;c<channels;c++){
    Dtype * temp_all = new Dtype[H * W];
    caffe_set(H*W, Dtype(0), temp_all);
    for(int rgb=0;rgb<3;rgb++){
      //printf("In gradient kmeans means %.2f\n", means[c*5+rgb]);
      //printf("Image 0 %.2f\n", *(image + rgb * H * W));
      caffe_copy(H*W, image + rgb * H * W, temp);
      //printf("temp 0 %.2f\n", *(temp));
      caffe_add_scalar(H*W, means[c*5+rgb]*Dtype(-1), temp);
      //printf("temp 0 -mean %.2f\n", *(temp));
      caffe_powx(H*W, temp, Dtype(2.0), temp);
      //printf("temp 0 -mean squared %.2f\n", *(temp));
      caffe_add(H*W, temp, temp_all, temp_all);
      //printf("temp_all 0 %.2f\n", *(temp_all));
    }
    caffe_mul(H*W, temp_all, ROI, temp_all);
    caffe_copy(H*W, temp_all, gradients + c * H * W);
    delete [] temp_all;
  }
  delete [] temp;
  
  /*
  for(int c=0;c<channels;c++){
    if(all_labels == false && label_exist_one_image[c] == false)
      continue;
    //float nominator = 0;
    //float denominator = 0;
    //for(int i=0;i<H * W;i++){
    //  nominator = nominato[r + AS_data[i+c*H*W] * segmentation[c*W*H + i];
    //  denominator = denominator + degrees_data[i] * segmentation[c*W*H + i];
    //}
    Dtype nominator = 0;
    caffe_cpu_gemm(CblasTrans,
    CblasNoTrans, 1, 1, H*W,
    Dtype(1.0), AS_data + c*W*H , segmentation + c*W*H, Dtype(0),
    & nominator);
    Dtype denominator = 0;
    caffe_cpu_gemm(CblasTrans,
    CblasNoTrans, 1, 1, H*W,
    Dtype(1.0), degrees_data , segmentation + c*W*H, Dtype(0),
    & denominator);
    for(int i=0;i<H * W;i++){
      gradients[H*W*c + i] = (degrees_data[i] * nominator - 2 * AS_data[i+c*H*W] * denominator) / (denominator*denominator + FLT_MIN);
    }
  }
  //printf("end of gradient_nc\n");*/
}

template <typename Dtype>
Dtype * KMeansLayer<Dtype>::Takepatch(int H, int W){
  //printf("start takepatch\n");
  Dtype * patch = new Dtype[H*W];
  caffe_set(H*W, Dtype(0.0), patch);
  int l = rand()%W*0.7;
  int r = l + 0.2 * W + rand()%((int)(W - l - 0.2 * W));
  l = std::min(l, W-1);
  l = std::max(l,0);
  r = std::min(r, W-1);
  r = std::max(r,0);
  int t = rand()%H*0.7;
  int b = t + 0.2 * H + rand()%((int)(H - t - 0.2 * H));
  t = std::min(t, H-1);
  t = std::max(t,0);
  b = std::min(b, H-1);
  b = std::max(b,0);
  for(int h=0;h<H;h++){
    for(int w=0;w<H;w++){
      if(w>=l && w<=r && h>=t && h<=b)
        patch[h*W + w] = Dtype(1.0);
    }
  }
  //printf("end takepatch\n");
  return patch;
}

#ifdef CPU_ONLY
STUB_GPU(KMeansLayer);
#endif

INSTANTIATE_CLASS(KMeansLayer);
REGISTER_LAYER_CLASS(KMeans);

} // namespace caffe
