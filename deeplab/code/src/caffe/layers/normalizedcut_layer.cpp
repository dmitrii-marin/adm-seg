// normalized cut layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include "caffe/layers/normalizedcut_layer.hpp"

namespace caffe{

template <typename Dtype>
NormalizedCutLayer<Dtype>::~NormalizedCutLayer() {
  delete degrees;
  delete allones;
  delete AS;
  delete ROI_allimages;
  delete [] labelexists;
}

template <typename Dtype>
void NormalizedCutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  NormalizedCutParameter normalized_cut_param = this->layer_param_.normalized_cut_param();
  bi_xy_std_ = normalized_cut_param.bi_xy_std();
  bi_rgb_std_ = normalized_cut_param.bi_rgb_std();
  encode_scribble_ = normalized_cut_param.encode_scribble();
  nonexist_penalty_ = normalized_cut_param.nonexist_penalty();
  channels = bottom[1]->shape(1);
  printf("LayerSetup\n");
  degrees = new Blob<Dtype>(bottom[0]->shape(0),1,bottom[0]->shape(2),bottom[0]->shape(3));
  allones = new Blob<Dtype>(1,1,bottom[0]->shape(2),bottom[0]->shape(3));
  caffe_set(allones->count(), Dtype(1), allones->mutable_cpu_data());
  AS = new Blob<Dtype>(bottom[0]->shape(0),bottom[1]->shape(1),bottom[0]->shape(2),bottom[0]->shape(3));
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  printf("has_ignore_label_ and ignore_label_ %d %d\n", has_ignore_label_, ignore_label_);
  ROI_allimages = new Dtype[bottom[0]->shape(0)*bottom[0]->shape(2)*bottom[0]->shape(3)];
  scribblesegmentations = new Blob<Dtype>(bottom[0]->shape(0),channels,bottom[0]->shape(2),bottom[0]->shape(3)); 
  scribbleROIs = new Blob<Dtype>(bottom[0]->shape(0),channels,bottom[0]->shape(2),bottom[0]->shape(3)); 
  permutohedrals = vector<Permutohedral>(bottom[0]->shape(0));
  labelexists =new bool[bottom[1]->shape(0) * bottom[1]->shape(1)];
}
      
template <typename Dtype>
void NormalizedCutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(1,1,1,1);
  //top[1]->Reshape(bottom[0]->shape(0), 1, bottom[0]->shape(2), bottom[0]->shape(3));
  //printf("image size: %d %d %d\n", bottom[0]->shape(1), bottom[0]->shape(2),bottom[0]->shape(3));
  //printf("probability size: %d %d %d\n", bottom[1]->shape(1), bottom[1]->shape(2),bottom[1]->shape(3));
}

template <typename Dtype>
void NormalizedCutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  const Dtype* labels = bottom[2]->cpu_data();
  int N = bottom[0]->shape(0);
  int H = bottom[0]->shape(2);
  int W = bottom[0]->shape(3);
  
  // segmentation encoded with scribbles
  if(encode_scribble_){
      for(int i=0;i<bottom[1]->shape(0) * bottom[1]->shape(1);i++)
        labelexists[i] = false;
      scribblesegmentations->CopyFrom(*(bottom[1]));
      caffe_set(N*H*W*channels, Dtype(1.0), scribbleROIs->mutable_cpu_data());
      Dtype * scribblesegmentation, * scribbleROI;
      for(int n=0;n<N;n++){
        scribblesegmentation = scribblesegmentations->mutable_cpu_data() + H*W*channels*n;
        scribbleROI = scribbleROIs->mutable_cpu_data() + H*W*channels*n;
        for(int h=0;h<H;h++){
          for(int w=0;w<W;w++){
            int gtlabel = (int)(labels[n*H*W + h*W + w]);
            if(gtlabel==ignore_label_)
              continue;
            labelexists[n * channels + gtlabel] = true;
            for(int c=0;c<channels;c++){
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
    for(int c=0;c<channels;c++){
      if(true== labelexists[ n * channels + c]){
        printf("image %d channel %d exist!\n", n, c);
      }
    }
  }*/
  
  // ROI
  caffe_set(N*H*W, Dtype(1.0), ROI_allimages);
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
  

  // initialize permutohedrals
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    initializePermutohedral((float *)image, W, H, bi_rgb_std_, bi_xy_std_, permutohedrals[n]);
  }
  
  Dtype nc = Dtype(0);
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    // compute degrees
    //clock_t start = clock();
    //filterrgbxy((float *)image, (float *)ROI_allimages + n*H*W, W, H, bi_rgb_std_, bi_rgb_std_, bi_rgb_std_, bi_xy_std_, bi_xy_std_, (float *)degrees->mutable_cpu_data() + n*W*H);
    //clock_t finish = clock();
    //printf("time for filtering %.5f\n", (double)(finish-start)/CLOCKS_PER_SEC);
    
    permutohedrals[n].compute((float *)degrees->mutable_cpu_data() + n*W*H, (float *)ROI_allimages + n*H*W, 1);

    //printf("sum of degrees %.8f\n", caffe_cpu_dot(H*W, degrees->mutable_cpu_data() + n*W*H, allones->cpu_data()));
    //printf("size of Dtype %d\n", sizeof(new Dtype[1]));
    //exit(-1);
    if(!encode_scribble_)
        nc = nc + Compute_nc(image, segmentations + H*W*channels*n, H, W, degrees->cpu_data() + n*W*H, AS->mutable_cpu_data() + n*channels*H*W, ROI_allimages + n*H*W, permutohedrals[n]);
    else
        nc = nc + Compute_nc(image, scribblesegmentations->cpu_data() + H*W*channels*n, H, W, degrees->cpu_data() + n*W*H, AS->mutable_cpu_data() + n*channels*H*W, ROI_allimages + n*H*W, permutohedrals[n]);
    //exit(-1);
  }
  nc = nc / N;
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = nc;
}

template <typename Dtype>
void NormalizedCutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
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
    for(int n=0;n<N;n++){
      const Dtype * image = images + H*W*3*n;
      if(!encode_scribble_)
        Gradient_nc(image, segmentations + H*W*channels*n, H, W, bottom_diff + H*W*channels*n, degrees->cpu_data() + n*W*H, AS->cpu_data()+n*channels*H*W, ROI_allimages + n*H*W);
      else{
        Gradient_nc(image, scribblesegmentations->cpu_data() + H*W*channels*n, H, W, bottom_diff + H*W*channels*n, degrees->cpu_data() + n*W*H, AS->cpu_data()+n*channels*H*W, ROI_allimages + n*H*W);
        caffe_mul(H*W*channels, bottom[1]->cpu_diff() + H*W*channels*n, scribbleROIs->cpu_data() + H*W*channels*n, bottom_diff + H*W*channels*n);
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N;
    caffe_scal(bottom[1]->count(), loss_weight, bottom_diff);
    
    if(encode_scribble_){
        for(int n=0;n<N;n++){
          for(int c=0;c<channels;c++){
            if(false == labelexists[ n * channels + c]){
              caffe_set(H*W, Dtype(nonexist_penalty_), bottom_diff + bottom[1]->offset(n) + c*H*W);
            }
          }
        }
    }
    
    //printf("loss_weight is %.2f\n", loss_weight);
  }
  
}

template <typename Dtype>
Dtype NormalizedCutLayer<Dtype>::Compute_nc(const Dtype * image, const Dtype * segmentation, int H, int W, const Dtype * degrees_data, Dtype * AS_data,  const Dtype * ROI, Permutohedral & permutohedral)
{
  float normalizedassociation = 0;
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<channels;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    //printf("sum of segmentation of this channel %.8f\n", caffe_cpu_dot(H*W, temp, allones->cpu_data()));
    //filterrgbxy((float *)image, (float *)temp, W, H, bi_rgb_std_, bi_rgb_std_, bi_rgb_std_, bi_xy_std_, bi_xy_std_, (float *)AS_data + c*W*H);
    permutohedral.compute((float *)AS_data + c*W*H, (float *)temp, 1);
    //float nominator = 0;
    //float denominator = 0;
    //for(int i=0;i<H * W;i++){
      //nominator = nominator + AS_data[i + c*W*H] * segmentation[c*W*H + i];
      //denominator = denominator + degrees_data[i] * segmentation[c*W*H + i];
    //}
    //for(int i=0;i<H*W;i++)
    //  if(isnan(degrees_data[i]))
    //    LOG(FATAL) << this->type()
    //           << " Layer degrees "<< i <<" "<<degrees_data[i]<<std::endl;
    //for(int i=0;i<H*W;i++)
    //  if(isnan(segmentation[i + c*W*H]))
    //    LOG(FATAL) << this->type()
    //           << " Layer segmentation "<< i <<" "<<segmentation[i + c*W*H]<<std::endl;
               
    Dtype nominator   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    Dtype denominator = caffe_cpu_dot(H*W, temp, degrees_data);
    //normalizedassociation += nominator / (denominator + FLT_MIN);
    normalizedassociation += (nominator) / (denominator + FLT_MIN);
    if(isnan(nominator) || isnan(denominator))
      LOG(FATAL) << this->type()
               << " Layer nominator and denominator: "<< nominator <<" "<<denominator<<std::endl;
    //printf("NA for channel %d: %.7f = %.2f / %.2f \n", c, (nominator) / (denominator + FLT_MIN),(nominator), (denominator + FLT_MIN));
  }
  delete [] temp;
  return - normalizedassociation;
}

template <typename Dtype>
void NormalizedCutLayer<Dtype>::Gradient_nc(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * gradients, const Dtype * degrees_data, const Dtype * AS_data,  const Dtype * ROI){
  
  caffe_set(H*W*channels, Dtype(0), gradients);
  // segmentation in ROI, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<channels;c++){
    caffe_mul(H*W, segmentation+c*W*H, ROI, temp);
    //float nominator = 0;
    //float denominator = 0;
    //for(int i=0;i<H * W;i++){
    //  nominator = nominato[r + AS_data[i+c*H*W] * segmentation[c*W*H + i];
    //  denominator = denominator + degrees_data[i] * segmentation[c*W*H + i];
    //}
    Dtype nominator   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    Dtype denominator = caffe_cpu_dot(H*W, temp, degrees_data);
    Dtype gradientmax = Dtype(-1e+20);
    Dtype gradientmin = Dtype(1e+20);
    Dtype degreemax = Dtype(-1e+20);
    for(int i=0;i<H * W;i++){
      gradients[H*W*c + i] = (degrees_data[i] * nominator - 2 * AS_data[i+c*H*W] * denominator) / (denominator*denominator + FLT_MIN);
      if(degrees_data[i] > degreemax)
        degreemax = degrees_data[i];
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
  for(int c=0;c<channels;c++)
    caffe_mul(H*W, gradients + c*H*W, ROI, gradients + c*H*W);*/
  //exit(-1);
  //printf("end of gradient_nc\n");
}

#ifdef CPU_ONLY
STUB_GPU(NormalizedCutLayer);
#endif

INSTANTIATE_CLASS(NormalizedCutLayer);
REGISTER_LAYER_CLASS(NormalizedCut);

} // namespace caffe
