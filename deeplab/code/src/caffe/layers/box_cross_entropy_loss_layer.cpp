// box cross entropy loss layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include "caffe/layers/box_cross_entropy_loss_layer.hpp"

namespace caffe{

template <typename Dtype>
BoxCrossEntropyLossLayer<Dtype>::~BoxCrossEntropyLossLayer() {
  delete croppings;
  delete boxes;
}

template <typename Dtype>
void BoxCrossEntropyLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  N = bottom[1]->shape(0);
  C = bottom[1]->shape(1);
  H = bottom[1]->shape(2);
  W = bottom[1]->shape(3);
  printf("LayerSetup\n");
  croppings = new Dtype[N*H*W];
  boxes = new Dtype[N*C*H*W];
}
      
template <typename Dtype>
void BoxCrossEntropyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void BoxCrossEntropyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  const Dtype* labels = bottom[2]->cpu_data();
  
  // croppings
  caffe_set(N*H*W, Dtype(1.0), croppings);
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if(((int)(image[0*H*W + h*W + w])==0)&&((int)(image[1*H*W + h*W + w])==0)&&((int)(image[2*H*W + h*W + w])==0))
          croppings[n*H*W + h*W + w] = Dtype(0);
      }
    }
  }
  
  // binary box indicators
  caffe_set(N*C*H*W, Dtype(0.0), boxes);
  for(int n=0;n<N;n++){
    Dtype * box = boxes + n*C*H*W;
    Dtype * cropping = croppings + n*H*W;
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        int label = (int)(labels[n*H*W + h*W + w]);
        // binaryrize this number
        for(int c=0;c<C;c++){
          if(cropping[h*W + w] > Dtype(1e-10))
            box[c*H*W + h*W + w] = (Dtype)(1-label%2);
          label = (label - label%2) / 2;
        }
      }
    }
  }
  //printf("BoxCrossEntropyLossLayer forward\n");
  
  
  
  Dtype * allones = new Dtype[N*C*H*W];
  caffe_set(N*C*H*W, Dtype(1.0), allones);
  //printf("boxes sum is %.2f\n",caffe_cpu_dot(N*C*H*W, boxes, allones));
  Dtype * temp = new Dtype[N*C*H*W];
  // p
  caffe_copy(N*C*H*W, segmentations, temp);
  //printf("segmentations sum is %.2f\n",caffe_cpu_dot(N*C*H*W, segmentations, allones));
  //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
  // - p
  caffe_scal(N*C*H*W, Dtype(-1), temp);
  //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
  // 1-p + 1e-20
  caffe_add_scalar(N*C*H*W, Dtype(1.0+1e-20), temp);
  //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
  // log(1-p)
  caffe_log(N*C*H*W, temp, temp);
  for(int n=0;n<N;n++){
    for(int c=0;c<C;c++){
      for(int h=0;h<H;h++){
        for(int w=0;w<W;w++){
          if(isinf(temp[n*C*H*W + c*H*W + h*W +w])){
            printf("boxes sum is %.2f\n",caffe_cpu_dot(C*H*W, boxes+n*C*H*W, boxes+n*C*H*W));
            printf("temp[n*C*H*W + c*H*W + h*W +w]%.2f\n",temp[n*C*H*W + c*H*W + h*W +w]);
            printf("segmentations[n*C*H*W + c*H*W + h*W +w]%.2f\n",temp[n*C*H*W + c*H*W + h*W +w]);
            exit(-1);
          }
          // binaryrize this number
        }
      }
    }
  }
        
  
  //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
  // -log(1-p)
  caffe_scal(N*C*H*W, Dtype(-1), temp);
  //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
  // sum -log(1-p)
  Dtype crossentropyloss = caffe_cpu_dot(N*C*H*W, temp, boxes);
  delete [] temp;
  
  
  //printf("crossentropyloss is %.2f\n", crossentropyloss);
  
  crossentropyloss = crossentropyloss / N / C / H / W;
  //crossentropyloss = crossentropyloss / caffe_cpu_dot(N*C*H*W, allones, boxes);
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = crossentropyloss;
  
  //printf("top_data[0] is %.2f\n", top_data[0]);
  //exit(-1);
  delete [] allones;
}

template <typename Dtype>
void BoxCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                   const vector<bool> & propagate_down,
                                   const vector<Blob<Dtype>*> & bottom)
{
    
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to images.";
  }
  if (propagate_down[1]) {
    //printf("NC backward\n");
    const Dtype* segmentations = bottom[1]->cpu_data();
    Dtype * bottom_diff = bottom[1]->mutable_cpu_diff();
    Dtype * temp = new Dtype[N*C*H*W];
    Dtype * allones = new Dtype[N*C*H*W];
    caffe_set(N*C*H*W, Dtype(1.0), allones);
    // p
    caffe_copy(N*C*H*W, segmentations, temp);
    //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
    // - p
    caffe_scal(N*C*H*W, Dtype(-1), temp);
    //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
    // 1-p + 1e-20
    caffe_add_scalar(N*C*H*W, Dtype(1.0+1e-20), temp);
    //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
    // 1 / (1-p)
    caffe_powx(N*C*H*W, temp, Dtype(-1), bottom_diff);
    
    for(int n=0;n<N;n++){
      for(int c=0;c<C;c++){
        for(int h=0;h<H;h++){
          for(int w=0;w<W;w++){
            if(bottom_diff[n*C*H*W + c*H*W + h*W +w] > Dtype(1e+2))
              bottom_diff[n*C*H*W + c*H*W + h*W +w] = Dtype(1e+2);
          }
        }
      }
    }
        
    //printf("temp sum is %.2f\n",caffe_cpu_dot(N*C*H*W, temp, allones));
    delete [] temp;
    
    caffe_mul(N*C*H*W, boxes, bottom_diff, bottom_diff);
    
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N / C / H / W;
    //Dtype loss_weight = top[0]->cpu_diff()[0] / caffe_cpu_dot(N*C*H*W, allones, boxes);
    caffe_scal(bottom[1]->count(), loss_weight, bottom_diff);
    delete [] allones;
    //printf("loss_weight is %.2f\n", loss_weight);
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to boxes.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(BoxCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(BoxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(BoxCrossEntropyLoss);

} // namespace caffe
