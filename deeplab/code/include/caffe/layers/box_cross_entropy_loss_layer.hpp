#ifndef CAFFE_BOX_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_BOX_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <cstddef>

namespace caffe{

template <typename Dtype>

class BoxCrossEntropyLossLayer : public Layer<Dtype>{
 public:
  virtual ~BoxCrossEntropyLossLayer();
  explicit BoxCrossEntropyLossLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "BoxCrossEntropyLoss";}
  virtual inline int ExactNumBottomBlobs() const {return 3;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  int N,C,H,W;
  
  Dtype * croppings;
  Dtype * boxes;
  
};

} // namespace caffe

#endif // CAFFE_BOX_CROSS_ENTROPY_LOSS_LAYER_HPP_
