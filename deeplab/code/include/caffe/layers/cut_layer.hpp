#ifndef CAFFE_CUT_LAYER_HPP_
#define CAFFE_CUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <cstddef>
#include "caffe/filterrgbxy.hpp"

namespace caffe{

template <typename Dtype>

class CutLayer : public Layer<Dtype>{
 public:
  virtual ~CutLayer();
  explicit CutLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "Cut";}
  virtual inline int ExactNumBottomBlobs() const {return 3;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  
 protected:
  Dtype Compute_cut(const Dtype * image, const Dtype * segmentation, Dtype * AS_data, const Dtype * ROI, Permutohedral & permutohedral); 
  void Gradient_cut(const Dtype * image, const Dtype * segmentation, Dtype * gradients, const Dtype * AS_data, const Dtype * ROI); 
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  int N, C, H, W;
  
  float bi_xy_std_;
  float bi_rgb_std_;
  Blob<Dtype> * AS; //A * S
  
  bool has_ignore_label_;
  int ignore_label_;
  Dtype * ROI_allimages;
  
  Blob<Dtype> * scribblesegmentations;
  Blob<Dtype> * scribbleROIs;
  bool * labelexists;
  
  bool encode_scribble_;
  float nonexist_penalty_;
  
  vector<Permutohedral> permutohedrals;
  
};

} // namespace caffe

#endif // CAFFE_CUT_LAYER_HPP_
