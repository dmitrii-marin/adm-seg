#ifndef CAFFE_NORMALIZEDCUTBOUND_LAYER_HPP_
#define CAFFE_NORMALIZEDCUTBOUND_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/modified_permutohedral.hpp"

namespace caffe{

template <typename Dtype>

class NormalizedCutBoundLayer : public Layer<Dtype>{
 public:
  virtual ~NormalizedCutBoundLayer();
  explicit NormalizedCutBoundLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "NormalizedCutBound";}
  virtual inline int ExactNumBottomBlobs() const {return 2;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel);
  
 protected:
  void Compute_ncbound(const Dtype * image, const Dtype * segmentation, const Dtype * degrees_data, Dtype * AS_data, const Dtype * ROI, Dtype * bounds,  ModifiedPermutohedral & permutohedral); 
  void Gradient_ncbound(const Dtype * image, const Dtype * segmentation, const Dtype * degrees_data, const Dtype * AS_data, const Dtype * ROI, Dtype * gradients); 
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  float bi_xy_std_;
  float bi_rgb_std_;
  float bi_weight_;
  
  // size
  int N, C, H, W;
  Blob<Dtype> * degrees;
  Blob<Dtype> * allones;
  Blob<Dtype> * AS; //A * S
  
  bool has_ignore_label_;
  int ignore_label_;
  Blob<Dtype> ROIs;
  
  vector<shared_ptr<ModifiedPermutohedral> > bilateral_lattices_;
  float * bilateral_kernel_buffer_;
};

} // namespace caffe

#endif // CAFFE_NORMALIZEDCUTBOUND_LAYER_HPP_
