#ifndef CAFFE_NORMALIZEDCUT_LAYER_HPP_
#define CAFFE_NORMALIZEDCUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <cstddef>
#include "caffe/filterrgbxy.hpp"


namespace caffe{

template <typename Dtype>

class NormalizedCutLayer : public Layer<Dtype>{
 public:
  virtual ~NormalizedCutLayer();
  explicit NormalizedCutLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "NormalizedCut";}
  virtual inline int ExactNumBottomBlobs() const {return 3;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  
 protected:
  Dtype Compute_nc(const Dtype * image, const Dtype * segmentation, int H, int W, const Dtype * degrees_data, Dtype * AS_data, const Dtype * ROI, Permutohedral & permutohedral); 
  void Gradient_nc(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * gradients, const Dtype * degrees_data, const Dtype * AS_data, const Dtype * ROI); 
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  float bi_xy_std_;
  float bi_rgb_std_;
  int channels;
  Blob<Dtype> * degrees;
  Blob<Dtype> * allones;
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

#endif // CAFFE_NORMALIZEDCUT_LAYER_HPP_
