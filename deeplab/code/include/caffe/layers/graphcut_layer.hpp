#ifndef CAFFE_GRAPHCUT_LAYER_HPP_
#define CAFFE_GRAPHCUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filterrgbxy.hpp"

#include "gco-v3.0/GCoptimization.h"

#include <cstddef>

namespace caffe{

template <typename Dtype>

class GraphCutLayer : public Layer<Dtype>{
 public:
  virtual ~GraphCutLayer();
  explicit GraphCutLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "GraphCut";}
  virtual inline int ExactNumBottomBlobs() const {return 2;}
  virtual inline int ExactNumTopBlobs() const {return 2;}
  
 protected:
  void runGraphCut(const Dtype * image, const Dtype * unary, Dtype * gc_segmentation, bool * ROI);
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  int N, C, H, W;

  float sigma;
  int max_iter;
  float potts_weight;
  Blob<Dtype> * unaries;
  bool * ROI_allimages;


  
};

} // namespace caffe

#endif // CAFFE_GRAPHCUT_LAYER_HPP_
