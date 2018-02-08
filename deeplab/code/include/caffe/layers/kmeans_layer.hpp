#ifndef CAFFE_KMEANS_LAYER_HPP_
#define CAFFE_KMEANS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>

class KMeansLayer : public Layer<Dtype>{
 public:
  virtual ~KMeansLayer();
  explicit KMeansLayer(const LayerParameter & param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "KMeans";}
  virtual inline int ExactNumBottomBlobs() const {return 3;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  
 protected:
  Dtype Compute_kmeans(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * means, const Dtype * ROI);
  void Gradient_kmeans(const Dtype * image, const Dtype * segmentation, int H, int W, Dtype * gradients, Dtype * means,  const Dtype * ROI); 
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  Dtype * Takepatch(int H, int W);
  
  float xy_scale_;
  int channels;
  Blob<Dtype> * means_allimages;
  Dtype * allones;
  bool has_ignore_label_;
  int ignore_label_;
  Dtype * ROI_allimages;
  //Blob<Dtype> * locations;
};

} // namespace caffe

#endif // CAFFE_KMeans_LAYER_HPP_
