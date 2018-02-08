#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/kmeans_layer.hpp"

namespace caffe {

template <typename TypeParam>
class KMeansLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  
 protected:
  KMeansLayerTest()
      : blob_bottom_(new Blob<Dtype>(N,3,H,W)),
        blob_bottom2_(new Blob<Dtype>(N,C,H,W)),
        blob_top_(new Blob<Dtype>())
  {
    //Caffe::set_random_seed(1701);
    //FillerParameter filler_param;
    Dtype * image = blob_bottom_->mutable_cpu_data();
    Dtype * segmentation = blob_bottom2_->mutable_cpu_data();
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if((h>1) && (w >=2)){
          segmentation[0*H*W + h*W + w] = 1;
          segmentation[1*H*W + h*W + w] = 0;
          image[0*H*W + h*W + w] = h;
          image[1*H*W + h*W + w] = h;
          image[2*H*W + h*W + w] = h;
        }else{
          segmentation[0*H*W + h*W + w] = 0;
          segmentation[1*H*W + h*W + w] = 1;
          image[0*H*W + h*W + w] = w;
          image[1*H*W + h*W + w] = w;
          image[2*H*W + h*W + w] = w;
        }
      }
    }
    //printf("segmentation sum: %.2f\n", caffe_cpu_asum(H*W, segmentation));
    //printf("segmentation sum: %.2f\n", caffe_cpu_asum(H*W, segmentation + H*W));
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~KMeansLayerTest() {delete blob_bottom_; delete blob_top_;}
  
  void TestForward(Dtype filler_std)
  {
    printf("KMeansLayer Test forward\n");
    //FillerParameter filler_param;
    //filler_param.set_std(filler_std);
    //GaussianFiller<Dtype> filler(filler_param);
    //filler.Fill(this->blob_bottom_);
    
    LayerParameter layer_param;
    KMeansLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    printf("kmeans loss is %.2f\n", top_data[0]);
    //const Dtype min_precision = 1e-5;
    //for(int i=0; i< this->blob_bottom_->count();i++){
    //  Dtype expected_value = sin(bottom_data[i]);
    //  Dtype precision = std::max(
    //    Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
    //  EXPECT_NEAR(expected_value, top_data[i], precision);
    //}
  }
  
  void TestBackward(Dtype filler_std)
  {
    printf("KMeansLayer Back forward\n");
    LayerParameter layer_param;
    KMeansLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    vector<bool> propagate_down;
    propagate_down.push_back(false);
    propagate_down.push_back(true);
    
    blob_top_vec_[0]->mutable_cpu_diff()[0] = Dtype(1.0);
    layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
    
    const Dtype * gradients = blob_bottom_vec_[1]->cpu_diff();
    printf("bottom gradient is:\n");
    for(int c=0;c<C;c++){
      for(int h=0;h<H;h++){
        for(int w=0;w<W;w++){
          printf("%.2f\t",gradients[H*W*c + h*W + w]);
        }
        printf("\n");
      }
    }
    /*FillerParameter filler_param;
    filler_param.set_std(filler_std);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    
    LayerParameter layer_param;
    KMeansLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_, this->blob_top_vec_);*/
  }
  
  int W = 4;
  int H = 3;
  int C = 2;
  int N = 1;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  
}; // KMeansLayerTest

TYPED_TEST_CASE(KMeansLayerTest, TestDtypesAndDevices);
TYPED_TEST(KMeansLayerTest, TestKMeans) {
  this->TestForward(1.0);
}

TYPED_TEST(KMeansLayerTest, TestKMeansGradient) {
  this->TestBackward(1.0);
}


} // namespace caffe
