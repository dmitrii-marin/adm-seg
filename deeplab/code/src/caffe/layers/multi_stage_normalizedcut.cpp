/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/tvg_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_stage_normalizedcut_layer.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
MultiStageNormalizedCutLayer<Dtype>::~MultiStageNormalizedCutLayer(){
    delete [] norm_feed_;
    delete [] bilateral_kernel_buffer_;
    delete degrees;
    delete allones;
}

template <typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const caffe::MultiStageMeanfieldParameter meanfield_param = this->layer_param_.multi_stage_meanfield_param();

  num_iterations_ = meanfield_param.num_iterations();

  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";

  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();
  theta_gamma_ = meanfield_param.theta_gamma();
  bi_weight_ = meanfield_param.bilateral_filter_weight();
  temperature_ = meanfield_param.temperature();
    
  
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;
  ROIs.Reshape(num_, 1, height_, width_);
  degrees = new Blob<Dtype>(num_, channels_, height_, width_);
  allones = new Blob<Dtype>(1,1,height_,width_);
  caffe_set(num_pixels_, Dtype(1), allones->mutable_cpu_data());

  LOG(INFO) << "This implementation has not been tested batch size > 1.";

  top[0]->Reshape(num_, channels_, height_, width_);

  // Initialize the parameters that will updated by backpropagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "MultiStageNormalizedCut layer skipping parameter initialization.";
  } else {

    this->blobs_.resize(3);// blobs_[0] - spatial kernel weights, blobs_[1] - bilateral kernel weights, blobs_[2] - compatability matrix

    // Allocate space for kernel weights.
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));

    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[0]->mutable_cpu_data());
    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[1]->mutable_cpu_data());

    // Initialize the kernels weights. The two files spatial.par and bilateral.par should be available.
    FILE * pFile;
    pFile = fopen("spatial.par", "r");
    CHECK(pFile) << "The file 'spatial.par' is not found. Please create it with initial spatial kernel weights.";
    for (int i = 0; i < channels_; i++) {
      fscanf(pFile, "%lf", &this->blobs_[0]->mutable_cpu_data()[i * channels_ + i]);
    }
    fclose(pFile);

    pFile = fopen("bilateral.par", "r");
    CHECK(pFile) << "The file 'bilateral.par' is not found. Please create it with initial bilateral kernel weights.";
    for (int i = 0; i < channels_; i++) {
      fscanf(pFile, "%lf", &this->blobs_[1]->mutable_cpu_data()[i * channels_ + i]);
    }
    fclose(pFile);

    // Initialize the compatibility matrix.
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[2]->mutable_cpu_data());

    // Initialize it to have the Potts model.
    for (int c = 0; c < channels_; ++c) {
      (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
    }
  }

  // Initialize the spatial lattice. This does not need to be computed for every image because we use a fixed size.
  float spatial_kernel[2 * num_pixels_];
  compute_spatial_kernel(spatial_kernel);
  spatial_lattice_.reset(new ModifiedPermutohedral());
  spatial_lattice_->init(spatial_kernel, 2, num_pixels_);

  // Calculate spatial filter normalization factors.
  //norm_feed_.reset(new Dtype[num_pixels_]);
  norm_feed_ = new Dtype[num_pixels_]; // Meng
  //caffe_set(num_pixels_, Dtype(1.0), norm_feed_.get());
  caffe_set(num_pixels_, Dtype(1.0), norm_feed_); // Meng
  spatial_norm_.Reshape(1, 1, height_, width_);
  Dtype* norm_data = spatial_norm_.mutable_cpu_data();
  //spatial_lattice_->compute(norm_data, norm_feed_.get(), 1);
  spatial_lattice_->compute(norm_data, norm_feed_, 1); // Meng
  for (int i = 0; i < num_pixels_; ++i) {
    norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
  }

  // Allocate space for bilateral kernels. This is a temporary buffer used to compute bilateral lattices later.
  // Also allocate space for holding bilateral filter normalization values.
  //bilateral_kernel_buffer_.reset(new float[5 * num_pixels_]);
  bilateral_kernel_buffer_ = new float[5 * num_pixels_];
  bilateral_norms_.Reshape(num_, 1, height_, width_);

  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // It may be possible to optimize this calculation later.
  split_layer_bottom_vec_.clear();
  split_layer_bottom_vec_.push_back(bottom[0]);

  split_layer_top_vec_.clear();

  split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    split_layer_top_vec_.push_back(split_layer_out_blobs_[i].get());
  }

  LayerParameter split_layer_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  split_layer_->SetUp(split_layer_bottom_vec_, split_layer_top_vec_);

  // Make blobs to store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  }

  //check bottom[1]
  /*for(int i=0;i<bottom[1]->count();i++)
    if(isnan(*(bottom[1]->cpu_data()+i)))
        LOG(FATAL) << this->type()
               << " Layer before forward bottom[1]: "<< *(bottom[1]->cpu_data()+i)<<std::endl;*/
               
               
  // Make instances of MeanfieldIteration and initialize them.
  normalizedcut_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    normalizedcut_iterations_[i].reset(new NormalizedCutIteration<Dtype>());
    normalizedcut_iterations_[i]->OneTimeSetUp(
        split_layer_out_blobs_[i].get(), // unary terms
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(), // softmax input
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get(), // output blob
        spatial_lattice_, // spatial lattice
        &spatial_norm_, bi_weight_, temperature_, i); // spatial normalization factors.
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);

  LOG(INFO) << ("MultiStageNormalizedCutLayer initialized.");
  //exit(-1);
}

template <typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Do nothing.
}


/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - RGB images
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // ROI
  caffe_set(num_*height_*width_, Dtype(1.0), ROIs.mutable_cpu_data());
  for(int n=0;n<num_;n++){
    const Dtype * image = bottom[2]->cpu_data() + num_pixels_*3*n;
    Dtype * ROI = ROIs.mutable_cpu_data() + ROIs.offset(n);
    for(int h=0;h<height_;h++){
      for(int w=0;w<width_;w++){
        if(((int)(image[0*num_pixels_ + h*width_ + w])==0)&&((int)(image[1*num_pixels_ + h*width_ + w])==0)&&((int)(image[2*num_pixels_ + h*width_ + w])==0))
          ROI[h*width_ + w] = Dtype(0);
      }
    }
  }
  
  
  split_layer_bottom_vec_[0] = bottom[0];
  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  // Initialize the bilateral lattices.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {

    //compute_bilateral_kernel(bottom[2], n, bilateral_kernel_buffer_.get());
    compute_bilateral_kernel(bottom[2], n, bilateral_kernel_buffer_);
    bilateral_lattices_[n].reset(new ModifiedPermutohedral());
    //bilateral_lattices_[n]->init(bilateral_kernel_buffer_.get(), 5, num_pixels_);
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_, 5, num_pixels_);

    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
    //bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.get(), 1);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_, 1); // Meng
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
    }
  }
  
  // degrees
  for(int n=0;n<num_;n++){ 
    bilateral_lattices_[n]->compute((float *)degrees->mutable_cpu_data() + n*num_pixels_, (float *)(ROIs.cpu_data() + ROIs.offset(n)), 1); // Meng
    //printf("sum of degrees %.8f\n", caffe_cpu_dot(num_pixels_, degrees->mutable_cpu_data() + n*num_pixels_, allones->cpu_data()));
  }
  
  

  for (int i = 0; i < num_iterations_; ++i) {
  

    normalizedcut_iterations_[i]->PrePass(this->blobs_, &bilateral_lattices_, &bilateral_norms_, &ROIs, degrees);

  //check bottom[1]
  /*for(int i=0;i<bottom[1]->count();i++)
    if(isnan(*(bottom[1]->cpu_data()+i)))
        LOG(FATAL) << this->type()
               << " Layer before forward bottom[1]: "<< *(bottom[1]->cpu_data()+i)<<std::endl;*/
               
    normalizedcut_iterations_[i]->Forward_cpu();
  }
}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return; // no backward
  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    normalizedcut_iterations_[i]->Backward_cpu();
  }

  vector<bool> split_layer_propagate_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down, split_layer_bottom_vec_);
  
  Dtype gradientnorm0 = sqrt(caffe_cpu_dot(bottom[0]->count(), bottom[0]->cpu_diff(), bottom[0]->cpu_diff()));
  //printf("gradientnorm0 is %.8f\n", gradientnorm0);
  
  if(isinf(gradientnorm0)){
    LOG(FATAL) << this->type()
               << " Layer gradientnorm: "<< gradientnorm0 <<std::endl;
  }
  
  Dtype gradientnorm1 = sqrt(caffe_cpu_dot(bottom[1]->count(), bottom[1]->cpu_diff(), bottom[1]->cpu_diff()));
  //printf("gradientnorm1 is %.8f\n", gradientnorm1);
  
  if(isinf(gradientnorm1)){
    LOG(FATAL) << this->type()
               << " Layer gradientnorm: "<< gradientnorm1 <<std::endl;
  }
  
  // gradient clipping
  if(gradientnorm0 > 0.001){
    //exit(-1);
    //caffe_scal(bottom[0]->count(), Dtype(0.001) / gradientnorm0, bottom[0]->mutable_cpu_diff());
    //caffe_scal(bottom[1]->count(), Dtype(0.001) / gradientnorm0, bottom[1]->mutable_cpu_diff());
    LOG(INFO) << this->type()
               << " Layer gradient clipping: "<< gradientnorm0 <<std::endl;
    caffe_scal(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
    caffe_scal(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  }
  
  //gradientnorm = sqrt(caffe_cpu_dot(bottom[0]->count(), bottom[0]->cpu_diff(), bottom[0]->cpu_diff()));
  //printf("After clipping gradientnorm is %.8f\n", gradientnorm);
  
  

  // Accumulate diffs from mean field iterations.
  /*for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {

    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();

    if (this->param_propagate_down_[blob_id]) {

      caffe_set(cur_blob->count(), Dtype(0), cur_blob->mutable_cpu_diff());

      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = normalizedcut_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }*/
}

template<typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n,
                                                               float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[5 * p] = static_cast<float>(p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = static_cast<float>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<float>(rgb_data_start[p] / theta_beta_);
    output_kernel[5 * p + 3] = static_cast<float>((rgb_data_start + num_pixels_)[p] / theta_beta_);
    output_kernel[5 * p + 4] = static_cast<float>((rgb_data_start + num_pixels_ * 2)[p] / theta_beta_);
  }
}

template <typename Dtype>
void MultiStageNormalizedCutLayer<Dtype>::compute_spatial_kernel(float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}

INSTANTIATE_CLASS(MultiStageNormalizedCutLayer);
REGISTER_LAYER_CLASS(MultiStageNormalizedCut);
}  // namespace caffe
