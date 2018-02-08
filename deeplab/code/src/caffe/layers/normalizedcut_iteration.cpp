/*!
 *  \brief     A helper class for {@link MultiStageNormalizedCutLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
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
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalizedcut_iteration.hpp"

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void NormalizedCutIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype>* const unary_terms,
    Blob<Dtype>* const softmax_input,
    Blob<Dtype>* const output_blob,
    const shared_ptr<ModifiedPermutohedral> spatial_lattice,
    const Blob<Dtype>* const spatial_norm, Dtype bi_weight, Dtype temperature, int idx) {

  spatial_lattice_ = spatial_lattice;
  spatial_norm_ = spatial_norm;

  count_ = unary_terms->count();
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  height_ = unary_terms->height();
  width_ = unary_terms->width();
  num_pixels_ = height_ * width_;
  bi_weight_ = bi_weight;
  temperature_ = temperature;
  idx_ = idx;
  
  allones = new Blob<Dtype>(1,1,height_,width_);
  caffe_set(num_pixels_, Dtype(1), allones->mutable_cpu_data());
  
  //printf("num %d channel %d height %d width %d num_pixels %d\n", num_, channels_, height_, width_, num_pixels_);
  //printf("this->blobs_.size() %d\n", this->blobs_.size());
  //exit(-1);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "NormalizedCut iteration skipping parameter initialization.";
  } else {
    blobs_.resize(3);
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // spatial kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // bilateral kernel weight
    blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // compatibility transform matrix
  }

  pairwise_.Reshape(num_, channels_, height_, width_); //?
  spatial_out_blob_.Reshape(num_, channels_, height_, width_); //?
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_); //?
  message_passing_.Reshape(num_, channels_, height_, width_);

  // Softmax layer configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input);

  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);

  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);
  sum_bottom_vec_.push_back(&pairwise_);

  sum_top_vec_.clear();
  sum_top_vec_.push_back(output_blob);

  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.0)/temperature_);
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.0)/temperature_);
  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);
}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void NormalizedCutIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
    const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
    const Blob<Dtype>* const bilateral_norms,
    const Blob<Dtype>* ROIs,
    const Blob<Dtype>* degrees) {

  bilateral_lattices_ = bilateral_lattices;
  bilateral_norms_ = bilateral_norms;
  
  ROIs_ = ROIs;
  degrees_ = degrees;

  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }
  //printf("prepass \n");
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void NormalizedCutIteration<Dtype>::Forward_cpu() {

  //printf("NormalizedCutIteration %d Forward\n", idx_);
  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = spatial_out_blob_.mutable_cpu_data() + spatial_out_blob_.offset(n);
    const Dtype* prob_input_data = prob_.cpu_data() + prob_.offset(n);

    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_cpu_data() + bilateral_out_blob_.offset(n);
    
    Dtype * temp = new Dtype[height_*width_];
    
    Dtype * bounds = pairwise_.mutable_cpu_data() + pairwise_.offset(n);
    const Dtype * degrees_data = degrees_->cpu_data()+n*num_pixels_;
    for(int c=0;c<channels_;c++){
      caffe_mul(num_pixels_, prob_input_data+c*num_pixels_, ROIs_->cpu_data()+n*num_pixels_, temp);
      //(*bilateral_lattices_)[n]->compute(bilateral_out_data + c*num_pixels_, prob_input_data+c*num_pixels_, 1);
      (*bilateral_lattices_)[n]->compute(bilateral_out_data + c*num_pixels_, temp, 1);
      Dtype nominator   = caffe_cpu_dot(num_pixels_, temp, bilateral_out_data + c*num_pixels_);
      Dtype denominator = caffe_cpu_dot(num_pixels_, temp, degrees_->cpu_data()+n*num_pixels_);
      if(isnan(nominator) || isnan(denominator)){
        LOG(INFO) << this->type()
               << " Layer nominator and denominator: "<< nominator <<" "<<denominator<<std::endl;
        for(int i=0;i<num_pixels_;i++){
          if(isnan(*(prob_input_data+c*num_pixels_+i))){
            LOG(INFO) << this->type()
               << " Layer softmax_bottom_vec_: "<< *(softmax_bottom_vec_[0]->cpu_data()+softmax_bottom_vec_[0]->offset(n)+ +c*num_pixels_+i )<<std::endl;
            LOG(FATAL) << this->type()
               << " Layer prob_input_data: "<< *(prob_input_data+c*num_pixels_+i) <<std::endl;
          }
        }
        
      }
      //printf("nominator and denominator %.5f %.5f\n", nominator, denominator);
      
      // bound nc term
      for(int i=0;i<num_pixels_;i++){
        bounds[num_pixels_*c + i] = (degrees_data[i] * nominator - 2 * (*(bilateral_out_data + c*num_pixels_+i)) * denominator) / (denominator*denominator + FLT_MIN) * bi_weight_;
      }    
      caffe_mul(num_pixels_, bounds + c*num_pixels_, ROIs_->cpu_data()+n*num_pixels_, bounds + c*num_pixels_);
      
      //printf("sum of bounds %.8f\n", caffe_cpu_dot(num_pixels_, bounds + c*num_pixels_, allones->cpu_data()));
    
    }
    delete [] temp;
    //exit(-1);
    //(*bilateral_lattices_)[n]->compute(bilateral_out_data, prob_input_data, channels_, false);
    // Pixel-wise normalization.
    /*for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }*/
  }
  
  /*Dtype  boundsum = 0;
  for(int i=0;i<pairwise_.count();i++){
    boundsum = boundsum + std::abs(*(pairwise_.cpu_data()+i));
  }
  printf("Iteration %d bound abs mean %.15f\n", idx_, boundsum /pairwise_.count());*/
  /*caffe_set(count_, Dtype(0.), message_passing_.mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[0]->cpu_data(), spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[1]->cpu_data(), bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
        channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
        message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 0.,
        pairwise_.mutable_cpu_data() + pairwise_.offset(n));
  }*/

  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  /*for(int i=0;i<sum_bottom_vec_[0]->count();i++)
    if(isnan(*(sum_bottom_vec_[0]->cpu_data()+i)))
        LOG(FATAL) << this->type()
               << " Layer before forward sum_bottom_vec_[0]: "<< *(sum_bottom_vec_[0]->cpu_data()+i)<<std::endl;
  for(int i=0;i<sum_bottom_vec_[1]->count();i++)
    if(isnan(*(sum_bottom_vec_[1]->cpu_data()+i)))
        LOG(FATAL) << this->type()
               << " Layer before forward sum_bottom_vec_[1]: "<< *(sum_bottom_vec_[1]->cpu_data()+i)<<std::endl;*/
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}


template<typename Dtype>
void NormalizedCutIteration<Dtype>::Backward_cpu() {

  //printf("NormalizedCutIteration %d Backward\n", idx_);
  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);
  
  // Meng back propagate from pairwise_ to prob_
  Dtype * S = new Dtype[num_pixels_];
  Dtype * temp = new Dtype[num_pixels_];
  Dtype * temp2 = new Dtype[num_pixels_];
  for (int n = 0; n < num_; ++n){ // for each image
    const Dtype * W = degrees_->cpu_data()+n*num_pixels_; // degrees
    const Dtype * prob_input_data = prob_.cpu_data() + prob_.offset(n); // segmentation
    const Dtype * ROI = ROIs_->cpu_data()+n*num_pixels_;
    

    for(int c=0;c<channels_;c++){ // for each channel
      const Dtype * top_diff = pairwise_.cpu_diff() + pairwise_.offset(n) + c * num_pixels_;
      Dtype * bottom_diff = prob_.mutable_cpu_diff() + prob_.offset(n) + c * num_pixels_;
      
      caffe_mul(num_pixels_, prob_input_data + c * num_pixels_, ROI , S);
      //LOG(INFO) << "S * allones "<< caffe_cpu_dot(num_pixels_, S, allones->cpu_data())<<std::endl;
      if(caffe_cpu_dot(num_pixels_, S, allones->cpu_data()) < 1){
        LOG(INFO) << "S * allones "<< caffe_cpu_dot(num_pixels_, S, allones->cpu_data())<<std::endl;
        //caffe_set(num_pixels_, Dtype(0.01), S);
        //LOG(INFO) << "Rest S to 0.01 to avoid gradient blow out "<<std::endl;
      }
      
      Dtype ws = caffe_cpu_dot(num_pixels_, S, W);
      Dtype wd = caffe_cpu_dot(num_pixels_, W, top_diff);
  
      caffe_copy(num_pixels_, S, temp);
      caffe_scal(num_pixels_, Dtype(-1)*wd / (ws +FLT_MIN), temp);
      for(int i=0;i<num_pixels_;i++){
        if(isnan(temp[i])){
          LOG(INFO) << "wd and ws are "<< wd << ' '<<ws<<std::endl;
          LOG(INFO) << "W * allones "<< caffe_cpu_dot(num_pixels_, W, allones->cpu_data())<<std::endl;
          LOG(INFO) << "S * allones "<< caffe_cpu_dot(num_pixels_, S, allones->cpu_data())<<std::endl;
          LOG(FATAL) << this->type()
               << " Layer temp[i]: "<< temp[i] <<std::endl;
        }
      }
      
      caffe_add(num_pixels_, top_diff, temp, temp);
      // temp is now (I-swd/ws)*diff
      caffe_scal(num_pixels_, bi_weight_, temp);
      // scaling by kernel weight
      // temp is now A(I-swd/ws)*diff*bi_weight
      (*bilateral_lattices_)[n]->compute(temp, temp, 1);
      // temp is now A(I-swd/ws)*diff*bi_weight
      Dtype wtemp = caffe_cpu_dot(num_pixels_, W, temp);
      caffe_copy(num_pixels_, S, temp2);
      caffe_scal(num_pixels_, Dtype(-1)*wtemp / (ws +FLT_MIN), temp2);
      caffe_add(num_pixels_, temp, temp2, temp2);
      
      // scaling by -2 / ws
      caffe_scal(num_pixels_, Dtype(-2) / (ws + FLT_MIN), temp2);
      
      // copy to bottom_diff
      caffe_copy(num_pixels_, temp2, bottom_diff);
      
    }
  }
  delete [] S;
  delete [] temp;
  delete [] temp2;
  
  // scaling by kernel weight
  //caffe_scal(prob_.count(), bi_weight_, prob_.mutable_cpu_diff());
  
  /*Dtype gradientsum = 0;
  for(int i=0;i<prob_.count();i++){
    gradientsum = gradientsum + std::abs(*(prob_.cpu_diff()+i));
  }
  printf("Iteration %d probs gradient abs mean %.15f\n", idx_, gradientsum / prob_.count());
  
  if(isnan(gradientsum)){
        LOG(FATAL) << this->type()
               << " Layer gradientsum: "<< gradientsum <<std::endl;
  }*/

  /*//---------------------------- Update compatibility diffs ------------------
  caffe_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., pairwise_.cpu_diff() + pairwise_.offset(n),
                          message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 1.,
                          this->blobs_[2]->mutable_cpu_diff());
  }

  //-------------------------- Gradient after compatibility transform--- -----
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                          pairwise_.cpu_diff() + pairwise_.offset(n), (Dtype) 0.,
                          message_passing_.mutable_cpu_diff() + message_passing_.offset(n));
  }

  // ------------------------- Gradient w.r.t. kernels weights ------------
  caffe_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }

  // TODO: Check whether there's a way to improve the accuracy of this calculation.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[0]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[1]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }


  //---------------------------- BP thru normalization --------------------------
  for (int n = 0; n < num_; ++n) {

    Dtype *spatial_out_diff = spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
                spatial_out_diff + channel_id * num_pixels_,
                spatial_out_diff + channel_id * num_pixels_);
    }

    Dtype *bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
                bilateral_out_diff + channel_id * num_pixels_,
                bilateral_out_diff + channel_id * num_pixels_);
    }
  }

  //--------------------------- Gradient for message passing ---------------
  for (int n = 0; n < num_; ++n) {

    spatial_lattice_->compute(prob_.mutable_cpu_diff() + prob_.offset(n),
                              spatial_out_blob_.cpu_diff() + spatial_out_blob_.offset(n), channels_,
                              true, false);

    (*bilateral_lattices_)[n]->compute(prob_.mutable_cpu_diff() + prob_.offset(n),
                                       bilateral_out_blob_.cpu_diff() + bilateral_out_blob_.offset(n),
                                       channels_, true, true);
  }*/

  //--------------------------------------------------------------------------------
  vector<bool> propagate_down(2, true);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
  
  /*gradientsum = 0;
  for(int i=0;i<softmax_bottom_vec_[0]->count();i++){
    gradientsum = gradientsum + std::abs(*(softmax_bottom_vec_[0]->cpu_diff()+i));
  }
  printf("Iteration %d score gradient abs mean %.15f\n", idx_, gradientsum /softmax_bottom_vec_[0]->count());*/
}

INSTANTIATE_CLASS(NormalizedCutIteration);
}  // namespace caffe
