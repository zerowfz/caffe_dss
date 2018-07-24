#ifndef CAFFE_SPP_SEG_LAYER_HPP_
#define CAFFE_SPP_SEG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class SPP_SEGLayer : public Layer<Dtype> {
 public:
  explicit SPP_SEGLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SPPSEG"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const SPP_SEGParameter spp_seg_param);

  int pyramid_height_;
  int bottom_h_, bottom_w_;
  int num_;
  int channels_;
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the internal Split layer that feeds the pooling layers
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  vector<Blob<Dtype>*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_bottom_vecs_;
  /// the internal Pooling layers of different kernel sizes
  vector<shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  vector<Blob<Dtype>*> pooling_outputs_;
  /// conv layer 1*1 reduce the dim 
  vector<ConvolutionLayer<Dtype>*> conv_layers_;
  /// conv layer top vec
  vector<vector<Blob<Dtype>*>*> conv_top_vecs_;
  /// conv_outputs stores the outputs of the conv
  vector<Blob<Dtype>*> conv_outputs_;
  ///relu layer
  vector<ReLULayer<Dtype>*> relu_layers_;
  /// interp layer
  vector<shared_ptr<InterpLayer<Dtype>>> interp_layers_;
  /// interp layer top vec
  vector<vector<Blob<Dtype>*>*> interp_top_vecs_;
  /// interp outputs
  vector<Blob<Dtype>*> interp_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  vector<Blob<Dtype>*> concat_bottom_vec_;
  /// the internal Concat layers that the Flatten layers feed into
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_SPP__SEG_LAYER_HPP_
