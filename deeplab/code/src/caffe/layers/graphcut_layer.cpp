// graph cut layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include <omp.h>

#include "caffe/layers/graphcut_layer.hpp"


namespace caffe{

template <typename Dtype>
GraphCutLayer<Dtype>::~GraphCutLayer() {
  delete unaries;
  delete ROI_allimages;
}

template <typename Dtype>
void GraphCutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  // blob size
  N = bottom[0]->shape(0);
  C = bottom[1]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);
  // Graph Cut parameters
  GraphCutParameter graphcut_param = this->layer_param_.graphcut_param();
  max_iter = graphcut_param.max_iter();
  potts_weight = graphcut_param.potts_weight();
  
  printf("max_iter is %d\n", max_iter);
  printf("potts_weight is %.2f\n", potts_weight);
  unaries = new Blob<Dtype>(N,C,H,W); 
  ROI_allimages = new bool[N*H*W];
  printf("LayerSetup\n");
}
      
template <typename Dtype>
void GraphCutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(N,1,H,W);
  top[1]->Reshape(N,C,H,W);
}

template <typename Dtype>
void GraphCutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* probs = bottom[1]->cpu_data();

  // from probability to -log(p)
  caffe_log(N*C*H*W, probs, unaries->mutable_cpu_data());
  caffe_scal(N*C*H*W, Dtype(-1), unaries->mutable_cpu_data());


  // ROI
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if(((int)(image[0*H*W + h*W + w])==0)&&((int)(image[1*H*W + h*W + w])==0)&&((int)(image[2*H*W + h*W + w])==0))
          ROI_allimages[n*H*W + h*W + w] = false;
        else
          ROI_allimages[n*H*W + h*W + w] = true;
      }
    }
  }

  clock_t start = clock();

  const int maxNumThreads = omp_get_max_threads();
  //printf("Maximum number of threads for this machine: %i\n", maxNumThreads);
  //printf("Total number of cores in the CPU: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));

  //omp_set_num_threads(6);

  Dtype * gcsegmentations = top[0]->mutable_cpu_data();
  #pragma omp parallel for
  for(int n=0;n<N;n++){
    runGraphCut(images + H*W*3*n, unaries->cpu_data() + n*C*H*W, gcsegmentations+n*H*W, ROI_allimages + n*H*W);
    //printf(" Thread %d: %d\n", omp_get_thread_num(), n);
    //printf("Number of threads: %d\n", omp_get_num_threads());
  }

  Dtype * gcsegmentations_matrix = top[1]->mutable_cpu_data();
  caffe_set(N*C*H*W, Dtype(0), gcsegmentations_matrix);
  for(int n=0;n<N;n++){
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        int label = (int) gcsegmentations[n*H*W + h*W + w];
        gcsegmentations_matrix[n*C*H*W + label*H*W + h*W + w] = Dtype(1);
      }
    }
  }
  //printf("time for graph cut %.5f\n", (double)(clock()-start)/CLOCKS_PER_SEC);

}


template <typename Dtype>
void GraphCutLayer<Dtype>::runGraphCut(const Dtype * image, const Dtype * unary, Dtype * gc_segmentation, bool * ROI){
  for(int h=0;h<H;h++){
    for(int w=0;w<W;w++){
      Dtype minsofar = unary[0*H*W + h*W + w];
      Dtype label = Dtype(0);
      for(int c=1;c<C;c++){
        if(unary[c*H*W + h*W + w] < minsofar){
          minsofar = unary[c*H*W + h*W + w];
          label = c;
        }
      }
      gc_segmentation[h*W + w] = label;
    }
  }

  GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(W*H,C);

  // first set up data costs individually
  for (int c=0;c<C;c++){
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        gc->setDataCost(h*W + w, c , 1000 * unary[c*H*W + h*W + w]);
      }
    }
  }

  // next set up the array for smooth costs
  int *smooth = new int[C*C];
  for ( int l1 = 0; l1 < C; l1++ )
    for (int l2 = 0; l2 < C; l2++ )
      smooth[l1+l2*C] = (l1!=l2)  ? 1 :0;
  gc->setSmoothCost(smooth);

  // expected squared distance
  double diff_total = 0;
  int diff_count = 0;
  for (int h = 0; h < H; h++ )
    for (int  w = 0; w < W; w++ ){
      if(ROI[h*W + w]==false)
        ;//continue;
      for(int i=0;i<4;i++){
        int h2, w2;
        switch(i) {
          case 0:
            h2 = h; w2 = w+1;
		    break;       
          case 1 :
            h2 = h+1; w2 = w;
		    break;
          case 2:
            h2 = h+1; w2 = w+1;
		    break;       
          case 3 :
            h2 = h-1; w2 = w+1;
		    break;
          default:
            break;
	    }
        if(w2<0 || w2>=W || h2<0 || h2>=H)
          continue;
        if(ROI[h2*W + w2]==false)
          ;//continue;
        Dtype diff_b = image[0*H*W + h*W + w] - image[0*H*W + h2*W + w2];
        Dtype diff_g = image[1*H*W + h*W + w] - image[1*H*W + h2*W + w2];
        Dtype diff_r = image[2*H*W + h*W + w] - image[2*H*W + h2*W + w2];
        diff_total += diff_b*diff_b + diff_g*diff_g+diff_r*diff_r;
        diff_count++;
      }
      
    }

  double sigmasquared = diff_total / diff_count;
  //printf("sigma is %f\n",sqrt(sigmasquared));

  // now set up a grid neighborhood system
  for (int h = 0; h < H; h++ )
    for (int  w = 0; w < W; w++ ){
      if(ROI[h*W + w]==false)
        ;//continue;
      for(int i=0;i<4;i++){
        int h2, w2;
        switch(i) {
          case 0:
            h2 = h; w2 = w+1;
		    break;       
          case 1 :
            h2 = h+1; w2 = w;
		    break;
          case 2:
            h2 = h+1; w2 = w+1;
		    break;       
          case 3 :
            h2 = h-1; w2 = w+1;
		    break;
          default:
            break;
	    }
        if(w2<0 || w2>=W || h2<0 || h2>=H)
          continue;
        if(ROI[h2*W + w2]==false)
          ;//continue;
        Dtype diff_b = image[0*H*W + h*W + w] - image[0*H*W + h2*W + w2];
        Dtype diff_g = image[1*H*W + h*W + w] - image[1*H*W + h2*W + w2];
        Dtype diff_r = image[2*H*W + h*W + w] - image[2*H*W + h2*W + w2];
        Dtype diff = diff_b*diff_b + diff_g*diff_g+diff_r*diff_r;
        if(i==0 || i==1)
          gc->setNeighbors(h*W + w, h2*W + w2,1000*potts_weight*exp(-diff/2/sigmasquared));
        else
          gc->setNeighbors(h*W + w, h2*W + w2,1000*potts_weight*exp(-diff/2/sigmasquared)/sqrt(2.0));
      }
      
    }
  printf("sigmasquared is %.2f\n",sigmasquared);

  //printf("\nBefore optimization energy is %d\n",gc->compute_energy());
  //printf("max_iter: %d\n",max_iter);
  gc->expansion(max_iter);// run expansion for max_iter. For swap use gc->swap(num_iterations);
  //printf("\nAfter optimization energy is %d\n",gc->compute_energy());

  for(int h=0;h<H;h++){
    for(int w=0;w<W;w++){
      gc_segmentation[h*W+w] = gc->whatLabel(h*W+w);
    }
  }
  printf("smoothenergy is %.2f\n", double(gc->giveSmoothEnergy())/1000/potts_weight);
  delete gc;
  delete [] smooth;

}

template <typename Dtype>
void GraphCutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                   const vector<bool> & propagate_down,
                                   const vector<Blob<Dtype>*> & bottom)
{
    
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image inputs.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to segmentations.";
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(GraphCutLayer);
#endif

INSTANTIATE_CLASS(GraphCutLayer);
REGISTER_LAYER_CLASS(GraphCut);

} // namespace caffe
