#include "utils.h"

#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#define MIN(a,b) (a) < (b) ? (a) : (b)
#define MAX(a,b) (a) > (b) ? (a) : (b)


__global__ void maxPool(float *ptrinput, float *ptroutput, const int isize1, const int isize2, const int outsize1, const int outsize2, const int nOutputPlane, const int poolH, const int poolW, const int pooldH, const int pooldW, const int batchsize)
{
	// each thread does a pixel of the output
	const int pixi = blockIdx.x;
	const int pixj = blockIdx.y;
	const int bidx = blockIdx.z*blockDim.z+threadIdx.z;
	if(bidx>=batchsize) return;

	int i,j,k;

	// move pointers
	ptrinput   += (pixi * pooldH * isize2 + pixj * pooldW) * nOutputPlane + bidx*isize1*isize2*nOutputPlane;
	ptroutput  += (pixi * outsize2 + pixj) * nOutputPlane  + bidx*outsize1*outsize2*nOutputPlane;
	const int stridej = nOutputPlane;
	const int stridei = (isize2 - poolW) * nOutputPlane;
//	const int stridek = (isize1 - poolH) * isize2 * nOutputPlane;
	float * ptrinputsave = ptrinput;


	for(k=threadIdx.x; k<nOutputPlane; k+=blockDim.x) {
		float out=-2e38; 
		for(i=0; i<poolH; i++) {
			for(j=0; j<poolW; j++) {
				out=MAX(out, ptrinput[k]);
				ptrinput += stridej;
			}
			ptrinput += stridei;
		}
		ptroutput[k]=out;
		ptrinput =ptrinputsave;
	}	

}



__global__ void maxPoolBackward(float *ptrinput, float *ptroutput, float *ptrgradinput, float *ptrgradoutput, const int isize1, const int isize2, const int outsize1, const int outsize2, const int nOutputPlane, const int poolH, const int poolW, const int pooldH, const int pooldW, const int batchsize)
{

	// this one is a bit tricky : we have to add up the gradient if the pooling overlaps...
	// so each block (each thread ?) will do one pixel of the input...
	// 1) find which outputs are related to the input
	// 2) go

	const int pixi = blockIdx.x;
	const int pixj = blockIdx.y;
	const int bidx = blockIdx.z*blockDim.z+threadIdx.z;

	__shared__ int _imin, _jmin, _imax, _jmax;
	int imin, jmin, imax, jmax;
	if(threadIdx.z==0)
	{
      imin=(pixi - (poolH - 1) + (pooldH -1))/pooldH > 0 ? (pixi - (poolH - 1) + (pooldH -1))/pooldH : 0 ;
      jmin=(pixj - (poolW - 1) + (pooldW -1))/pooldW > 0 ? (pixj - (poolW - 1) + (pooldW -1))/pooldW : 0 ;
      imax= pixi / pooldH < outsize1 ? pixi / pooldH : outsize1 - 1 ;
      jmax= pixj / pooldW < outsize2 ? pixj / pooldW : outsize2 - 1 ;
		if(threadIdx.x==0)
		{
			_imin=imin;
			_jmin=jmin;
			_imax=imax;
			_jmax=jmax;
		}
	}
	
	__syncthreads();

	if(bidx>=batchsize) return;

	if(threadIdx.z>0)
	{
		if(threadIdx.x==0)
		{
			imin=_imin;
			jmin=_jmin;
			imax=_imax;
			jmax=_jmax;
		}
		imin=__shfl(imin,0);
		jmin=__shfl(jmin,0);
		imax=__shfl(imax,0);
		jmax=__shfl(jmax,0);
	}

	int i,j,k;

	// move pointers
	ptrinput   += (pixi * isize2 + pixj) * nOutputPlane + bidx*isize1*isize2*nOutputPlane ;
	ptrgradinput   += (pixi * isize2 + pixj) * nOutputPlane + bidx*isize1*isize2*nOutputPlane ;
	ptroutput  += (imin * outsize2 + jmin) * nOutputPlane  + bidx*outsize1*outsize2*nOutputPlane ;
	ptrgradoutput  += (imin * outsize2 + jmin) * nOutputPlane  + bidx*outsize1*outsize2*nOutputPlane ;
	float * ptroutputsave = ptroutput;
	float * ptrgradoutputsave = ptrgradoutput;
	
	const int stridej = nOutputPlane;
	const int stridei = (outsize2 -jmax+jmin-1) * nOutputPlane;
//	const int stridek = (imax+imin-1 ) * outsize2 * nOutputPlane; // this one just brings the pointer back to where it was...

	for(k=threadIdx.x; k<nOutputPlane; k+=blockDim.x) {
		float pixvalue=ptrinput[k];
		float gradinputvalue=0;
		for(i=imin; i<imax+1; i++) {
			for(j=jmin; j<jmax+1; j++) {
				float out=ptroutput[k];
				if(pixvalue==out) {
//					ptrgradinput[k*blk+tidx] += ptrgradoutput[k*blk+tidx];
					gradinputvalue += ptrgradoutput[k];
				}
				ptroutput += stridej;
				ptrgradoutput += stridej;
			}
			ptroutput += stridei;
			ptrgradoutput += stridei;
		}
		ptrgradinput[k]=gradinputvalue;
		ptroutput = ptroutputsave;
		ptrgradoutput = ptrgradoutputsave;
	}	


}




static int cunxn_SpatialMaxPooling_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long poolW = luaT_getfieldcheckint(L, 1, "poolW");
  long poolH = luaT_getfieldcheckint(L, 1, "poolH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");


  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(state, input);

  // find the size of kernelslices
  long bs     = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long isize3 = input->size[3];
  //assert(isize3%32 == 0);

  long outsize1 = (isize1 - poolH) / dH + 1;
  long outsize2 = (isize2 - poolW) / dW + 1;

  THCudaTensor_resize4d(state, output, bs, outsize1, outsize2, isize3);

  float* ptroutput  = THCudaTensor_data(state, output);
  float* ptrinput   = THCudaTensor_data(state, input);


  // cuda blocks & threads:
  dim3 blocks (outsize1, outsize2, (bs+3)/4);
  dim3 threads (32,1,4);

  maxPool<<<blocks,threads>>>(ptrinput, ptroutput, isize1, isize2, outsize1, outsize2, isize3, poolH, poolW, dH, dW, bs);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in maxPool: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }


  // final cut:
  THCudaTensor_free(state, input); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}





static int cunxn_SpatialMaxPooling_updateGradInput(lua_State *L)
{  
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  long poolW = luaT_getfieldcheckint(L, 1, "poolW");
  long poolH = luaT_getfieldcheckint(L, 1, "poolH");


  long bs     = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long isize3 = input->size[3];

  long outsize1 = output->size[1];
  long outsize2 = output->size[2];

  THCudaTensor_resizeAs(state, gradInput, input);

  dim3 blocks (isize1, isize2, (bs+7)/8);
  dim3 threads (32,1,8);

  float* ptroutput  = THCudaTensor_data(state, output);
  float* ptrinput   = THCudaTensor_data(state, input);
  float* ptrgradoutput  = THCudaTensor_data(state, gradOutput);
  float* ptrgradinput   = THCudaTensor_data(state, gradInput);


  maxPoolBackward <<<blocks,threads>>>(ptrinput, ptroutput, ptrgradinput, ptrgradoutput, isize1, isize2, outsize1, outsize2, isize3,  poolH, poolW, dH, dW, bs);
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in maxPoolBackward: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}



static const struct luaL_Reg cunxn_SpatialMaxPoolingBHWD__ [] = {
  {"SpatialMaxPoolingBHWD_updateOutput", cunxn_SpatialMaxPooling_updateOutput},
  {"SpatialMaxPoolingBHWD_updateGradInput", cunxn_SpatialMaxPooling_updateGradInput},
  {NULL, NULL}
};

static void cunxn_SpatialMaxPoolingBHWD_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_SpatialMaxPoolingBHWD__, "nn");
  lua_pop(L,1);
}
