#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPoolingBHWD.c"
#else

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) < (Y) ? (Y) : (X))
#endif

static int nxn_(SpatialMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "poolW");
  int kH = luaT_getfieldcheckint(L, 1, "poolH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  /* find the sizes */
  long bs     = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long isize3 = input->size[3];

  long istr0  = input->stride[0];
  long istr1  = input->stride[1];
  long istr2  = input->stride[2];
  long istr3  = input->stride[3];

  long outsize1 = (isize1 - kH) / dH + 1;
  long outsize2 = (isize2 - kW) / dW + 1;

  
  THTensor_(resize4d)(output, bs, outsize1, outsize2, isize3);
  THTensor_(resize4d)(indices, bs, outsize1, outsize2, isize3);

  long ostr0  = output->stride[0];
  long ostr1  = output->stride[1];
  long ostr2  = output->stride[2];
  long ostr3  = output->stride[3];

  real* ptroutput  = THTensor_(data)(output);
  real* ptrinput   = THTensor_(data)(input);
  real* ptrindices  = THTensor_(data)(indices);
  
  long idx;
  #pragma omp parallel for private(idx)
  for (idx=0; idx<bs; idx++) /* loop on batch */
  {
     long yo, xo, yi, xi, ch, chb;
     real* curptrout;
     real* curptrinp;
     real* curptrind;
     
     for (yo=0; yo<outsize1; yo++) /* loop on output coordinates */
     {
        for (xo=0; xo<outsize2; xo++)
        {   
            curptrout=ptroutput+idx*ostr0+yo*ostr1+xo*ostr2;
            curptrind=ptrindices+idx*ostr0+yo*ostr1+xo*ostr2;
            for(chb=0; chb < (isize3+15)/16; chb++)
            {
               real maxval[16]={-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38,-2e38};
               real maxind[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
               for (yi=0; yi<kH; yi++)            /* loop on input */ 
               {
                  for (xi=0; xi<kW; xi++)
                  {
                      curptrinp=ptrinput+idx*istr0+(yo*dH+yi)*istr1+(xo*dW+xi)*istr2;
                      for(ch=chb*16; ch<isize3 && ch<(chb+1)*16; ch++) /* loop on batches of 16 feature maps */
                      {
                          real val = curptrinp[ch];
                          if(val > maxval[ch-chb*16])
                          {
                              maxval[ch-chb*16]=val;
                              maxind[ch-chb*16]=yi*kW+xi;
                          }
                      }
                  }
               }
               for(ch=chb*16; ch<isize3 && ch<(chb+1)*16; ch++)
               {
                  curptrout[ch]=maxval[ch-chb*16];
                  curptrind[ch]=maxind[ch-chb*16];
               }
            }
        }
     }
     
  }
  
  
  
  /* cleanup */
  /*THTensor_(free)(input);*/
  return 1;
}

static int nxn_(SpatialMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "poolW");
  int kH = luaT_getfieldcheckint(L, 1, "poolH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(fill)(gradInput, 0);

  real* ptrgradoutput  = THTensor_(data)(gradOutput);
  real* ptrgradinput   = THTensor_(data)(gradInput);
  real* ptrindices     = THTensor_(data)(indices);
  

   
  /* find the sizes */
  long bs      = gradInput->size[0];
  long gisize1 = gradInput->size[1];
  long gisize2 = gradInput->size[2];
  long gisize3 = gradInput->size[3];

  long gistr0  = gradInput->stride[0];
  long gistr1  = gradInput->stride[1];
  long gistr2  = gradInput->stride[2];
  long gistr3  = gradInput->stride[3];

  long goutsize1 = gradOutput->size[1];
  long goutsize2 = gradOutput->size[2];

  long gostr0  = gradOutput->stride[0];
  long gostr1  = gradOutput->stride[1];
  long gostr2  = gradOutput->stride[2];
  long gostr3  = gradOutput->stride[3];

  long idx;
  #pragma omp parallel for private(idx)
  for (idx=0; idx<bs; idx++) /* loop on batch */
  {
     long yo, xo, chb, ch, yi, xi;
     real* curptrgradout;
     real* curptrindices;
     for (yo=0; yo<goutsize1; yo++) /* loop on output coordinates */
     {
        for (xo=0; xo<goutsize2; xo++)
        {   
            curptrgradout=ptrgradoutput + idx*gostr0 + yo*gostr1 + xo*gostr2;
            curptrindices=ptrindices + idx*gostr0 + yo*gostr1 + xo*gostr2;
            for(chb=0; chb < (gisize3+15)/16; chb++)
            {
                real goval[16];
                real maxind[16];
                for(ch=chb*16; ch<gisize3 && ch<(chb+1)*16; ch++) /* loop on batches of 16 feature maps */
                {
                    goval[ch-chb*16]=curptrgradout[ch];
                    maxind[ch-chb*16]=curptrindices[ch];
                }
                for(ch=chb*16; ch<gisize3 && ch<(chb+1)*16; ch++) /* loop on batches of 16 feature maps */
                {
                     yi=(int)(maxind[ch-chb*16])/kW;
                     xi=(int)(maxind[ch-chb*16])%kW;
                     ptrgradinput[idx*gistr0+(yo*dH+yi)*gistr1+(xo*dW+xi)*gistr2+ch]+=goval[ch-chb*16];
                }                
            }           
        }
     }
  }

  /* cleanup */
  /*THTensor_(free)(gradOutput);*/

  return 1;
}

static const struct luaL_Reg nxn_(SpatialMaxPoolingBHWD__) [] = {
  {"SpatialMaxPoolingBHWD_updateOutput", nxn_(SpatialMaxPooling_updateOutput)},
  {"SpatialMaxPoolingBHWD_updateGradInput", nxn_(SpatialMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nxn_(SpatialMaxPoolingBHWD_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(SpatialMaxPoolingBHWD__), "nn");
  lua_pop(L,1);
}

#endif
