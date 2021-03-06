#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nxn_(NAME) TH_CONCAT_3(nxn_, Real, NAME)

#include "generic/SpatialConvolutionUnfold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxPoolingBHWD.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libnnbhwd(lua_State *L);

int luaopen_libnnbhwd(lua_State *L)
{
  lua_newtable(L);

  nxn_FloatSpatialConvolutionUnfold_init(L);
  nxn_FloatSpatialMaxPoolingBHWD_init(L);

  nxn_DoubleSpatialConvolutionUnfold_init(L);
  nxn_DoubleSpatialMaxPoolingBHWD_init(L);

  return 1;
}
