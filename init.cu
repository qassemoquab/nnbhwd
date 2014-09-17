#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "SpatialMaxPoolingBHWD.cu"
#include "SpatialConvolutionUnfold.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunnbhwd(lua_State *L);

int luaopen_libcunnbhwd(lua_State *L)
{
  lua_newtable(L);

  cunxn_SpatialMaxPoolingBHWD_init(L);
  cunxn_SpatialConvolutionUnfold_init(L);

  return 1;
}
