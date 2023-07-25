/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"

#if defined(NALU_HYPRE_USING_ROCTX)
#include "hip/hip_runtime_api.h"
#include "roctracer/roctx.h"

#elif defined(NALU_HYPRE_USING_NVTX)

#include <string>
#include <algorithm>
#include <vector>
#include "nvToolsExt.h"
#include "nvToolsExtCudaRt.h"

/* 16 named colors by HTML 4.01. Repalce white with Orange */
typedef enum
{
   /* White, */
   Orange,
   Silver,
   Gray,
   Black,
   Red,
   Maroon,
   Yellow,
   Olive,
   Lime,
   Green,
   Aqua,
   Teal,
   Blue,
   Navy,
   Fuchsia,
   Purple
} color_names;

static const uint32_t colors[] =
{
   /* 0xFFFFFF, */
   0xFFA500,
   0xC0C0C0,
   0x808080,
   0x000000,
   0xFF0000,
   0x800000,
   0xFFFF00,
   0x808000,
   0x00FF00,
   0x008000,
   0x00FFFF,
   0x008080,
   0x0000FF,
   0x000080,
   0xFF00FF,
   0x800080
};

static const NALU_HYPRE_Int nalu_hypre_nvtx_num_colors = sizeof(colors) / sizeof(uint32_t);
static std::vector<std::string> nalu_hypre_nvtx_range_names;

#endif // defined(NALU_HYPRE_USING_NVTX)

void nalu_hypre_GpuProfilingPushRangeColor(const char *name, NALU_HYPRE_Int color_id)
{
#if defined (NALU_HYPRE_USING_NVTX)
   color_id = color_id % nalu_hypre_nvtx_num_colors;
   nvtxEventAttributes_t eventAttrib = {0};
   eventAttrib.version = NVTX_VERSION;
   eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
   eventAttrib.colorType = NVTX_COLOR_ARGB;
   eventAttrib.color = colors[color_id];
   eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
   eventAttrib.message.ascii = name;
   nvtxRangePushEx(&eventAttrib);

#elif defined (NALU_HYPRE_USING_ROCTX)
   roctxRangePush(name);

#endif
}

void nalu_hypre_GpuProfilingPushRange(const char *name)
{
#if defined (NALU_HYPRE_USING_NVTX)
   std::vector<std::string>::iterator p = std::find(nalu_hypre_nvtx_range_names.begin(),
                                                    nalu_hypre_nvtx_range_names.end(),
                                                    name);

   if (p == nalu_hypre_nvtx_range_names.end())
   {
      nalu_hypre_nvtx_range_names.push_back(name);
      p = nalu_hypre_nvtx_range_names.end() - 1;
   }

   NALU_HYPRE_Int color = p - nalu_hypre_nvtx_range_names.begin();

   nalu_hypre_GpuProfilingPushRangeColor(name, color);

#elif defined (NALU_HYPRE_USING_ROCTX)
   roctxRangePush(name);

#endif
}

void nalu_hypre_GpuProfilingPopRange(void)
{
#if defined (NALU_HYPRE_USING_NVTX)
   nalu_hypre_GpuProfilingPushRangeColor("StreamSync0", Red);
   cudaStreamSynchronize(0);
   nvtxRangePop();
   nvtxRangePop();

#elif defined (NALU_HYPRE_USING_ROCTX)
   roctxRangePush("StreamSync0");
   hipStreamSynchronize(0);
   roctxRangePop();
   roctxRangePop();
#endif
}
