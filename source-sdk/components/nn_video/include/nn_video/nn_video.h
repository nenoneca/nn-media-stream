/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <stdint.h>
#include "nn_link/nn_link.h"

/*
 * nn_video — H.264 frame transport over nn_link.
 *
 * A coded frame is larger than NN_LINK_MAX_PACKET, so it is fragmented: each
 * link packet is [nn_vid_hdr_t][payload].  The receiver reassembles by `seq`
 * and emits the whole frame once the END fragment (or all `nfrag`) arrive.
 */

#define NN_VID_MAGIC        0x56u   /* 'V' */
#define NN_VID_FLAG_KEY     0x01u   /* frame is an IDR/keyframe          */
#define NN_VID_FLAG_START   0x02u   /* first fragment of the frame       */
#define NN_VID_FLAG_END     0x04u   /* last  fragment of the frame       */

typedef struct __attribute__((packed)) {
    uint8_t  magic;     /* NN_VID_MAGIC                                   */
    uint8_t  flags;     /* NN_VID_FLAG_*                                  */
    uint16_t seq;       /* frame sequence number                         */
    uint16_t frag;      /* fragment index [0, nfrag)                     */
    uint16_t nfrag;     /* total fragments in this frame                 */
} nn_vid_hdr_t;

#define NN_VID_HDR_LEN      ((int)sizeof(nn_vid_hdr_t))      /* 8 */
#define NN_VID_MAX_PAYLOAD  (NN_LINK_MAX_PACKET - NN_VID_HDR_LEN)
