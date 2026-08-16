# SPDX-License-Identifier: Apache-2.0
#
# nn_sectun — encrypted session layer (reuses the provisioned X25519 identity).
# Only meaningful where there's Wi-Fi + the provisioning keys: the C6.

set(srcs "")
if(CONFIG_NN_FEATURE_WIFI AND CONFIG_NN_FEATURE_BLE)
    set(srcs "src/nn_sectun.c")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "include"
    REQUIRES "nn_prov"
    PRIV_REQUIRES "lwip"
)
