#ifndef HTP_VAE_H
#define HTP_VAE_H

#include "htp-common.h"

bool htp_kparams_solve_vae(const ggml_hexagon_session * sess, const ggml_tensor * op, void * kparams);

#endif // HTP_VAE_H
