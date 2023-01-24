#ifndef __cryptofl_ABY_H__
#define __cryptofl_ABY_H__

#include "../ABY/src/abycore/sharing/sharing.h"
#include "../ABY/src/abycore/circuit/booleancircuits.h"
#include "../ABY/src/abycore/circuit/arithmeticcircuits.h"
#include "../ABY/src/abycore/circuit/circuit.h"
#include "../ABY/src/abycore/aby/abyparty.h"

#include "../ABY/src/abycore/ABY_utils/ABYconstants.h"

#include <vector>


//for ABY
const uint32_t secparam = 128;
const uint32_t nthreads = 1;
const e_mt_gen_alg mt_alg = MT_OT;
const uint32_t bitlen = 8;
const uint32_t gbitlen = 32;


void init_cryptofl_aby(std::string address, uint16_t port, bool role_is_server);
void shutdown_cryptofl_aby();

void reset_cryptofl_aby();

#endif
