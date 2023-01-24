#include "cryptofl_aby.h"


#include <vector>

ABYParty *cryptofl_party;



void init_cryptofl_aby(std::string address, uint16_t port, bool role_is_server) {
    e_role role;
    if (role_is_server) {
        role = (e_role) 0;
    } else {
        role = (e_role) 1;
    }

    cryptofl_party = new ABYParty(role, address, port, LT, gbitlen, nthreads, mt_alg);
    std::cout << "init cryptofl success!" << std::endl;
}


void shutdown_cryptofl_aby() {
    delete cryptofl_party;
    std::cout << "shutdown cryptofl successs!" << std::endl;
}


void reset_cryptofl_aby() {
    cryptofl_party->Reset();
}
