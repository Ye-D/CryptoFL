from Common.Handler.handler import Handler
import Common.config as config
import numpy as np

import cppimport
import cppimport.import_hook
import Lib.CryptoFL_lib.CryptoFL as m


class CryptoFLHandler(Handler):

    def __init__(self, address, grad_port, role, num_workers, f):
        super(CryptoFLHandler, self).__init__()
        self.address = address
        self.grad_port = grad_port
        self.role = role
        self.num_workers = num_workers
        self.f = f

   

    def computation_grad(self, data_in):

        rst = np.array(data_in).reshape((self.num_workers, -1)).sum(axis=0)
        return rst

    

    def init_CryptoFL_aby(self):
        m.init_CryptoFL_aby(self.address, self.grad_port, self.role)

    def shutdown_CryptoFL_aby(self):
        m.shutdown_CryptoFL_aby()
