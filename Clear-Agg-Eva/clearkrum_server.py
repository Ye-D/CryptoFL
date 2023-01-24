from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_float
from Common.Handler.handler import Handler

import Common.config as config

import numpy as np


class ClearKrumServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearKrumServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

    def UpdateGrad_float(self, request, context): # using Krum
        data_dict = {request.id: request.grad_ori}
        print("have received:", data_dict.keys())
        rst = super().process(dict_data=data_dict, handler=self.handler.computation)
        return GradResponse_float(grad_upd=rst)


class KrumGradientHandler(Handler):
    def __init__(self, num_workers, f):
        super(KrumGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.f = f
        self.m =num_workers - f - 1

    def computation(self, data_in):
        grad_in = np.array(data_in).reshape((self.num_workers, -1))
        distance = [list() for _ in range(self.num_workers)]
        for i in range(self.num_workers):
            for j in range(i+1, self.num_workers):
                tmp = np.linalg.norm( (grad_in[i]-grad_in[j]), ord=2)
                distance[i].append(tmp)
                distance[j].append(tmp)
        score = []
        for i in range(self.num_workers):
            score.append(sum(np.sort(distance[i])[:self.m]))

        #idx = score.index(min(score))
        idx = np.argpartition(score, self.m)
        grad_agg = np.mean(grad_in[idx[:self.m]], axis=0)
        return grad_agg.tolist()


if __name__ == "__main__":
    gradient_handler = KrumGradientHandler(num_workers=config.num_workers, f = config.f)

    krum_server = ClearKrumServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    krum_server.start()
