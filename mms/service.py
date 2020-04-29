import logging
import mxnet as mx
import os
import pickle


class ModelService(object):
    """
    A base Model handler implementation.
    """
    def __init__(self):
        self.ctx = None
        self.model = None
        self.initialized = False

    def initialize(self, context):
        logging.info("INITIALIZE")
        sys_prop = context.system_properties
        gpu_id = sys_prop.get("gpu_id")
        model_dir = sys_prop.get("model_dir")
        self.ctx = [mx.gpu(gpu_id)]
        self.model = mx.gluon.SymbolBlock.imports(
            os.path.join(model_dir, 'model-symbol.json'),
            ['data'],
            os.path.join(model_dir, 'model-0000.params'),
            ctx=self.ctx)
        self.initialized = True

    def inference(self, data):
        logging.info('INFERENCE')
        data = pickle.loads(data[0]['data'])
        data = data.as_in_context(self.ctx[0]).expand_dims(axis=0)
        output = self.model(data).asnumpy()
        if output.ndim < 3:
            return output.tolist()
        else:
            return [pickle.dumps(output)]


def handle(data, context):
    logging.info('HANDLER')

    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        probs = svc.inference(data)
        return probs


svc = ModelService()
