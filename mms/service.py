import argparse
import io
import logging
import mxnet as mx
import os

import rsna_heme as rsna

class ModelService(object):
    """
    A base Model handler implementation.
    """
    def __init__(self):
        args = argparse.Namespace()
        args.model_name = 'resnet101_v1d'
        args.pretrained = False
        args.classes = 6
        args.wl = [(40, 80), (80, 200), (40, 380)]
        self.args = args

        self.model = None
        self.initialized = False

    def initialize(self, context):
        logging.info("INITIALIZE")

        sys_prop = context.system_properties
        gpu_id = sys_prop.get("gpu_id")
        model_dir = sys_prop.get("model_dir")

        self.args.ctx = [mx.gpu(gpu_id)]

        self.model = rsna.cnn.get_model(self.args)
        self.model.load_parameters(os.path.join(model_dir, 'params.params'))
        self.model.hybridize()

        self.initialized = True

    def preprocess(self, data):
        logging.info('PREPROCESS')
        f = io.BytesIO(data[0]['data'])
        dcm = rsna.dicom.Dicom(f)
        img = dcm.img_for_plot3(self.args.wl)
        img, _ = rsna.transforms.common_transform(mx.nd.array(img), 0)
        data = mx.gluon.data.SimpleDataset([(img, 0)])
        data = data.transform_first(rsna.transforms.train_transform)
        return data

    def inference(self, data):
        logging.info('INFERENCE')
        data = data[0][0].as_in_context(self.args.ctx[0])
        data = mx.nd.expand_dims(data, 0)
        output = self.model(data)
        probs = mx.nd.sigmoid(output).asnumpy().tolist()
        return probs


def handle(data, context):
    logging.info('HANDLER')

    if not svc.initialized:
        svc.initialize(context)
    
    if data is not None:
        data = svc.preprocess(data)
        probs = svc.inference(data)
        return probs


svc = ModelService()
