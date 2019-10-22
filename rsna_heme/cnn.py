import gluoncv
import mxnet as mx

def get_model(args):
    net = gluoncv.model_zoo.get_model(args.model_name, pretrained = args.pretrained)
    with net.name_scope():
        if hasattr(net, 'fc'):
            output_name = 'fc'
        elif hasattr(net, 'output'):
            output_name = 'output'
        setattr(net, output_name, mx.gluon.nn.Dense(args.classes))
    if hasattr(args, 'params_path'):
        net.load_parameters(args.params_path)
    elif args.pretrained is True:
        getattr(net, output_name).initialize(mx.init.Xavier(), ctx = args.ctx)
    else:
        net.initialize(mx.init.Xavier(), ctx = args.ctx)
    net.collect_params().reset_ctx(args.ctx)
    net.hybridize()
    return net