import gluoncv
import mxnet as mx

from mxnet import nd, autograd, gluon

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

def forward_pass(net, loss_fcn, data, labels, weights):
    outputs = [net(X) for X in data]
    losses = [loss_fcn(yhat, y, w) for yhat, y, w in zip(outputs, labels, weights)]
    return [outputs, losses]

def process_data(net, loss_fcn, dataloader, ctx, trainer=None):
    metric_loss = mx.metric.Loss()
    metric_acc = mx.metric.Accuracy()
    for i, batch in enumerate(dataloader):
        weights = nd.ones_like(batch[1]) * nd.array([2, 1, 1, 1, 1, 1])
        weights = gluon.utils.split_and_load(weights,  ctx_list=ctx, batch_axis=0, even_split=False)
        data    = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels  = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        if trainer is not None:
            with autograd.record():
                [outputs, losses] = forward_pass(net, loss_fcn, data, labels, weights)
            for l in losses:
                l.backward()
            trainer.step(len(batch[0]))
        else:
            [outputs, losses] = forward_pass(net, loss_fcn, data, labels, weights)
        
        metric_loss.update(labels, losses)
        metric_acc.update([l[:,0] for l in labels], [(nd.sign(o[:,0]) + 1) / 2 for o in outputs])

        if dataloader.__class__.__name__ is 'tqdm' and (i < (len(dataloader) - 1)):
            dataloader.set_description(f'loss {metric_loss.get()[1]:.4f}')

    _, loss = metric_loss.get()
    _, acc = metric_acc.get()

    return [loss, acc]
