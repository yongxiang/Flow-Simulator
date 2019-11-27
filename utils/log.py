from torch.utils.tensorboard import SummaryWriter

def log_writer(args):
    label = args.pg_type
    label = '{}_{}'.format(label, args.ps) if args.ps != "" else label

    print('label: {}'.format(label))
    tb_writer = SummaryWriter('performance_log/{}_{}'.format(args.seed, label))

    return tb_writer, label
