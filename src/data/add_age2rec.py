# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import numbers
import os
import random
import sys
import time
import traceback

import cv2
import numpy as np
import mxnet as mx

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


class AgeEstimator():
    def __init__(self, ctx, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec)==2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading',prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)

    def predict_age(self, image):
        input_blob = np.expand_dims(np.transpose(image, (2,0,1)), axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        result = self.model.get_outputs()[0].asnumpy()
        a = result[:,2:202].reshape( (100,2) )
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return age


def encode_new_record(age_model, idx, s_in, q_out):
    header_in, img = mx.recordio.unpack(s_in)
    if header_in.flag>0:
        header_out = mx.recordio.IRHeader(header_in.flag, header_in.label, idx, 0)
        s_out = mx.recordio.pack(header_out, '')
        q_out.put((idx, s_out))
    else:
        age = age_model.predict_age(mx.image.imdecode(img).asnumpy())
        if isinstance(header_in.label, list):
            label = header_in.label + float(age)
        elif isinstance(header_in.label, numbers.Number):
            label = [header_in.label, float(age)]
        else:
            raise
        header_out = mx.recordio.IRHeader(header_in.flag, label, idx, 0)
        s_out = mx.recordio.pack(header_out, img)
        q_out.put((idx, s_out))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Add age label to .rec file by age estimation model')

    parser.add_argument('--input-prefix', type=str, required=True,
                        help='Prefix of the .idx and .rec file to be added with age label')
    parser.add_argument('--output-prefix', type=str, required=True,
                        help='Prefix of the output .idx and .rec file added with age label')
    parser.add_argument('--device', type=str, default='0',
                        help="Available gpu list, e.g. 0")
    parser.add_argument('--age-model', type=str, required=True,
                        help="Age model, e.g. 'models/age-estimation/gamodel-r50/model,0'")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    ctx = []
    cvd = args.device.strip()
    if len(cvd)>0:
        for i in xrange(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx)==0:
        ctx = [mx.cpu()]
        print('use cpu')
    
    age_model = AgeEstimator(ctx, args.age_model, 'fc1')

    path_input_imgidx = args.input_prefix + '.idx'
    path_input_imgrec = args.input_prefix + '.rec'
    assert(os.path.isfile(path_input_imgidx))
    assert(os.path.isfile(path_input_imgrec))
    input_imgrec = mx.recordio.MXIndexedRecordIO(path_input_imgidx, path_input_imgrec, 'r')  # pylint: disable=redefined-variable-type

    path_output_imgrec = args.output_prefix + '.rec'
    path_output_imgidx = args.output_prefix + '.idx'
    output_imgrec = mx.recordio.MXIndexedRecordIO(path_output_imgidx, path_output_imgrec, 'w')
    
    try:
        import Queue as queue
    except ImportError:
        import queue
    q_out = queue.Queue()

    s = input_imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag>0:
        print('header0 label', header.label)
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = range(1, int(header.label[0]))
        id2range = {}
        seq_identity = range(int(header.label[0]), int(header.label[1]))

        encode_new_record(age_model, 0, s, q_out)
        idx, s_out = q_out.get()
        output_imgrec.write_idx(idx, s_out)

        for identity in seq_identity:
            s_in = input_imgrec.read_idx(identity)
            encode_new_record(age_model, 0, s_in, q_out)
            idx, s_out = q_out.get()
            output_imgrec.write_idx(idx, s_out)
            
    cnt = 0
    pre_time = time.time()
    for idx in imgidx:
        s_in = input_imgrec.read_idx(idx)
        encode_new_record(age_model, idx, s_in, q_out)
        idx, s_out = q_out.get()
        output_imgrec.write_idx(idx, s_out)
        if cnt % 1000 == 0:
            cur_time = time.time()
            print('time:', cur_time - pre_time, ' count:', cnt)
            pre_time = cur_time
        cnt += 1

    if cnt == 0:
        print('Did not find and list file with prefix %s' % args.prefix)
