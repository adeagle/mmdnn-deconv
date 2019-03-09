import tensorflow as tf
import caffe
import numpy as np


x_data = np.random.uniform(-4, 4, [1, 608, 608, 3])


def run_pb(pb_model):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(pb_model, 'rb') as pb_f:
        graph_def.ParseFromString(pb_f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
        for n in graph.get_operations():
            print(n.name)
        x = tf.get_default_graph().get_tensor_by_name('input:0')
        training = tf.get_default_graph().get_tensor_by_name('train_flag:0')
        output = tf.get_default_graph().get_tensor_by_name('a:0')
        
        #x_shape = x.shape
        #x_data = np.random.uniform(-0.01, 0.01, x_shape)

        config = tf.ConfigProto()
        #config.mlu_options.fusion = True        
        with tf.Session() as sess:
            out = sess.run(output, feed_dict={x: x_data, training: False})
            print("tf:",out.shape)
            return out


def run_caffe(model, weights):
    net = caffe.Net(model, weights, caffe.TEST)
    net.blobs['Placeholder'].data[...] = np.transpose(x_data, [0, 3, 1, 2])
    #net.blobs['train_flag'].data[...] = False

    out = net.forward()
    for b in out:
        print(b,net.blobs[b].data.shape)
        
    res = net.blobs['conv2d_transpose'].data
    print(res.shape)
    return res


if __name__ == '__main__':
    tf_res = run_pb('model.pb') #nhwc 
    caffe_res = run_caffe('caffe_model.prototxt', 'caffe_model.caffemodel') #nchw
    
    print("tf data:",np.sum(tf_res))
    print("data data:",np.sum(np.transpose(caffe_res, [0, 2, 3, 1])))
    print(np.max(np.abs(tf_res - np.transpose(caffe_res, [0, 2, 3, 1]))))
    print(tf_res.shape)
    #print(np.sum(tf_res) - np.sum(caffe_res))
